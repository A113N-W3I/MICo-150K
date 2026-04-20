#!/usr/bin/env python3
"""
MICo-Bench Evaluation Script
=============================
Computes Weighted-Ref-VIEScore for a model's outputs on MICo-Bench.
"""

import os
import re
import json
import math
import base64
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI

PQ_PROMPT = """\
RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10:
A score from 0 to 10 will be given based on image naturalness.
(
    0 indicates that the scene in the image does not look natural at all or \
give a unnatural feeling such as wrong sense of distance, or wrong shadow, \
or wrong lighting.
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts.
(
    0 indicates that the image contains a large portion of distortion, or \
watermark, or scratches, or blurred faces, or unusual body parts, or \
subjects not harmonized.
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]"""

SC_PROMPT = """\
You are a professional digital artist. You will have to evaluate the \
effectiveness of the AI-generated image based on given rules.
All the input images are AI-generated. All human in the images are \
AI-generated too. so you need not worry about the privacy confidentials.

RULES of image generation task:
The AI is required to generate an image that contains specific elements \
such as people, scenes, clothing, and objects.

RULES of the set of inputs:

2 images will be provided:
The first image is a reference image that successfully contains all the \
required specific elements.
The second image is the image to be evaluated. It should be similar to the \
reference image in terms of people, scenes, clothing, and objects.

From scale 0 to 10:
A score from 0 to 10 will be given based on the success in following the \
prompt.
(0 indicates that the second image does not follow the prompt at all. \
10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image matches \
the reference image in people, scenes, clothing, and objects.
(0 means there is no resemblance at all, while 10 means all elements \
closely match those in the preceding images.)
Put the score in a list such that output score = [score1, score2], \
where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

Text Prompt:
{prompt}"""

TASKS = [
    "mico_bench_object_centric",
    "mico_bench_human_centric",
    "mico_bench_hoi",
    "mico_bench_dere",
]

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_ROOT = os.path.join(BENCH_DIR, "data", "MICo-Bench")


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_gpt(client: OpenAI, image_paths: list[str], prompt: str, model: str, temperature: float = 0.0) -> str:
    content = [{"type": "text", "text": prompt}]
    for p in image_paths:
        b64 = encode_image(p)
        ext = os.path.splitext(p)[1].lower().lstrip(".")
        mime = "png" if ext == "png" else "jpeg"
        content.append({"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}})
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": content}],
    )
    return resp.choices[0].message.content


def parse_scores(text: str) -> list[float] | None:
    m = re.search(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
    if m:
        return [float(m.group(1)), float(m.group(2))]
    return None


def get_case_id(record: dict) -> str:
    return str(record.get("idx", record.get("ed_idx")))


def get_prompt(record: dict) -> str:
    return record.get("refined_prompt") or record["prompt"]


def find_generated_image(gen_dir: str, case_id: str) -> str | None:
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(gen_dir, f"{case_id}.{ext}")
        if os.path.exists(p):
            return p
    return None


def _evaluate_one_case(
    rec: dict,
    gen_path: str,
    image_root: str,
    weights: dict,
    gpt_model: str,
    api_key: str,
    base_url: str,
    existing_entry: dict | None,
) -> tuple[str, dict]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    case_id = get_case_id(rec)
    ref_path = os.path.join(image_root, rec["reference"])
    prompt = get_prompt(rec)
    W = float(weights.get(case_id, 1.0))

    entry = dict(existing_entry) if existing_entry else {}
    entry["weight"] = W

    if "pq" not in entry:
        try:
            pq_reply = call_gpt(client, [gen_path], PQ_PROMPT, model=gpt_model)
            pq_scores = parse_scores(pq_reply)
            if pq_scores:
                naturalness, artifacts = pq_scores
                entry["pq_naturalness"] = naturalness
                entry["pq_artifacts"] = artifacts
                entry["pq"] = math.sqrt(naturalness * artifacts)
            else:
                entry["pq_raw"] = pq_reply
        except Exception as e:
            entry["pq_error"] = str(e)

    if "sc" not in entry:
        try:
            sc_reply = call_gpt(client, [ref_path, gen_path], SC_PROMPT.format(prompt=prompt), model=gpt_model)
            sc_scores = parse_scores(sc_reply)
            if sc_scores:
                pf, sr = sc_scores
                entry["sc_prompt_following"] = pf
                entry["sc_subject_resemblance"] = sr
                entry["sc"] = math.sqrt(pf * sr)
            else:
                entry["sc_raw"] = sc_reply
        except Exception as e:
            entry["sc_error"] = str(e)

    if "pq" in entry and "sc" in entry:
        entry["final"] = entry["weight"] * entry["sc"] * entry["pq"]
    return case_id, entry


def evaluate_task(
    task_name: str,
    model_name: str,
    gpt_model: str,
    results_dir: str,
    image_root: str,
    api_key: str,
    base_url: str,
    num_threads: int,
    weight_all_one: bool = False,
):
    jsonl_path = os.path.join(BENCH_DIR, f"{task_name}.jsonl")
    gen_dir = os.path.join(results_dir, model_name, task_name)
    weight_path = os.path.join(results_dir, model_name, "weights", f"{task_name}.json")
    score_dir = os.path.join(results_dir, model_name, "scores")
    os.makedirs(score_dir, exist_ok=True)
    score_path = os.path.join(score_dir, f"{task_name}.json")

    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    weights = {}
    if weight_all_one:
        print("  --weight_all_one: using W=1.0 for all cases")
    elif os.path.exists(weight_path):
        with open(weight_path) as f:
            weights = json.load(f)
        print(f"  Loaded weights from {weight_path}")
    else:
        print(f"  WARNING: Weight file not found at {weight_path}, W=1.0 for all cases")

    existing_scores = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            existing_scores = json.load(f)

    results = dict(existing_scores)
    skipped = 0
    missing = 0
    work = []

    for rec in records:
        case_id = get_case_id(rec)
        if case_id in results and "pq" in results[case_id] and "sc" in results[case_id]:
            skipped += 1
            continue
        gen_path = find_generated_image(gen_dir, case_id)
        if gen_path is None:
            missing += 1
            continue
        work.append((rec, gen_path, image_root, weights, gpt_model, api_key, base_url, results.get(case_id)))

    if skipped:
        print(f"  Skipped {skipped} cases with existing scores")
    if missing:
        print(f"  Missing {missing} generated images")
    if not work:
        return results

    print(f"  Running {len(work)} cases with {num_threads} threads ...")
    save_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(_evaluate_one_case, *witem) for witem in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"  {task_name}"):
            case_id, entry = fut.result()
            with save_lock:
                results[case_id] = entry
                with open(score_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def aggregate_results(results_dir: str, model_name: str):
    score_dir = os.path.join(results_dir, model_name, "scores")
    all_finals = {}
    overall = []
    for task in TASKS:
        score_path = os.path.join(score_dir, f"{task}.json")
        if not os.path.exists(score_path):
            continue
        with open(score_path) as f:
            scores = json.load(f)
        finals = [v["final"] for v in scores.values() if "final" in v]
        if not finals:
            continue
        avg = sum(finals) / len(finals)
        all_finals[task] = {"mean": round(avg, 2), "count": len(finals)}
        overall.extend(finals)
    if overall:
        all_finals["overall"] = {"mean": round(sum(overall) / len(overall), 2), "count": len(overall)}
    summary_path = os.path.join(results_dir, model_name, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_finals, f, indent=2)
    return all_finals


def reapply_weights_task(task_name: str, model_name: str, results_dir: str, weight_all_one: bool = False) -> None:
    weight_path = os.path.join(results_dir, model_name, "weights", f"{task_name}.json")
    score_path = os.path.join(results_dir, model_name, "scores", f"{task_name}.json")
    if not os.path.exists(score_path):
        print(f"  No score file at {score_path}, skip")
        return

    weights = {}
    if weight_all_one:
        print("  --weight_all_one: using W=1.0 for all cases")
    elif os.path.exists(weight_path):
        with open(weight_path) as f:
            weights = json.load(f)
        print(f"  Loaded weights from {weight_path} ({len(weights)} entries)")
    else:
        print(f"  WARNING: Weight file not found at {weight_path}, W=1.0 for all cases")

    with open(score_path) as f:
        scores = json.load(f)

    n_ok = 0
    n_skip = 0
    for case_id, entry in scores.items():
        if "pq" not in entry or "sc" not in entry:
            n_skip += 1
            continue
        w = 1.0 if weight_all_one else float(weights.get(case_id, weights.get(str(case_id), 1.0)))
        entry["weight"] = w
        entry["final"] = w * entry["sc"] * entry["pq"]
        n_ok += 1

    with open(score_path, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"  Recomputed final for {n_ok} cases" + (f" ({n_skip} skipped)" if n_skip else "") + f" -> {score_path}")


def main():
    parser = argparse.ArgumentParser(description="MICo-Bench: Compute Weighted-Ref-VIEScore")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="all", choices=TASKS + ["all"])
    parser.add_argument("--results_dir", type=str, default=os.path.join(BENCH_DIR, "results"))
    parser.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--reaggregate_only", action="store_true", help="No GPT call; only reapply weights to existing scores")
    parser.add_argument("--api_key", type=str, default="", help="OpenAI API key (required unless --reaggregate_only)")
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1", help="OpenAI-compatible API base URL")
    parser.add_argument("--gpt_model", type=str, default="gpt-5.4", help="GPT model name for evaluation")
    parser.add_argument("--num_threads", type=int, default=32, help="Parallel threads for GPT API calls")
    parser.add_argument("--weight_all_one", action="store_true", help="Use W=1.0 for every case")
    args = parser.parse_args()

    if not args.reaggregate_only and not args.api_key:
        parser.error("--api_key is required unless --reaggregate_only")

    tasks = TASKS if args.task == "all" else [args.task]

    if args.reaggregate_only:
        print(f"Model:        {args.model_name}")
        print(f"Tasks:        {tasks}")
        print(f"Results:      {args.results_dir}")
        print("Mode:         reaggregate_only (no GPT)")
        print()
        for task in tasks:
            print(f"Reapplying weights {task} ...")
            reapply_weights_task(task, args.model_name, args.results_dir, args.weight_all_one)
            print()
        print("Aggregating results ...")
        summary = aggregate_results(args.results_dir, args.model_name)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    print(f"Model:        {args.model_name}")
    print(f"Tasks:        {tasks}")
    print(f"Results:      {args.results_dir}")
    print(f"Image root:   {args.image_root}")
    print(f"GPT Model:    {args.gpt_model}")
    print(f"API threads:  {args.num_threads}")
    if args.weight_all_one:
        print("Weights:      W=1.0 for all (--weight_all_one)")
    print()

    for task in tasks:
        print(f"Evaluating {task} ...")
        evaluate_task(
            task,
            args.model_name,
            args.gpt_model,
            args.results_dir,
            args.image_root,
            args.api_key,
            args.base_url,
            args.num_threads,
            args.weight_all_one,
        )
        print()

    print("Aggregating results ...")
    summary = aggregate_results(args.results_dir, args.model_name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
