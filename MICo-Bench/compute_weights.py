#!/usr/bin/env python3
"""
MICo-Bench Step 1: Compute per-case preservation weights W.

For each source image:
  - human_indices positions: ArcFace cosine similarity vs. generated image; map max
    similarity to graded score (README bins).
  - other positions: local VLM yes/no whether the source element appears in the generated image.

W = (sum of element scores) / (number of source images)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE_ROOT = os.path.join(BENCH_DIR, "data", "MICo-Bench")

TASKS = [
    "mico_bench_object_centric",
    "mico_bench_human_centric",
    "mico_bench_hoi",
    "mico_bench_dere",
]

TASKS_SEQUENTIAL_ORDER = [
    "mico_bench_hoi",
    "mico_bench_human_centric",
    "mico_bench_object_centric",
    "mico_bench_dere",
]


def ensure_insightface_model_dir(root: str, name: str = "buffalo_l") -> str:
    root = os.path.expanduser(root)
    expected = os.path.join(root, "models", name)
    if os.path.isdir(expected):
        return root
    flat = os.path.join(root, name)
    if os.path.isdir(flat):
        os.makedirs(os.path.join(root, "models"), exist_ok=True)
        if not os.path.exists(expected):
            os.symlink(os.path.abspath(flat), expected, target_is_directory=True)
    return root


def create_face_app(ctx_id: int, arcface_root: str | None):
    from insightface.app import FaceAnalysis

    if arcface_root:
        arc_root = ensure_insightface_model_dir(arcface_root)
        app = FaceAnalysis(name="buffalo_l", root=arc_root)
    else:
        # Official behavior: auto-download buffalo_l into InsightFace cache.
        app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id)
    return app


VLM_PROMPT = """Given two images:
- Image 1 is a source image containing a specific element.
- Image 2 is an AI-generated composite image.

Does the element from Image 1 appear in Image 2?
Answer only "yes" or "no"."""


def get_case_id(rec: dict) -> str:
    return str(rec.get("idx", rec.get("ed_idx")))


def face_score_from_similarity(sim: float) -> float:
    if sim >= 0.45:
        return 1.0
    if sim >= 0.30:
        return 0.7
    if sim >= 0.15:
        return 0.5
    if sim >= 0.05:
        return 0.2
    return 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def largest_face_embedding(app, bgr: np.ndarray) -> np.ndarray | None:
    faces = app.get(bgr)
    if not faces:
        return None

    def area(f):
        x1, y1, x2, y2 = f.bbox
        return float((x2 - x1) * (y2 - y1))

    return max(faces, key=area).embedding


def face_preserved_single_source(app, src_path: str, gen_path: str) -> float:
    src = cv2.imread(src_path)
    gen = cv2.imread(gen_path)
    if src is None or gen is None:
        return 0.0
    emb_src = largest_face_embedding(app, src)
    if emb_src is None:
        return 0.0
    out_faces = app.get(gen)
    if not out_faces:
        return 0.0
    emb_src = np.asarray(emb_src, dtype=np.float32)
    best = max(cosine_sim(emb_src, f.embedding) for f in out_faces)
    return face_score_from_similarity(best)


def face_preserved_multi(app, src_paths: list[str], gen_path: str) -> float:
    gen = cv2.imread(gen_path)
    if gen is None:
        return 0.0

    input_embs = []
    for p in src_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            input_embs.append(None)
            continue
        input_embs.append(largest_face_embedding(app, bgr))

    if any(e is None for e in input_embs):
        return 0.0

    out_faces = app.get(gen)
    if not out_faces:
        return 0.0

    B = np.stack([f.embedding for f in out_faces], axis=0)
    B = np.asarray(B, dtype=np.float32)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)

    score_sum = 0.0
    for e in input_embs:
        a = np.asarray(e, dtype=np.float32).reshape(-1)
        a /= max(float(np.linalg.norm(a)), 1e-12)
        best = float(np.max(a @ B.T))
        score_sum += face_score_from_similarity(best)
    return score_sum


def parse_vlm_yes_no(text: str) -> bool | None:
    t = text.strip().lower()
    if re.search(r"\byes\b", t) and not re.search(r"\bno\b", t):
        return True
    if re.search(r"\bno\b", t) and not re.search(r"\byes\b", t):
        return False
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def load_vlm(model_path_or_repo: str):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLMoeForConditionalGeneration
    except ImportError as e:
        raise RuntimeError(
            "Need transformers with Qwen3-VL support (see Qwen3-VL README)."
        ) from e

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_path_or_repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path_or_repo, trust_remote_code=True)
    return model, processor


def _vlm_element_preserved_impl(model, processor, src_path: str, gen_path: str) -> int:
    import torch

    img1 = Image.open(src_path).convert("RGB")
    img2 = Image.open(gen_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = (
        inputs.to(device)
        if hasattr(inputs, "to")
        else {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    )
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    in_ids = inputs["input_ids"]
    gen_ids = out[0]
    new_tokens = gen_ids[len(in_ids[0]) :]
    text = processor.decode(new_tokens.cpu(), skip_special_tokens=True)
    yn = parse_vlm_yes_no(text)
    if yn is None:
        return 0
    return 1 if yn else 0


def find_generated_image(gen_dir: str, case_id: str) -> str | None:
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = os.path.join(gen_dir, f"{case_id}.{ext}")
        if os.path.isfile(p):
            return p
    return None


def compute_case_weight(
    rec: dict,
    gen_path: str,
    image_root: str,
    app,
    vlm_model,
    vlm_processor,
) -> float | None:
    inputs: list[str] = rec["input"]
    human_set = set(rec.get("human_indices") or [])
    n = len(inputs)
    if n == 0:
        return None

    preserved = 0.0
    human_indices_sorted = sorted(human_set)
    non_human_indices = [i for i in range(n) if i not in human_set]

    if len(human_indices_sorted) == 1:
        i = human_indices_sorted[0]
        src_p = os.path.join(image_root, inputs[i])
        preserved += face_preserved_single_source(app, src_p, gen_path)
    elif len(human_indices_sorted) > 1:
        src_paths = [os.path.join(image_root, inputs[i]) for i in human_indices_sorted]
        preserved += face_preserved_multi(app, src_paths, gen_path)

    for i in non_human_indices:
        src_p = os.path.join(image_root, inputs[i])
        preserved += _vlm_element_preserved_impl(vlm_model, vlm_processor, src_p, gen_path)

    return preserved / float(n)


def _worker_weights_multi_tasks(payload: dict) -> dict[str, dict[str, float]]:
    gid = str(payload["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gid

    print(f"[worker gpu={gid}] Loading ArcFace ...", flush=True)
    app = create_face_app(ctx_id=0, arcface_root=payload["arcface_root"])

    print(f"[worker gpu={gid}] Loading VLM ...", flush=True)
    vlm_model, vlm_processor = load_vlm(payload["vlm_model_path"])

    image_root = payload["image_root"]
    out_by_task: dict[str, dict[str, float]] = {}
    for seg in payload["segments"]:
        task_name = seg["task_name"]
        gen_root = seg["gen_root"]
        records = seg["records"]
        bucket = out_by_task.setdefault(task_name, {})
        n_done = 0
        for rec in records:
            cid = get_case_id(rec)
            gen_path = find_generated_image(gen_root, cid)
            if gen_path is None:
                continue
            try:
                w = compute_case_weight(rec, gen_path, image_root, app, vlm_model, vlm_processor)
            except Exception as e:
                print(f"[worker gpu={gid}] {task_name} case {cid}: {e}", file=sys.stderr, flush=True)
                continue
            if w is not None:
                bucket[cid] = round(w, 6)
                n_done += 1
        print(f"[worker gpu={gid}] {task_name}: processed {len(records)} assigned, {n_done} ok", flush=True)
    return out_by_task


def _split_records(records: list, n: int) -> list[list]:
    if n <= 1:
        return [records]
    chunks: list[list] = [[] for _ in range(n)]
    for i, rec in enumerate(records):
        chunks[i % n].append(rec)
    return chunks


def _load_task_state(bench_dir: str, results_dir: str, model_name: str, task_name: str) -> dict:
    jsonl_path = os.path.join(bench_dir, f"{task_name}.jsonl")
    gen_root = os.path.join(results_dir, model_name, task_name)
    out_path = os.path.join(results_dir, model_name, "weights", f"{task_name}.json")
    with open(jsonl_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    weights: dict[str, float] = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            weights = json.load(f)
    pending = [rec for rec in records if get_case_id(rec) not in weights]
    return {
        "task_name": task_name,
        "gen_root": gen_root,
        "out_path": out_path,
        "weights": weights,
        "pending": pending,
    }


def main():
    parser = argparse.ArgumentParser(description="MICo-Bench: compute W weights")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", type=str, default="all", choices=TASKS + ["all"])
    parser.add_argument("--results_dir", type=str, default=os.path.join(BENCH_DIR, "results"))
    parser.add_argument(
        "--bench_dir",
        type=str,
        default=BENCH_DIR,
        help="Directory containing *.jsonl annotation files",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help="Directory with benchmark source images; JSONL input paths are relative to this",
    )
    parser.add_argument(
        "--arcface_root",
        type=str,
        default="",
        help="Optional InsightFace root. Leave empty to use official auto-download/cache",
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Hugging Face repo id or local path for VLM",
    )
    parser.add_argument(
        "--arcface_ctx_id",
        type=int,
        default=0,
        help="GPU id for InsightFace when --num_workers=1",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Parallel worker processes (each worker loads ArcFace+VLM)",
    )
    parser.add_argument(
        "--worker_gpus",
        type=str,
        default="0,1,2",
        help="Comma-separated physical GPU ids for workers, e.g. 0,1,2",
    )
    args = parser.parse_args()

    print(f"Annotations: {args.bench_dir}")
    print(f"Image root:  {args.image_root}")
    if args.num_workers > 1:
        print(f"Workers:     {args.num_workers} (GPUs: {args.worker_gpus})")

    os.makedirs(os.path.join(args.results_dir, args.model_name, "weights"), exist_ok=True)

    tasks_to_run = list(TASKS_SEQUENTIAL_ORDER) if args.task == "all" else [args.task]
    task_states = [
        _load_task_state(args.bench_dir, args.results_dir, args.model_name, tn) for tn in tasks_to_run
    ]

    if not any(st["pending"] for st in task_states):
        for st in task_states:
            print(f"{st['task_name']}: all weights present ({len(st['weights'])}) -> {st['out_path']}")
        print("Done.")
        return

    gpu_list = [int(x.strip()) for x in args.worker_gpus.split(",") if x.strip()]
    while len(gpu_list) < max(1, args.num_workers):
        gpu_list.append(gpu_list[-1] if gpu_list else 0)
    gpu_list = gpu_list[: max(1, args.num_workers)]

    if args.num_workers <= 1:
        print("Loading ArcFace (buffalo_l) ...")
        app = create_face_app(ctx_id=args.arcface_ctx_id, arcface_root=args.arcface_root or None)

        print("Loading VLM ...")
        vlm_model, vlm_processor = load_vlm(args.vlm_model_path)

        for st in task_states:
            task_name = st["task_name"]
            pending = st["pending"]
            weights = st["weights"]
            out_path = st["out_path"]
            gen_root = st["gen_root"]

            if not pending:
                print(f"{task_name}: all weights present ({len(weights)}) -> {out_path}")
                continue

            for rec in tqdm(pending, desc=task_name):
                cid = get_case_id(rec)
                gen_path = find_generated_image(gen_root, cid)
                if gen_path is None:
                    continue
                try:
                    w = compute_case_weight(rec, gen_path, args.image_root, app, vlm_model, vlm_processor)
                except Exception as e:
                    print(f"  ERROR case {cid}: {e}", file=sys.stderr)
                    continue
                if w is not None:
                    weights[cid] = round(w, 6)

            with open(out_path, "w") as f:
                json.dump(weights, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(weights)} weights -> {out_path}")
    else:
        merged: dict[str, dict[str, float]] = {
            st["task_name"]: dict(st["weights"]) for st in task_states if st["pending"]
        }

        payloads: list[dict] = []
        for wi in range(args.num_workers):
            segments: list[dict] = []
            for st in task_states:
                pending = st["pending"]
                if not pending:
                    continue
                chunks = _split_records(pending, args.num_workers)
                chunk = chunks[wi]
                if not chunk:
                    continue
                segments.append(
                    {"task_name": st["task_name"], "gen_root": st["gen_root"], "records": chunk}
                )
            if not segments:
                continue
            payloads.append(
                {
                    "gpu_id": gpu_list[wi],
                    "segments": segments,
                    "image_root": args.image_root,
                    "arcface_root": args.arcface_root,
                    "vlm_model_path": args.vlm_model_path,
                }
            )

        if not payloads:
            print("No workers assigned (nothing pending).")
            print("Done.")
            return

        n_pending = sum(len(st["pending"]) for st in task_states)
        print(f"Spawning {len(payloads)} workers for {n_pending} total pending cases ...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(payloads)) as pool:
            parts = pool.map(_worker_weights_multi_tasks, payloads)

        for part in parts:
            for task_name, wmap in part.items():
                merged.setdefault(task_name, {})
                merged[task_name].update(wmap)

        for st in task_states:
            tn = st["task_name"]
            if tn not in merged:
                continue
            out_path = st["out_path"]
            with open(out_path, "w") as f:
                json.dump(merged[tn], f, indent=2, ensure_ascii=False)
            print(f"Saved {len(merged[tn])} weights -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
