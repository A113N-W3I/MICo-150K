#!/usr/bin/env python3
"""
Modified from https://github.com/VectorSpaceLab/OmniGen2/blob/main/inference.py
"""
import dotenv
dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List

import torch
from PIL import Image, ImageOps
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading

from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


def parse_args():
    parser = argparse.ArgumentParser("OmniGen2 multi-image inference")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--transformer_lora_path", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default="euler", choices=["euler", "dpmsolver++"])
    parser.add_argument("--num_inference_step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--text_guidance_scale", type=float, default=5.0)
    parser.add_argument("--image_guidance_scale", type=float, default=2.0)
    parser.add_argument("--cfg_range_start", type=float, default=0.0)
    parser.add_argument("--cfg_range_end", type=float, default=1.0)

    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")

    parser.add_argument("--input_image_path", nargs="+", required=True)
    parser.add_argument("--output_image_path", type=str, required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)

    parser.add_argument("--enable_model_cpu_offload", action="store_true")
    parser.add_argument("--enable_sequential_cpu_offload", action="store_true")
    parser.add_argument("--enable_group_offload", action="store_true")

    return parser.parse_args()


def load_pipeline(args, accelerator, weight_dtype):
    pipe = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if args.transformer_path:
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )

    if args.transformer_lora_path:
        pipe.load_lora_weights(args.transformer_lora_path)

    if args.scheduler == "dpmsolver++":
        from omnigen2.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )

    if args.enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipe.transformer, accelerator.device, "block_level", 2)
        apply_group_offloading(pipe.mllm, accelerator.device, "block_level", 2)
        apply_group_offloading(pipe.vae, accelerator.device, "block_level", 2)
    else:
        pipe = pipe.to(accelerator.device)

    return pipe


def load_images(paths: List[str]) -> List[Image.Image]:
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        img = ImageOps.exif_transpose(img)
        images.append(img)
    return images


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.dtype if args.dtype != "fp32" else "no"
    )

    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    torch.manual_seed(args.seed)

    pipeline = load_pipeline(args, accelerator, weight_dtype)

    input_images = load_images(args.input_image_path)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=args.instruction,
        input_images=input_images,
        width=1024,
        height=1024,
        num_inference_steps=args.num_inference_step,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=args.negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    ).images

    os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)

    base, ext = os.path.splitext(args.output_image_path)
    for i, img in enumerate(results):
        save_path = f"{base}_{i:02d}{ext}"
        img.save(save_path)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()