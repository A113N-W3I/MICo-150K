#!/usr/bin/env python3
"""
Modified from https://github.com/Alpha-VLLM/Lumina-DiMOO/blob/main/inference/inference_i2i.py
"""
import os
import json
import argparse
import time
from PIL import Image
import torch
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import (
    preprocess_image,
    decode_vq_to_image,
    calculate_vq_params,
    generate_crop_size_list,
    var_center_crop,
    add_break_line,
    encode_img_with_breaks
)
from generators.image_to_image_generator import generate_i2i
from utils.prompt_utils import generate_image_to_image_prompt, create_prompt_templates

def main():
    parser = argparse.ArgumentParser(description="Image-to-image or multi-image inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path or comma-separated list of images")
    parser.add_argument("--ref_image_path", type=str, default=None, help="Reference image for style transfer")
    parser.add_argument("--edit_type", type=str, default="canny_pred", help="Edit type")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="CFG scale")
    parser.add_argument("--cfg_img", type=float, default=4.0, help="Image CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_image_to_image", help="Output directory")

    args = parser.parse_args()

    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # <answer>
    EOA = SPECIAL_TOKENS["answer_end"]    # </answer>
    BOI = SPECIAL_TOKENS["boi"]           # <IMAGE>
    EOI = SPECIAL_TOKENS["eoi"]           # </IMAGE>

    # Random seed
    if args.seed != 0:
        setup_seed(args.seed)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model & tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    # Get prompt templates
    templates = create_prompt_templates()
    prompt_text = args.prompt
    edit_type = args.edit_type

    # Generate prompt triplets
    input_prompt, uncon_text, system_prompt = generate_image_to_image_prompt(prompt_text, edit_type, templates)

    # Handle "multi_inference" case separately
    if edit_type == "multi_inference":
        # Multiple images separated by comma
        img_paths = [p.strip() for p in args.image_path.split(",") if p.strip()]
        if len(img_paths) < 2:
            raise ValueError("For multi_inference, please provide at least 2 comma-separated image paths via --image_path")

        # Encode all input images to tokens
        all_img_tokens = []
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
            img = var_center_crop(img, crop_size_list=crop_size_list)
            tokens = encode_img_with_breaks(img, vqvae)
            all_img_tokens += [BOI] + tokens + [EOI]

        # Tokenize prompt text
        prompt_ids = tokenizer(input_prompt)["input_ids"]
        uncon_text_ids = tokenizer(uncon_text)["input_ids"]

        # Merge prompt and image tokens (like train.py multi-image structure)
        con_input = prompt_ids[:-1] + all_img_tokens + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + all_img_tokens + uncon_text_ids[-1:]
        uncon_input_image = prompt_ids  # unchanged

        # Build prediction tokens
        # Use last image size as reference
        last_img = Image.open(img_paths[-1]).convert("RGB")
        image_width, image_height = last_img.size
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width, vae_scale)
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)
        img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

        code_start = len(con_input) + 2
        con_input = torch.tensor(con_input + img_pred_token, device=device).unsqueeze(0)
        uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
        uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)

    else:
        # Single-image or ref-based editing (原逻辑保持不变)
        if "image_ref_transfer" in edit_type:
            input_img, input_ref = args.image_path, args.ref_image_path
            img = Image.open(input_img).convert("RGB")
            crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
            img = var_center_crop(img, crop_size_list=crop_size_list)
            img_token_input = encode_img_with_breaks(img, vqvae)
            input_image = input_ref
        else:
            input_image = args.image_path

        prompt_ids = tokenizer(input_prompt)["input_ids"]
        uncon_text_ids = tokenizer(uncon_text)["input_ids"]

        img = Image.open(input_image).convert("RGB")
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        img = var_center_crop(img, crop_size_list=crop_size_list)

        image_width, image_height = img.size
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
            image_height, image_width, vae_scale
        )

        input_img_token = encode_img_with_breaks(img, vqvae)

        if "image_ref_transfer" in edit_type:
            con_input = prompt_ids[:-1] + img_token_input + input_img_token + prompt_ids[-1:]
            uncon_input_text = uncon_text_ids[:-1] + img_token_input + input_img_token + uncon_text_ids[-1:]
        else:
            con_input = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
            uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
        uncon_input_image = prompt_ids

        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number=NEW_LINE)
        img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

        code_start = len(con_input) + 2
        con_input = torch.tensor(con_input + img_pred_token, device=device).unsqueeze(0)
        uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
        uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)

    # =============================
    #       Generation
    # =============================
    start_time = time.time()
    vq_tokens = generate_i2i(
        model,
        con_input,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.timesteps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_input_text,
        uncon_image=uncon_input_image,
        code_start=code_start,
    )

    # =============================
    #       Output Saving
    # =============================
    words = (prompt_text or "").split()
    filename_words = words[:10] if len(words) > 10 else words
    filename = "_".join(filename_words)
    filename = "".join(c for c in filename if c.isalnum() or c in ("_", "-"))
    filename = f"{filename}_{args.height}x{args.width}_t{args.timesteps}_cfg{args.cfg_scale}_cfgimg{args.cfg_img}_{edit_type}.png"
    save_path = os.path.join(args.output_dir, filename)

    out_img = decode_vq_to_image(
        vq_tokens,
        save_path,
        vae_ckpt=args.vae_ckpt,
        image_height=image_height,
        image_width=image_width,
        vqvae=vqvae,
    )
    out_img.save(save_path)
    print(f"[✓] Saved {save_path}")

    all_imgs = [Image.open(p).convert("RGB") for p in img_paths]
    all_imgs.append(out_img)

    max_h = max(img.height for img in all_imgs)
    total_w = sum(img.width for img in all_imgs)
    canvas = Image.new("RGB", (total_w, max_h), "white")

    x_offset = 0
    for img in all_imgs:
        if img.height != max_h:
            img = img.resize((int(img.width * max_h / img.height), max_h))
        canvas.paste(img, (x_offset, 0))
        x_offset += img.width
    
    concat_path = save_path.replace(".png", "_concat.png")
    canvas.save(concat_path)

    elapsed_time = time.time() - start_time
    print(f"[✓] Saved {concat_path} (Time {elapsed_time:.2f}s)")

if __name__ == "__main__":
    main()