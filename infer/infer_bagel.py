#!/usr/bin/env python3
"""
Modified from https://github.com/ByteDance-Seed/Bagel/blob/main/inference.ipynb
"""
import os
import argparse
import random
from typing import List

import numpy as np
import torch
from PIL import Image
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from inferencer import InterleaveInferencer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_paths", nargs="+", required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/notebook/data/group/ckr/ckpt/bagel")
    parser.add_argument("--ft_ema_ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_images(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    set_seed(args.seed)

    llm_config = Qwen2Config.from_json_file(
        os.path.join(args.model_path, "llm_config.json")
    )
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(args.model_path, "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(
        local_path=os.path.join(args.model_path, "ae.safetensors")
    )

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        device_map[k] = first_device

    ckpt_path = (
        args.ft_ema_ckpt
        if args.ft_ema_ckpt is not None
        else os.path.join(args.model_path, "ema.safetensors")
    )

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=ckpt_path,
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder="/tmp/offload",
    ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    inference_hyper = dict(
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    images = load_images(args.image_paths)
    input_lists = images + [args.prompt]

    outputs = inferencer.interleave_inference(
        input_lists=input_lists,
        **inference_hyper,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    outputs[0].save(args.output_path)


if __name__ == "__main__":
    main()