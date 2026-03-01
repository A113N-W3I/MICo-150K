# Copyright 2025 MICo-150K Team. All rights reserved.

from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel, AutoencoderKLQwenImage
from transformers import AutoModel, AutoTokenizer, Qwen2VLProcessor
import torch
import argparse
from peft import PeftModel
from PIL import Image
import json
import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.insert(0, parent_dir)

from modeling_qwen_image import QwenImageEditPipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--lora", type=str, default=None)
    # parser.add_argument("--transformer", type=str, default=None)
    parser.add_argument("--image_paths", type=str, nargs='+', required=True, 
                       help="input image paths, separated by space")
    parser.add_argument("--prompt", type=str, required=True, 
                       help="editing prompt describing the desired changes to the input image(s)")
    parser.add_argument("--true_cfg_scale", type=float, default=4.0,
                       help="CFG scaling parameter for controlling the strength of the editing, default is 4.0")
    parser.add_argument("--seed", type=int, default=0,
                       help="Seed for random number generator, default is 0")
    parser.add_argument("--output_path", type=str, default="example_ours.png",
                       help="output image path, default is example_ours.png")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--neg_prompt", type=str, default="")
    args = parser.parse_args()

    # Automatically downloaded from Hugging Face Hub. 
    # You can also specify a local path to the model if you have already downloaded it.
    model_name = "kr-cen/Qwen-Image-MICo"
    use_safetensors = False
    
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='scheduler'
    )
    text_encoder = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='text_encoder',
        device_map=args.device_map, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='tokenizer',
    )
    processor = Qwen2VLProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='processor',
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='transformer', device_map=args.device_map, torch_dtype=torch.bfloat16, use_safetensors=use_safetensors
    )

    vae = AutoencoderKLQwenImage.from_pretrained(
        pretrained_model_name_or_path=model_name, subfolder='vae', torch_dtype=torch.bfloat16,
    ).to(transformer.device)

    if args.lora is not None:
        transformer = PeftModel.from_pretrained(transformer, args.lora)
        print(f'Load lora weights from {args.lora}.')

    pipe = QwenImageEditPipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                                 tokenizer=tokenizer, processor=processor, transformer=transformer)

    # Parse inputs
    image_paths = args.image_paths
    prompt = args.prompt
    true_cfg_scale = args.true_cfg_scale
    seed = args.seed
    output_path = args.output_path

    print(f"Input image paths: {image_paths}")
    print(f"Editing prompt: {prompt}")
    print(f"CFG scaling parameter: {true_cfg_scale}")
    print(f"Random seed: {seed}")
    print(f"Output path: {output_path}")
    # Load the input images
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # Execute the image editing pipeline
    image = pipe(
        images=images,
        height=args.height,
        width=args.width,
        prompt=prompt,
        negative_prompt=args.neg_prompt,
        num_inference_steps=25,
        true_cfg_scale=true_cfg_scale,
        generator=torch.manual_seed(seed)
    ).images[0]

    # Saving the edited image
    image.save(output_path)
    print(f"Image saved to: {output_path}")

"""
python infer/infer_qwenimage.py \
    --image_paths /PATH/TO/IMAGE_1 /PATH/TO/IMAGE_2 /PATH/TO/IMAGE_3 \
    --prompt "..." \
    --true_cfg_scale 4.0 \
    --seed 42 \
    --output_path /PATH/TO/OUTPUT_IMAGE.png
"""