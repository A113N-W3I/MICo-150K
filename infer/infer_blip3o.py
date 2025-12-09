#!/usr/bin/env python3
"""
Modified from https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT-edit/inference_editing.py
"""
import os
import sys
import argparse
import torch
import numpy as np
import random
from PIL import Image
from transformers import AutoTokenizer
from dataclasses import dataclass
import copy
from typing import Any, Optional

# Add BLIP3o modules to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from blip3o.constants import *
from blip3o.model import *
from torchvision.transforms import v2

target_transform = v2.Compose(
    [
        v2.Resize(1024),
        v2.CenterCrop(1024),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
    ]
    )

@dataclass
class EditingConfig:
    model_path: str = ""
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    # generation config
    scale: int = 0  
    seq_len: int = 729  
    top_p: float = 0.95
    top_k: int = 1200
    use_tar_siglip_features: bool = False
    use_und_image_vae: bool = False
    use_und_image_vae_as_noise: bool = False
    only_use_und_image_vae_as_noise: bool = False
    config: Optional[argparse.Namespace] = None

def set_global_seed(seed=42):
    """Set global random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_editing_prompt(instruction, has_input_image=True, num_input_images=0):
    """Create conversation template for image editing matching training format."""

    from blip3o.data.dataset import SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    if has_input_image:
        if num_input_images > 1:
        # For image editing - matches the training format from dataset.py:439
            user_message = f"Please edit these {num_input_images} images according to the following instruction: {instruction}.\n<image>"
        else:
            user_message = f"Please edit this image according to the following instruction: {instruction}.\n<image>"
    else:
        # For text-to-image generation
        user_message = f"Please generate image based on the following caption: {instruction}"
    
    messages.append({"role": "user", "content": user_message})
    
    return messages

class ImageEditingInference:
    def __init__(self, config: EditingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_models()
        
    def _load_models(self):
        print(f"Loading editing model from: {self.config.model_path}")
        
        # Load the editing model
        # need to double check what weights are loaded and what is not loaded here
        self.model, loading_info = blip3oQwenForInferenceLM.from_pretrained(
            self.config.model_path, 
            torch_dtype=self.config.dtype,
            output_loading_info=True
        )
        
        self.model.to(self.device)
        print("Missing keys:", loading_info["missing_keys"])
        print("Unexpected keys:", loading_info["unexpected_keys"])
        # print("Discarded keys:", loading_info["discarded_keys"])
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        print("✅ Model and tokenizer loaded successfully!")

    def prepare_inputs(self, messages, input_images=None):
        """Prepare inputs for the model, handling understanding images."""
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # input_text += f"<im_start><S{self.config.scale}>"
        input_text += f"<im_start><S{self.config.scale}>"
        
        # Handle understanding images (input images for editing)
        if input_images is not None and len(input_images) > 0:
            # Replace any <image> tokens in the user message with understanding image placeholders
            # Based on the training data format from dataset.py:102
            n_und_query = 729  # From dataset.py:207
            num_input_images = len(input_images)
            total_und_tokens = num_input_images * n_und_query
            
            # Create understanding image placeholder
            und_placeholder = "<|vision_start|>" + "<|image_pad|>" * total_und_tokens + "<|vision_end|>"
            
            # Replace <image> in the input text with understanding placeholder followed by generation image token
            input_text = input_text.replace("<image>", und_placeholder)
        
        return input_text

    def process_understanding_images(self, input_images):
        """Process input images for understanding (editing)."""
        if not input_images:
            return None
            
        processed_images = []
        for img in input_images:
            # Use the same processing as in the dataset
            if hasattr(self.model, 'get_vision_tower') and self.model.get_vision_tower() is not None:
                vision_tower = self.model.get_vision_tower()
                # Use the vision tower's image processor if available
                if hasattr(vision_tower, 'image_processor'):
                    processed = vision_tower.image_processor.preprocess(
                        img, return_tensors="pt"
                     )["pixel_values"][0]
                elif hasattr(self.model.config, 'image_processor'):
                    processed = self.model.config.image_processor.preprocess(
                        img, return_tensors="pt"
                    )["pixel_values"][0]
                else:
                    # Fallback processing
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    processed = transform(img)
            else:
                # Fallback processing
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                processed = transform(img)
            
            processed_images.append(processed.unsqueeze(0))  # Add batch dimension
        
        return processed_images

    def process_understanding_images_vae(self, input_images):
        """Process input images for understanding (editing) using UND image VAE."""
        if not input_images:
            return None
        
        processed_images = []
        for img in input_images:
            img = target_transform(img)
            img = img.to(dtype=next(self.model.model.sana_vae.parameters()).dtype, device=self.device).unsqueeze(0)
            latent = self.model.model.sana_vae.encode(img).latent
            if "shift_factor" in self.model.model.sana_vae.config and self.model.model.sana_vae.config.shift_factor is not None:
                latent = latent - self.model.model.sana_vae.config.shift_factor
            latent = latent * self.model.model.sana_vae.config.scaling_factor
            processed_images.append(latent)
        
        return processed_images
    def generate_image(self, instruction: str, input_images=None, seed=42) -> Image.Image:
        """
        Generate or edit an image based on instruction.
        
        Args:
            instruction: Text instruction for generation/editing
            input_images: List of PIL Images for editing (None for text-to-image)
        
        Returns:
            Generated/edited PIL Image
        """
        set_global_seed(seed)  # For reproducibility
        
        has_input_images = input_images is not None and len(input_images) > 0
        num_input_images = len(input_images) if input_images else 0
        
        # Create conversation messages
        messages = create_editing_prompt(instruction, has_input_images, num_input_images)
        
        # Prepare inputs
        input_text = self.prepare_inputs(messages, input_images)
        
        print(f"Input text: {input_text[:200]}...")
        print(f"Number of input images: {num_input_images}")
        
        # Tokenize
        inputs = self.tokenizer([input_text], return_tensors="pt", padding=True, truncation=True)
        
        # Process understanding images if provided - replace <|image_pad|> tokens with understanding image tokens
        if input_images is not None and len(input_images) > 0:
            print(f"✅ Processing {num_input_images} understanding images for multi-image editing")
            
            # Process understanding images to get tokens
            processed_images = self.process_understanding_images(input_images)
            if self.config.use_und_image_vae:
                print("✅ Processing understanding images for image editing using UND image VAE")
                processed_images_vae = self.process_understanding_images_vae(input_images)
                image_concat_vae = torch.cat(processed_images_vae, dim=0) if isinstance(processed_images_vae, list) else processed_images_vae
            else:
                image_concat_vae = None
            image_concat = torch.cat(processed_images, dim=0) if isinstance(processed_images, list) else processed_images
            
            # Get understanding image tokens from vision tower
            vision_features = self.model.encode_images(image_concat.to(self.device), None, pool_scale=1)
            und_image_tokens = vision_features['image_tokens'].flatten()
            
            if self.config.use_tar_siglip_features:
                und_image_siglip_features = vision_features['siglip_features'].to(dtype=torch.bfloat16)
                und_image_siglip_features = self.model.model.tar_siglip_features_connector(und_image_siglip_features)
                # keep the last two dimensions as num_tokens x hidden_size
                und_image_siglip_features = und_image_siglip_features.reshape(-1, und_image_siglip_features.shape[-1])
            
            # Replace <|image_pad|> tokens with understanding image tokens
            image_pad_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PAD_TOKEN)
            input_ids = inputs.input_ids.clone()
            
            und_image_tokens = und_image_tokens.to(input_ids.device)
            for batch_idx in range(input_ids.shape[0]):
                pad_indices = torch.where(input_ids[batch_idx] == image_pad_token_id)[0].to(input_ids.device)
                if len(pad_indices) > 0:
                    num_replacements = min(len(pad_indices), len(und_image_tokens))
                    if num_replacements > 0 and not self.config.use_tar_siglip_features:
                        input_ids[batch_idx, pad_indices[:num_replacements]] = und_image_tokens[:num_replacements]
                        print(f"✅ Replaced {num_replacements} padding tokens with understanding image tokens from {num_input_images} input images")

        # Generate using the model's generate_images method
        with torch.no_grad():
            # Use modified input_ids if understanding images were processed, otherwise use original
            final_input_ids = input_ids.to(self.device) if input_images else inputs.input_ids.to(self.device)
            
            if getattr(self.model.model, "use_tar_siglip_features", False):
                # Embed the input_ids to get input_embeds
                input_embeds = self.model.get_input_embeddings()(final_input_ids)
                # Find the <|image_pad|> token id
                image_pad_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PAD_TOKEN)
                # Get the indices of <|image_pad|> tokens in input_ids
                for batch_idx in range(final_input_ids.shape[0]):
                    pad_indices = torch.where(final_input_ids[batch_idx] == image_pad_token_id)[0].to(final_input_ids.device)
                    if len(pad_indices) > 0:
                        num_replacements = min(len(pad_indices), und_image_siglip_features.shape[0])
                        if num_replacements > 0:
                            # Replace the corresponding input_embeds with the image embeddings
                            input_embeds[batch_idx, pad_indices[:num_replacements], :] = und_image_siglip_features[:num_replacements].to(input_embeds.device)
                            print(f"✅ Replaced {num_replacements} input embeddings with image embeddings from {num_input_images} input images")
                
                # Call generate_images with input_embeds instead of input_ids
                print("🎨 Generating image with TAR SigLIP features...")
                gen_ids, output_image = self.model.generate_images(
                    final_input_ids,
                    inputs.attention_mask.to(self.device),
                    max_new_tokens=self.config.seq_len,
                    do_sample=True,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    input_embeds=input_embeds.to(self.device),
                    und_image_siglip_features=und_image_siglip_features.to(self.device),
                    und_image_vae_latents=image_concat_vae.to(self.device) if image_concat_vae is not None else None,
                )
            else:
                print("🎨 Generating image with image tokens...")
                gen_ids, output_image = self.model.generate_images(
                    final_input_ids,
                    inputs.attention_mask.to(self.device),
                    max_new_tokens=self.config.seq_len,
                    do_sample=True,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    und_image_vae_latents=image_concat_vae.to(self.device) if image_concat_vae is not None else None,
                )

        return output_image[0] if output_image else None

def main():
    parser = argparse.ArgumentParser(
        description="BLIP3o Multi-Image Editing Inference",
        epilog="""
            Examples:
            # Text-to-image generation
            python inference_editing.py -t "A beautiful sunset over mountains"
            
            # Single image editing
            python inference_editing.py -i input.jpg -t "Make the sky blue"
            
            # Multi-image editing
            python inference_editing.py -i input1.jpg input2.jpg input3.jpg -t "Combine these images"
        """
    )
    parser.add_argument("--model_path", "-m", 
                       default="",
                       help="Path to the trained editing model")
    parser.add_argument("--input_images", "-i", nargs="+", help="Paths to input images for editing (optional, multiple images supported)")
    parser.add_argument("--instruction", "-t", required=True, help="Editing/generation instruction")
    parser.add_argument("--output", "-o", help="Output image path (optional)")
    parser.add_argument("--device", "-d", default="cuda:0", help="Device to use")
    parser.add_argument("--use_tar_siglip_features", action="store_true", 
                       help="Use TAR SigLIP features for image generation (requires model with TAR SigLIP support)")
    parser.add_argument("--use_und_image_vae", action="store_true", 
                       help="Use UND image VAE for image generation (requires model with UND image VAE support)")
    parser.add_argument("--use_und_image_vae_as_noise", action="store_true", 
                       help="Use UND image VAE as noise for image generation (requires model with UND image VAE support)")
    parser.add_argument("--only_use_und_image_vae_as_noise", action="store_true", 
                       help="Only use UND image VAE as noise for image generation (requires model with UND image VAE support)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model path not found: {args.model_path}")
        return 1
    
    # Load input images if provided
    input_images = None
    if args.input_images:
        input_images = []
        for img_path in args.input_images:
            if not os.path.exists(img_path):
                print(f"❌ Error: Input image not found: {img_path}")
                return 1
            input_images.append(Image.open(img_path).convert('RGB'))
            print(f"📸 Loaded input image: {img_path}")
    
    print("=" * 60)
    print("🎨 BLIP3o Multi-Image Editing Inference")
    print("=" * 60)
    print(f"📂 Model: {args.model_path}")
    print(f"🖼️  Input: {len(input_images) if input_images else 0} image(s)")
    if input_images:
        for i, img_path in enumerate(args.input_images):
            print(f"     Image {i+1}: {img_path}")
    print(f"📝 Instruction: {args.instruction}")
    print(f"💾 Output: {args.output or 'Auto-generated'}")
    print(f"🖥️  Device: {args.device}")
    print(f"🔧 TAR SigLIP Features: {'Enabled' if args.use_tar_siglip_features else 'Disabled'}")
    print(f"🔧 UND image VAE: {'Enabled' if args.use_und_image_vae else 'Disabled'}")
    print("=" * 60)
    
    try:
        # Initialize config and inference
        config = EditingConfig(
            model_path=args.model_path,
            device=args.device,
            use_tar_siglip_features=args.use_tar_siglip_features,
            use_und_image_vae=args.use_und_image_vae,
            use_und_image_vae_as_noise=args.use_und_image_vae_as_noise,
            only_use_und_image_vae_as_noise=args.only_use_und_image_vae_as_noise,
            config=args,
        )
        
        inference = ImageEditingInference(config)
        
        # Generate/edit image
        print("🎨 Generating image...")
        result_image = inference.generate_image(args.instruction, input_images, args.seed)
        
        if result_image is None:
            print("❌ Failed to generate image")
            return 1
        
        # Save result
        if args.output:
            output_path = args.output
        else:
            # Create output filename
            safe_instruction = "".join(c for c in args.instruction if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_instruction = safe_instruction.replace(' ', '_')[:30]
            if input_images:
                output_path = f"edited_{len(input_images)}images_{safe_instruction}.png"
            else:
                output_path = f"generated_{safe_instruction}.png"
        
        result_image.save(output_path)
        print(f"✅ Image saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())