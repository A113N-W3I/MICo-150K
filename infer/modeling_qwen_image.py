# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Independently adapted and refactored by the MICo Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor, PretrainedConfig
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import QwenImageLoraLoaderMixin
from diffusers.models import AutoencoderKLQwenImage, QwenImageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


class _MICoPipelineUtils:
    """
    Internal utility class grouping static computational methods for the diffusion pipeline.
    Maintains clean namespace and modularizes mathematical operations.
    """

    @staticmethod
    def compute_temporal_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        """Calculates the sequence shift parameter based on sequence length ratios."""
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        return image_seq_len * slope + intercept

    @staticmethod
    def fetch_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """Resolves custom timesteps or sigmas from the given scheduler instance."""
        if timesteps is not None and sigmas is not None:
            raise ValueError("Configuration Error: Define either `timesteps` or `sigmas`, but not both.")

        if timesteps is not None:
            if "timesteps" not in inspect.signature(scheduler.set_timesteps).parameters:
                raise ValueError(f"Scheduler {scheduler.__class__.__name__} rejects custom timesteps.")
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        elif sigmas is not None:
            if "sigmas" not in inspect.signature(scheduler.set_timesteps).parameters:
                raise ValueError(f"Scheduler {scheduler.__class__.__name__} rejects custom sigmas.")
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)

        return scheduler.timesteps, len(scheduler.timesteps)

    @staticmethod
    def extract_latent_distribution(
        encoder_output: Any, generator: Optional[torch.Generator] = None, mode: str = "sample"
    ) -> torch.Tensor:
        """Safely extracts tensors from VAE distributions."""
        if hasattr(encoder_output, "latent_dist"):
            return encoder_output.latent_dist.sample(generator) if mode == "sample" else encoder_output.latent_dist.mode()
        elif hasattr(encoder_output, "latents"):
            return encoder_output.latents
        raise AttributeError("Failed to extract latents: Unsupported encoder output format.")

    @staticmethod
    def pad_image_dimensions(image: Image.Image, multiple_of: int = 32) -> Image.Image:
        """Resizes a PIL image ensuring its dimensions are strict multiples of the given integer."""
        w, h = image.size
        aligned_w = round(w / multiple_of) * multiple_of
        aligned_h = round(h / multiple_of) * multiple_of
        return image.resize((aligned_w, aligned_h))


class QwenImageConfig(PretrainedConfig):
    """Configuration class for parameterized model tracking."""
    model_type = "qwen_image_transformer"

    def __init__(
        self,
        attention_head_dim=128,
        num_attention_heads=24,
        num_layers=60,
        in_channels=64,
        out_channels=16,
        patch_size=2,
        joint_attention_dim=3584,
        axes_dims_rope=[16, 56, 56],
        guidance_embeds=False,
        **kwargs,
    ):
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.joint_attention_dim = joint_attention_dim
        self.axes_dims_rope = axes_dims_rope
        self.guidance_embeds = guidance_embeds
        super().__init__(**kwargs)


MICO_EXAMPLE_DOCS = """
    Examples:
        ```py
        >>> import torch
        >>> from PIL import Image
        >>> from diffusers import QwenImageEditPipeline
        >>> from diffusers.utils import load_image

        >>> # Initialize the MICo-adapted Multi-Modal Pipeline
        >>> pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
        >>> pipeline.to("cuda")
        
        >>> # Fetch source vision data
        >>> source_vision = load_image("https://.../sample_workspace.png").convert("RGB")
        
        >>> # Define agent-based instruction for visual modification
        >>> task_instruction = (
        ...     "Modify the scene into a futuristic computer vision lab, "
        ...     "featuring an autonomous agent analyzing a complex LaTeX manuscript, highly detailed."
        ... )
        
        >>> generated_output = pipeline(source_vision, task_instruction, num_inference_steps=50).images[0]
        >>> generated_output.save("mico_agent_output.png")
        ```
"""


class QwenImageEditPipeline(DiffusionPipeline, QwenImageLoraLoaderMixin):
    r"""
    Advanced Pipeline Architecture adapted for dynamic image editing via multi-modal instructions.
    
    Architecture Components:
        transformer: The core MMDiT architecture handling conditional denoising.
        scheduler: Flow-matching scheduler controlling the noise traversal sequence.
        vae: Handles transition between pixel space and latent dimensional space.
        text_encoder: Qwen2.5-VL instruction encoder framework.
        tokenizer: Linguistic boundary mapping utility.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLQwenImage,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tokenizer: Qwen2Tokenizer,
        processor: Qwen2VLProcessor,
        transformer: QwenImageTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.vl_processor = processor
        self.tokenizer_max_length = 1024

        self.system_message = (
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. Generate a new "
            "image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        )
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Isolates valid computational vectors from padded hidden states."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected_states = hidden_states[bool_mask]
        return torch.split(selected_states, valid_lengths.tolist(), dim=0)

    def _get_qwen_prompt_embeds(
        self,
        prompts: Union[str, List[str]] = None,
        images: List[List[Image.Image]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Processes and extracts embeddings via the visual-language processor framework."""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Standardize nested image lists
        if isinstance(images, Image.Image):
            images = [[images]]
        elif isinstance(images[0], Image.Image):
            images = [images]
            
        assert len(prompts) == len(images), "Data integrity error: Prompt and vision input batch sizes mismatch."

        compiled_texts = []
        for prompt, image_list in zip(prompts, images):
            chat_payload = [
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img} for img in image_list] + [{"type": "text", "text": prompt}],
                },
            ]
            compiled_texts.append(self.processor.apply_chat_template(chat_payload, tokenize=False, add_generation_prompt=True))

        model_inputs = self.processor(
            text=compiled_texts,
            images=images,
            do_resize=False,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        drop_idx = self.prompt_template_encode_start_idx
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_states = [state[drop_idx:] for state in split_states]
        
        attn_mask_list = [torch.ones(state.size(0), dtype=torch.long, device=state.device) for state in split_states]
        max_seq_len = max([state.size(0) for state in split_states])
        
        prompt_embeds = torch.stack(
            [torch.cat([state, state.new_zeros(max_seq_len - state.size(0), state.size(1))]) for state in split_states]
        ).to(dtype=dtype, device=device)
        
        encoder_attention_mask = torch.stack(
            [torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) for mask in attn_mask_list]
        )

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        images: List[Image.Image] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        """Constructs and duplicates conditional multi-modal latents for execution batches."""
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, images, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1).view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        """Performs pre-flight integrity checks on all pipeline parameters."""
        spatial_constraint = self.vae_scale_factor * 2
        if height % spatial_constraint != 0 or width % spatial_constraint != 0:
            logger.warning(f"Resolution ({width}x{height}) is not divisible by {spatial_constraint}. Auto-resizing will trigger.")

        if callback_on_step_end_tensor_inputs:
            invalid_cbs = [k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]
            if invalid_cbs:
                raise ValueError(f"Invalid callback tensors: {invalid_cbs}. Permitted: {self._callback_tensor_inputs}")

        if bool(prompt is not None) == bool(prompt_embeds is not None):
            raise ValueError("Configuration logic trap: Define strictly one of `prompt` OR `prompt_embeds`.")
            
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` validation failed. Expected str/list, got {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Conflict: Both `negative_prompt` and `negative_prompt_embeds` were supplied.")

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("Masking integrity error: `prompt_embeds_mask` is mandatory when injecting `prompt_embeds`.")
            
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError("Masking integrity error: `negative_prompt_embeds_mask` is mandatory alongside `negative_prompt_embeds`.")

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"Sequence cap exceeded: {max_sequence_length} > 1024 limit.")

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """Transforms standard latent layouts into packed multi-patch configurations."""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """Reverses multi-patch configurations back to classical latent layouts."""
        batch_size, _, channels = latents.shape
        h_adjusted = 2 * (int(height) // (vae_scale_factor * 2))
        w_adjusted = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, h_adjusted // 2, w_adjusted // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, 1, h_adjusted, w_adjusted)

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        """Converts raw pixel space into initial VAE distributions."""
        if isinstance(generator, list):
            latents_cache = [
                _MICoPipelineUtils.extract_latent_distribution(self.vae.encode(image[i : i + 1]), generator=generator[i], mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(latents_cache, dim=0)
        else:
            image_latents = _MICoPipelineUtils.extract_latent_distribution(self.vae.encode(image), generator=generator, mode="argmax")
            
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        
        return (image_latents - latents_mean) / latents_std

    def enable_vae_slicing(self): self.vae.enable_slicing()
    def disable_vae_slicing(self): self.vae.disable_slicing()
    def enable_vae_tiling(self): self.vae.enable_tiling()
    def disable_vae_tiling(self): self.vae.disable_tiling()

    def prepare_latents(
        self, images, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None
    ):
        """Allocates and standardizes initial noise vectors and vision encodings for the step loop."""
        h_adjusted = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_adjusted = 2 * (int(width) // (self.vae_scale_factor * 2))
        noise_shape = (batch_size, 1, num_channels_latents, h_adjusted, w_adjusted)

        processed_img_latents = []
        for img_tensor in images:
            img_tensor = img_tensor.to(device=device, dtype=dtype)
            current_latents = self._encode_vae_image(img_tensor, generator) if img_tensor.shape[1] != self.latent_channels else img_tensor
            
            if batch_size > current_latents.shape[0] and batch_size % current_latents.shape[0] == 0:
                duplication_factor = batch_size // current_latents.shape[0]
                current_latents = torch.cat([current_latents] * duplication_factor, dim=0)
            elif batch_size > current_latents.shape[0]:
                raise ValueError(f"Batch misalignment: Cannot broadcast {current_latents.shape[0]} inputs to size {batch_size}.")

            img_h, img_w = current_latents.shape[3:]
            packed_img = self._pack_latents(current_latents, batch_size, num_channels_latents, img_h, img_w)
            processed_img_latents.append(packed_img)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator constraint failed. Passed {len(generator)} generators for batch size {batch_size}.")
            
        if latents is None:
            latents = randn_tensor(noise_shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, h_adjusted, w_adjusted)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, processed_img_latents

    @property
    def guidance_scale(self): return self._guidance_scale

    @property
    def attention_kwargs(self): return self._attention_kwargs

    @property
    def num_timesteps(self): return self._num_timesteps

    @property
    def current_timestep(self): return self._current_timestep

    @property
    def interrupt(self): return self._interrupt

    def _prepare_vision_inputs(self, images: List[PipelineImageInput]) -> Tuple[List[Image.Image], int, int]:
        """Calculates area-constrained ratios and standardizes spatial constraints."""
        total_pixels = sum(math.prod(img.size) for img in images)
        ratio = 1024 / (total_pixels ** 0.5)
        
        resized_imgs = [img.resize(size=(round(img.width * ratio), round(img.height * ratio))) for img in images]
        aligned_imgs = [_MICoPipelineUtils.pad_image_dimensions(img, multiple_of=32) for img in resized_imgs]
        
        return aligned_imgs, aligned_imgs[0].width, aligned_imgs[0].height

    def _decode_latents_to_image(self, latents, height, width, output_type):
        """Transforms structural noise output back to human-interpretable pixels."""
        if output_type == "latent":
            return latents

        unpacked_latents = self._unpack_latents(latents, height, width, self.vae_scale_factor).to(self.vae.dtype)
        
        l_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(unpacked_latents.device, unpacked_latents.dtype)
        l_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(unpacked_latents.device, unpacked_latents.dtype)
        
        normalized_latents = unpacked_latents / l_std + l_mean
        decoded_pixels = self.vae.decode(normalized_latents, return_dict=False)[0][:, :, 0]
        
        return self.image_processor.postprocess(decoded_pixels, output_type=output_type)

    @torch.no_grad()
    @replace_example_docstring(MICO_EXAMPLE_DOCS)
    def __call__(
        self,
        images: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Executes the multi-modal editing sequence based on the provided vision inputs and textual instructions.

        Args:
            images: Source vision data to be modified.
            prompt: Textual instructions dictating the structural or stylistic changes.
            negative_prompt: Elements or styles to actively suppress during generation.
            num_inference_steps: Number of denoising iterations (default: 50).
            guidance_scale: Modifier for classifier-free guidance adherence.
            output_type: Format of the returned vision data ('pil' or 'latent').
            return_dict: Whether to wrap outputs in a pipeline standard dictionary.

        Examples:

        Returns:
            `QwenImagePipelineOutput` or `tuple`: Contains the generated vision data.
        """
        # 1. Image preparation & Alignment
        images = [images] if not isinstance(images, (list, tuple)) else images
        images, derived_w, derived_h = self._prepare_vision_inputs(images)
        
        width, height = width or derived_w, height or derived_h
        padding_base = self.vae_scale_factor * 2
        width = (width // padding_base) * padding_base
        height = (height // padding_base) * padding_base

        # 2. Input Sanity Checks
        self.check_inputs(
            prompt, height, width, negative_prompt, prompt_embeds,
            negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs, max_sequence_length
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        batch_size = len(prompt) if isinstance(prompt, list) else (prompt_embeds.shape[0] if prompt_embeds is not None else 1)
        device = self._execution_device

        # 3. Instruction Encoding
        scaled_vision = [img.resize((round(img.width * 28 / 32), round(img.height * 28 / 32))) for img in images]
        processed_images = [self.image_processor.preprocess(img, img.height, img.width).unsqueeze(2) for img in images]

        cfg_eligible = true_cfg_scale > 1.0 and (negative_prompt is not None or negative_prompt_embeds is not None)
        
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            images=scaled_vision, prompt=prompt, prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask, device=device,
            num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
        )

        if cfg_eligible:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                images=scaled_vision, prompt=negative_prompt, prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask, device=device,
                num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
            )

        # 4. Latent Space Injection
        v_channels = self.transformer.config.in_channels // 4
        latents, img_latents = self.prepare_latents(
            processed_images, batch_size * num_images_per_prompt, v_channels,
            height, width, prompt_embeds.dtype, device, generator, latents
        )

        img_shapes = [[(1, height // padding_base, width // padding_base)] + [(1, img.shape[-2] // padding_base, img.shape[-1] // padding_base) for img in processed_images]] * batch_size

        # 5. Scheduling Alignment
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        shift_mu = _MICoPipelineUtils.compute_temporal_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _MICoPipelineUtils.fetch_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=shift_mu)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        guidance_vec = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if self.transformer.config.guidance_embeds else None

        txt_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        neg_txt_lens = negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None

        # 6. Target Denoising Loop execution
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue
                self._current_timestep = t
                t_expanded = t.expand(latents.shape[0]).to(latents.dtype)
                combined_states = torch.cat([latents] + img_latents, dim=1)

                with self.transformer.cache_context("cond"):
                    pred_noise = self.transformer(
                        hidden_states=combined_states, timestep=t_expanded / 1000, guidance=guidance_vec,
                        encoder_hidden_states_mask=prompt_embeds_mask, encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes, txt_seq_lens=txt_lens, attention_kwargs=self.attention_kwargs, return_dict=False
                    )[0][:, : latents.size(1)]

                if cfg_eligible:
                    with self.transformer.cache_context("uncond"):
                        neg_pred_noise = self.transformer(
                            hidden_states=combined_states, timestep=t_expanded / 1000, guidance=guidance_vec,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask, encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes, txt_seq_lens=neg_txt_lens, attention_kwargs=self.attention_kwargs, return_dict=False
                        )[0][:, : latents.size(1)]
                        
                    blended_pred = neg_pred_noise + true_cfg_scale * (pred_noise - neg_pred_noise)
                    cond_norm, blend_norm = torch.norm(pred_noise, dim=-1, keepdim=True), torch.norm(blended_pred, dim=-1, keepdim=True)
                    pred_noise = blended_pred * (cond_norm / blend_norm)

                base_dtype = latents.dtype
                latents = self.scheduler.step(pred_noise, t, latents, return_dict=False)[0]

                if latents.dtype != base_dtype:
                    latents = latents.to(base_dtype) if torch.backends.mps.is_available() else latents

                if callback_on_step_end:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    cb_outputs = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_outputs.pop("latents", latents)
                    prompt_embeds = cb_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE: xm.mark_step()

        self._current_timestep = None
        
        # 7. Final Output Resolution
        final_image = self._decode_latents_to_image(latents, height, width, output_type)
        self.maybe_free_model_hooks()

        return QwenImagePipelineOutput(images=final_image) if return_dict else (final_image,)