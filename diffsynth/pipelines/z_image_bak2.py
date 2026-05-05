import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Union, List, Optional, Tuple

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoTokenizer
from ..models.z_image_text_encoder import ZImageTextEncoder
from ..models.z_image_dit import ZImageDiT
from ..models.flux_vae import FluxVAEEncoder, FluxVAEDecoder


class ZImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("Z-Image")
        self.text_encoder: ZImageTextEncoder = None
        self.dit: ZImageDiT = None
        self.vae_encoder: FluxVAEEncoder = None
        self.vae_decoder: FluxVAEDecoder = None
        self.tokenizer: AutoTokenizer = None
        self.in_iteration_models = ("dit",)
        self.units = [
            ZImageUnit_ShapeChecker(),
            ZImageUnit_PromptEmbedder(),
            ZImageUnit_NoiseInitializer(),
            ZImageUnit_InputImageEmbedder(),
        ]
        self.model_fn = model_fn_z_image
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = ZImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("z_image_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        # Image
        input_image: Image.Image = None,
        condition_image: Image.Image = None,   # <<< NEW
        denoising_strength: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 8,
        # Progress bar
        progress_bar_cmd=tqdm,

        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image,
            "condition_image": condition_image,   # <<< NEW
            "denoising_strength": denoising_strength,
            "height": height,
            "width": width,
            "seed": seed,
            "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": use_gradient_checkpointing_offload,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class ZImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: ZImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class ZImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_embeds",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(
        self,
        pipe,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        embeddings_list = []

        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

        return embeddings_list

    def process(self, pipe: ZImagePipeline, prompt):
        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds = self.encode_prompt(pipe, prompt, pipe.device)
        return {"prompt_embeds": prompt_embeds}


class ZImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: ZImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}


class ZImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "condition_image", "noise"),
            output_params=("latents", "condition_latents", "input_latents"),
            onload_model_names=("vae_encoder",)
        )

    def process(self, pipe: ZImagePipeline, input_image, condition_image, noise):
        # 正确加载 encoder
        pipe.load_models_to_device(["vae_encoder"])

        # condition 可以没有
        condition_latents = None
        if condition_image is not None:
            condition_image = pipe.preprocess_image(condition_image)
            condition_latents = pipe.vae_encoder(condition_image)

        # input_image 允许 None（纯噪声起步）
        if input_image is None:
            return {
                "latents": noise,
                "input_latents": None,
                "condition_latents": condition_latents,
            }

        # 有 input_image 才做 encode
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae_encoder(image)

        # 训练：直接从 noise 走（你的原逻辑）
        if pipe.scheduler.training:
            return {
                "latents": noise,
                "input_latents": input_latents,
                "condition_latents": condition_latents,
            }

        # 推理：从 input_latents 加噪作为初值（img2img）
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {
            "latents": latents,
            "input_latents": input_latents,
            "condition_latents": condition_latents,
        }



def model_fn_z_image(
    dit: ZImageDiT,
    latents=None,              # (B, C, H, W)
    condition_latents=None,    # (B, C, H, W)
    timestep=None,             # (B,) or scalar
    prompt_embeds=None,        # List[Tensor] length B (通常)
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    B, C, H, W = latents.shape

    # 让 ZImageDiT 的 bsz = B；每个 sample 的 F=1
    x = [latents[i].unsqueeze(1) for i in range(B)]  # each: (C, 1, H, W)

    cond_x = None
    if condition_latents is not None:
        cond_x = [condition_latents[i].unsqueeze(1) for i in range(B)]  # each: (C, 1, H, W)

    timestep = (1000 - timestep) / 1000

    # 建议用关键字传参，避免 forward 参数顺序未来改动导致错位
    out_list = dit(
        x,
        timestep,
        prompt_embeds,
        condition_x=cond_x,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )[0]  # List[(C, 1, H, W)]

    # 只取 F=1 的那一帧，stack 回 (B,C,H,W)
    out = torch.stack([o[:, 0, :, :] for o in out_list], dim=0)

    out = -out
    return out

