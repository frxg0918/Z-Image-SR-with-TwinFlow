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

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, condition_timestep_zero=False):
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
        self.condition_timestep_zero = condition_timestep_zero
        self.load_models_to_device(['vae_decoder'])
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
        condition_timestep_zero: bool = False,
        enable_2_temb: bool = False,
    ):
        # Initialize pipeline
        # print("condition_timestep_zero: ", condition_timestep_zero)

        # 傻逼玩意儿，没有用
        # import copy
        # updated_model_configs = []
        
        # for model_config in model_configs:
        #     new_config = copy.copy(model_config)
        #     if not hasattr(new_config, 'extra_kwargs') or new_config.extra_kwargs is None:
        #         new_config.extra_kwargs = {}
        #     if hasattr(model_config, 'model_name') and model_config.model_name == "z_image_dit":
        #         new_config.extra_kwargs['enable_2_temb'] = enable_2_temb
        #     updated_model_configs.append(new_config)

        pipe = ZImagePipeline(device=device, torch_dtype=torch_dtype, condition_timestep_zero=condition_timestep_zero)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("z_image_dit")
        pipe.vae_encoder = model_pool.fetch_model("flux_vae_encoder")
        pipe.vae_decoder = model_pool.fetch_model("flux_vae_decoder")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        pipe.dit.condition_timestep_zero = condition_timestep_zero
        # pipe.dit.enable_2_temb = enable_2_temb
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
        progress_bar_cmd = tqdm,

        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
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

        # Denoise # 对应于模型的两个时间步输入
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        # print("self.scheduler.timesteps", self.scheduler.timesteps)
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # print("before transformation timestep:", timestep) # 
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device) # 这里的数据类型转换导致数据产生误差, 尝试先不量化
            # timestep = timestep.unsqueeze(0).to(device=self.device) # 这里的数据类型转换导致数据产生误差, 尝试先不量化
            
            # 计算目标时间步（下一个时间步，如果是最后一步则为0）
            if progress_id + 1 < len(self.scheduler.timesteps):
                target_timestep = self.scheduler.timesteps[progress_id + 1].unsqueeze(0).to(dtype=timestep.dtype, device=self.device) # 这里的数据类型转换导致数据产生误差, 尝试先不量化
                # 这是twinflow对于多步infer的写法
                target_timestep = torch.zeros_like(timestep, dtype=timestep.dtype, device=self.device)
            else:
                target_timestep = torch.zeros_like(timestep, dtype=timestep.dtype, device=self.device)
            # print("timestep, target_timestep: ", timestep, target_timestep) # 这边和上面print scheduler.timesteps对应不上，说明scheduler.timesteps有问题
            # 如果要多步正常infer，不带target timestep，就设置为target_timestep=None
            # target_timestep = None
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id, target_timestep=target_timestep, condition_timestep_zero=self.condition_timestep_zero,
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
            # print("inputs_shared['latents'].dtype", inputs_shared["latents"].dtype)
            # inputs_shared["latents"] = inputs_shared["latents"].to(dtype=self.torch_dtype)
            # print(f"timestep {timestep} dtype:", timestep.dtype)
        
        # Decode
        # self.load_models_to_device(['vae_decoder'])
        image = self.vae_decoder(inputs_shared["latents"])
        image = self.vae_output_to_image(image)
        # self.load_models_to_device([])

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
            # 增加 input_image, condition_image, prompt 作为输入，用来判断 Batch Size
            input_params=("height", "width", "seed", "rand_device", "input_image", "condition_image", "prompt"),
            output_params=("noise",),
        )

    def process(self, pipe: ZImagePipeline, height, width, seed, rand_device, input_image, condition_image, prompt):
        # 1. 自动推断 Batch Size
        batch_size = 1
        
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                batch_size = input_image.shape[0]
            elif isinstance(input_image, list):
                batch_size = len(input_image)
        elif condition_image is not None:
             if isinstance(condition_image, torch.Tensor):
                batch_size = condition_image.shape[0]
             elif isinstance(condition_image, list):
                batch_size = len(condition_image)
        elif isinstance(prompt, list):
            batch_size = len(prompt)

        # 2. 生成对应 Batch Size 的噪声
        # 注意将第一个维度从 1 改为 batch_size
        noise = pipe.generate_noise(
            (batch_size, 16, height//8, width//8), # 16 是 VAE 的通道数
            seed=seed, 
            rand_device=rand_device, 
            rand_torch_dtype=pipe.torch_dtype
        )
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

        # --- 辅助函数：统一处理 Tensor/PIL/List[PIL] ---
        def prepare_image_tensor(img_input):
            if img_input is None:
                return None
            
            # [情况 A]: 训练模式 - 输入已经是 Batch Tensor (B, 3, H, W)
            # 来自 DataLoader，已经在 Dataset 里 Resize 和 Normalize 过了
            if isinstance(img_input, torch.Tensor):
                # 如果是单张 Tensor (3, H, W)，增加 Batch 维度
                if img_input.ndim == 3:
                    img_input = img_input.unsqueeze(0)
                # 移动到正确的设备和精度
                return img_input.to(device=pipe.device, dtype=pipe.torch_dtype)
            
            # [情况 B]: 推理模式 - 输入是 PIL Image List (Batch Inference)
            if isinstance(img_input, list):
                # pipe.preprocess_image 负责 resize/norm, 返回 List[Tensor] 或 Tensor
                tensors = [pipe.preprocess_image(img) for img in img_input]
                # preprocess_image 通常返回 (1, 3, H, W)，cat 起来变成 (B, 3, H, W)
                return torch.cat(tensors, dim=0).to(device=pipe.device, dtype=pipe.torch_dtype)

            # [情况 C]: 推理模式 - 输入是单张 PIL Image
            # pipe.preprocess_image 返回 (1, 3, H, W)
            return pipe.preprocess_image(img_input).to(device=pipe.device, dtype=pipe.torch_dtype)
        # ------------------------------------------------

        # 1. 准备像素数据 (自动分流 PIL 或 Tensor)
        pixel_values = prepare_image_tensor(input_image)          # GT / HQ
        condition_pixel_values = prepare_image_tensor(condition_image) # Condition / LQ

        # 2. VAE Encode (Condition)
        condition_latents = None
        if condition_pixel_values is not None:
            # 此时 condition_pixel_values 已经是正确的 Tensor (B, 3, H, W)
            condition_latents = pipe.vae_encoder(condition_pixel_values)
            # 注意: Flux VAE 通常不需要 scaling_factor，但如果是 SDXL VAE 则需要。
            # 这里假设是 Flux VAE，不做 scaling。如果需要，请解开下行注释:
            # condition_latents = condition_latents * pipe.vae_scaling_factor

        # 3. 处理 Latents (Training vs Inference)
        
        # [分支 1]: 纯噪声启动 (无 input_image)
        if pixel_values is None:
            return {
                "latents": noise, # 注意：此时 noise 的 batch size 需要在 NoiseInitializer 中正确设置
                "input_latents": None,
                "condition_latents": condition_latents,
                "condition_rgb": condition_pixel_values, # 返回 Tensor 用于可视化
                "gt_rgb": pixel_values,
            }
        
        # VAE Encode (Input/GT)
        input_latents = pipe.vae_encoder(pixel_values)
        # if scaling_needed: input_latents = input_latents * pipe.vae_scaling_factor

        # [分支 2]: 训练模式
        if pipe.scheduler.training:
            # 训练时，直接返回 GT latents 作为 input_latents
            # 外部 Loss 函数会负责加噪，这里的 noise 变量主要作为 shape 参考
            return {
                "latents": noise, # 这里的 noise 在 loss 计算前会被重置或作为初始噪声
                "input_latents": input_latents,
                "condition_latents": condition_latents,
                "condition_rgb": condition_pixel_values,
                "gt_rgb": pixel_values,
            }

        # [分支 3]: 推理模式 (img2img)
        # 确保 noise 的 batch size 和 input_latents 一致
        if noise.shape[0] != input_latents.shape[0]:
            # 如果不一致，通常意味着 NoiseInitializer 没能正确推断 batch size
            # 这里可以做一个 fallback，但这应该在 NoiseInitializer 中解决
            pass

        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        
        return {
            "latents": latents,
            "input_latents": input_latents,
            "condition_latents": condition_latents,
            "condition_rgb": condition_pixel_values, # 返回 Tensor
            "gt_rgb": pixel_values,
        }



def model_fn_z_image(
    dit: ZImageDiT,
    latents=None,
    condition_latents=None,
    timestep=None,
    target_timestep=None,
    prompt_embeds=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    condition_timestep_zero=False,
    **kwargs,
):
    if dit.condition_timestep_zero is not None:
        condition_timestep_zero = dit.condition_timestep_zero
    B, C, H, W = latents.shape

    DEBUG = False
    # print("condition_timestep_zero: ", condition_timestep_zero)

    # print("B, C, H, W: ", B, C, H, W) # 还没patch化
    # raise Exception("Not implemented")

    # 让 ZImageDiT 的 bsz = B；每个 sample 的 F=1
    x = [latents[i].unsqueeze(1) for i in range(B)]  # each: (C, 1, H, W)

    cond_x = None
    if condition_latents is not None:
        cond_x = [condition_latents[i].unsqueeze(1) for i in range(B)]  # each: (C, 1, H, W)
    # latents = [rearrange(latents, "B C H W -> C B H W")]
    # timestep = (1000 - timestep) / 1000

    # if target_timestep is None:
    #     target_timestep = timestep
    # else:
    #     target_timestep = (1000 - target_timestep) / 1000

    # for 2 t_embedders -------------------------------------------------------------------------------------------
    # 更优雅地处理时间步归一化，支持负数和batch形式
    def normalize_timestep(t):
        # 首先将-1000~1000范围映射到-1~1范围
        normalized = t / 1000.0
        
        # 然后根据正负数应用不同的变换
        # 正数: 0~1 -> 1~0 (1 - normalized)
        # 负数: -1~0 -> -1~0 (-1 - normalized)
        # normalized_sign = (normalized >= 0).to(normalized.dtype) * 2 - 1  # True->1, False->-1
        normalized_sign = torch.where(normalized >= 0, 1, -1)
        normalized_abs = normalized.abs()
        return normalized_sign * (1.0 - normalized_abs)
    
    # timestep = timestep # only use timestep for multi-step duallora training
    t_input = normalize_timestep(timestep)
    if target_timestep is None:
        tt_input = t_input
    else:
        tt_input = normalize_timestep(target_timestep)
    # for 2 t_embedders -------------------------------------------------------------------------------------------

    # def normalize_timestep(t):
    #     # 首先将-1000~1000范围映射到-1~1范围
    #     normalized = t / 1000.0
        
    #     # 然后根据正负数应用不同的变换
    #     # 正数: 0~1 -> 1~0 (1 - normalized)
    #     # 负数: -1~0 -> -1~0 (-1 - normalized)
    #     result = torch.where(normalized >= 0, 1 - normalized, -1 - normalized)
    #     return result
    
    # if target_timestep is None:
    #     target_timestep = timestep
    
    # timestep = timestep - target_timestep / 2
    # timestep = normalize_timestep(timestep)


    if DEBUG:
        print("timestep, target_timestep", t_input, tt_input)
    # set timestep, target_timestep for condition_x, make it always positive
    # TODO 目前是对噪声分支的t作绝对值，对条件分支的tt保留负号指示真流假流
    t_input = t_input.abs()
    tt_input = tt_input

    if condition_timestep_zero:
        print("条件分支的时间步设置为0")
        cond_t = normalize_timestep(torch.zeros_like(t_input))
        # cond_tt = target_timestep.abs()
        cond_tt = normalize_timestep(torch.zeros_like(t_input))
    else:
        cond_t = t_input
        cond_tt = tt_input

    if DEBUG:
        print("latents dtype:", latents[0].dtype)
        print("timestep: ", timestep, "timestep dtype:", timestep.dtype)
    
    # 在进模型前再量化，避免精度损失
    # t_input = t_input.to(latents.dtype)
    # tt_input = tt_input.to(latents.dtype)
    # cond_t = cond_t.to(latents.dtype)
    # cond_tt = cond_tt.to(latents.dtype)

    # 建议用关键字传参，避免 forward 参数顺序未来改动导致错位
    # 条件分支的双时间步都用 0
    out_list = dit(
        x,
        [t_input, cond_t],
        prompt_embeds,
        condition_x=cond_x,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        target_timestep=[tt_input, cond_tt],  # NEW: Pass target_timestep to the model
    )[0]  # List[(C, 1, H, W)]

    if DEBUG:
        print("dit forward done!")

    # 只取 F=1 的那一帧，stack 回 (B,C,H,W)
    out = torch.stack([o[:, 0, :, :] for o in out_list], dim=0)

    out = -out
    return out
    
    
    
    # model_output = dit(
    #     latents,
    #     timestep,
    #     prompt_embeds,
    #     use_gradient_checkpointing=use_gradient_checkpointing,
    #     use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    # )[0][0]
    # if DEBUG:
    #     print("dit forward done!")
    # model_output = -model_output
    # model_output = rearrange(model_output, "C B H W -> B C H W")
    # return model_output
