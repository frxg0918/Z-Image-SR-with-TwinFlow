import torch, os, argparse, accelerate, copy
from datasets_utils.dataset import PairedSROnlineTxtDataset
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core import load_state_dict
import lpips
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def replace_linear_with_duallora(model, patterns, rank, alpha1, alpha2, use_fp8=False):
    """
    先冻结model所有参数，
    将匹配patterns的nn.Linear替换为DualLoRALinear
    不匹配的替换为 LinearFP8Wrapper
    """
    def to_fp8(tensor):
        # 仅在 use_fp8 时转换
        if use_fp8:
            return tensor.to(dtype=torch.float8_e4m3fn)
        return tensor

    # 冻结全部参数
    for p in model.parameters():
        p.requires_grad = False

    use_normal_lora_for_non_matched = True
    # [新增] 2. 显式解冻 time_fusion 和 t_embedder_2 (如果有的话)
    # 因为它们是新层，应该全量训练
    # for name, module in model.named_modules():
    #     if "time_fusion" in name or "t_embedder_2" in name:
    #         for p in module.parameters():
    #             p.requires_grad = True
    #         print(f"[Unlocked] {name} for full finetuning")

    def _replace_module(parent, name_prefix=""):
        for name, module in list(parent.named_children()):
            full_name = f"{name_prefix}{name}"
            # [关键修改] 如果遇到 time_fusion 或 t_embedder_2，直接跳过，不进行 LoRA 替换
            # if "time_fusion" in full_name or "t_embedder_2" in full_name:
            #     continue
            if isinstance(module, torch.nn.Linear):
                module.weight.data = to_fp8(module.weight.data)
                if module.bias is not None:
                    module.bias.data = to_fp8(module.bias.data)

                if patterns == ["all-linear"]:
                    # 替换为 DualLoRALinear
                    new_module = DualLoRALinear(module, rank, alpha1, alpha2)
                    setattr(parent, name, new_module)
                    print(f"[lora] {full_name} -> DualLoRALinear")
                else:
                    if any(p in full_name for p in patterns):
                        # 替换为 DualLoRALinear
                        if alpha1 != 0:
                            new_module = DualLoRALinear(module, rank, alpha1, alpha2)
                            print(f"[lora] {full_name} -> DualLoRALinear")
                        else:
                            new_module = MaskedLoRALinear(module, rank, alpha2)
                            print(f"[lora] {full_name} -> MaskedLoRALinear")
                        setattr(parent, name, new_module)
                    else:
                        # 不匹配的根据参数决定替换为普通 LoraLinear 或 LinearFP8Wrapper
                        if use_normal_lora_for_non_matched and ('t_embedder' in full_name) or ('all_x_embedder' in full_name) or ('fusion' in full_name):
                            new_module = LoraLinear(module, rank, alpha1)
                            setattr(parent, name, new_module)
                            print(f"[lora] {full_name} -> LoraLinear (rank={rank}, alpha={alpha2})")
                        else:
                            # 替换为自动精度转换的 FP8LinearWrapper
                            new_module = LinearFP8Wrapper(module)
                            setattr(parent, name, new_module)
                            print(f"[cast] {full_name} -> NoLora LinearFP8Wrapper (fp8={use_fp8})")
            else:
                _replace_module(module, name_prefix=full_name + ".")
    
    _replace_module(model)

class LoraLinear(torch.nn.Module):
    """
    普通 LoRA 线性层包装器
    """
    def __init__(self, linear, rank, alpha):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.rank = rank
        self.alpha = alpha
        self.linear = linear

        dev = linear.weight.device
        dt = torch.bfloat16

        # 一套低秩矩阵
        self.lora_A = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)

        self.scaling = alpha / max(1, rank)

        torch.nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)

    def forward(self, x):
        y = self._base_linear(x)
        delta = self.lora_B(self.lora_A(x)) * self.scaling
        return y + delta
    
class LinearFP8Wrapper(torch.nn.Module):
    """普通 Linear forward 时自动 cast 到输入 dtype"""
    def __init__(self, linear):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.weight = linear.weight
        self.bias = linear.bias
        # 冻结参数
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)

class DualLoRALinear(torch.nn.Module):
    """
    对序列前一半/后一半分别使用不同 LoRA 的 Linear 包装器。
    原 Linear 权重原地共享，不额外复制。
    """
    def __init__(self, linear, rank, alpha1, alpha2):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.rank    = rank
        self.alpha1  = alpha1
        self.alpha2  = alpha2
        self.linear  = linear

        dev = linear.weight.device
        dt  = torch.bfloat16

        # 两套低秩矩阵，参数量 2 * (in * rank + rank * out)
        self.lora_A1 = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B1 = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)
        self.lora_A2 = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B2 = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)

        self.scaling1 = alpha1 /  max(1, rank)
        self.scaling2 = alpha2 /  max(1, rank)

        torch.nn.init.normal_(self.lora_A1.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B1.weight)
        torch.nn.init.normal_(self.lora_A2.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B2.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)

    def forward(self, x):
        if x.ndim == 2:
            print("[lora] DualLoRALinear: x.ndim == 2") # 检查是否有变量意外经过这个forward
            y = self._base_linear(x)  # (B, C_out)
            delta1 = self.lora_B1(self.lora_A1(x)) * self.scaling1
            delta2 = self.lora_B2(self.lora_A2(x)) * self.scaling2
            y1 = y + delta1
            y2 = y + delta2
            return torch.stack([y1, y2], dim=1)  # (B, 2, C_out)
        else:
            B, L2, _ = x.shape
            # print("Sequence lenth is ： ", L2)
            assert L2 % 2 == 0, "sequence length must be even"
            L = L2 // 2

            y = self._base_linear(x)                           # [B, 2L, C_out]
            x1 = x[:, :L, :]                             # [B, L, C_in]
            x2 = x[:, L:, :]                             # [B, L, C_in]

            delta1 = self.lora_B1(self.lora_A1(x1)) * self.scaling1   # [B, L, C_out]
            delta2 = self.lora_B2(self.lora_A2(x2)) * self.scaling2   # [B, L, C_out]

            y[:, :L] += delta1
            y[:, L:] += delta2
            return y

class MaskedLoRALinear(torch.nn.Module):
    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        assert isinstance(linear, torch.nn.Linear)
        self.rank       = rank
        self.alpha      = alpha
        self.linear     = linear

        dev = linear.weight.device
        dt  = torch.bfloat16

        # 两套低秩矩阵，参数量 2 * (in * rank + rank * out)
        self.lora_A1 = torch.nn.Linear(linear.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B1 = torch.nn.Linear(rank, linear.out_features, bias=False).to(device=dev, dtype=dt)

        self.scaling = alpha /  max(1, rank)

        torch.nn.init.normal_(self.lora_A1.weight, std=1.0 / rank)
        torch.nn.init.zeros_(self.lora_B1.weight)

    def _base_linear(self, x):
        # 仅在计算里把 float8 权重/偏置转成 x.dtype（bf16）
        w = self.linear.weight.detach().to(dtype=x.dtype)
        b = None
        if self.linear.bias is not None:
            b = self.linear.bias.detach().to(dtype=x.dtype)
        return torch.nn.functional.linear(x, w, b)
    def forward(self, x, slice_info=None):
        """
        x: [B, Seq_Len, Dim]
        slice_info: tuple (noise_len, cond_len, text_len)
        """
        # 1. 计算基础输出 (全序列)
        base_out = self._base_linear(x)

        # TODO 先这样保证adaln_modulation不报错，不然需要去掉adaLN的sequential结构，重写loadstatedict逻辑
        if x.shape[1] == 2:
            slice_info = (1,1,0)

        if slice_info is None:
            print("MaskedLoRALinear slice_info is None.")
            return base_out
        
        # print("MaskedLoRALinear slice_info: ", slice_info)
        # print("MaskedLoRALinear x: ", x.shape)
        assert sum(slice_info) == x.shape[1], "slice_info error! x.shape[1] should be equal to sum(slice_info)" 
            
        noise_len, cond_len, text_len = slice_info
        
        # 2. 仅对 Condition 部分计算 LoRA
        # 假设序列结构为 [Noise, Cond, Text]
        cond_start = noise_len
        cond_end = noise_len + cond_len
        
        # 提取 Condition Tokens
        x_cond = x[:, cond_start:cond_end, :]
        
        # 计算 LoRA 增量
        lora_delta = self.lora_B1(self.lora_A1(x_cond)) * self.scaling
        
        # 3. 将增量加回基础输出
        # 注意：这里涉及到原地修改，为了梯度安全，建议用 clone 或 tensor slice assignment
        # output = base_out.clone() # 费显存，但安全
        # output[:, cond_start:cond_end, :] += lora_delta
        
        # 更省显存的写法：拼接
        out_noise = base_out[:, :cond_start, :]
        out_cond  = base_out[:, cond_start:cond_end, :] + lora_delta
        out_text  = base_out[:, cond_end:, :]
        
        return torch.cat([out_noise, out_cond, out_text], dim=1)
            
class ZImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_alpha=2.0, lora_checkpoint=None, enable_alpha1=True, condition_timestep_zero=False,
        enable_2_temb=False,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        # print("Zimagepipeline loading... condition_timestep_zero:", condition_timestep_zero, "task:", task)
        self.pipe = ZImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, condition_timestep_zero=condition_timestep_zero, enable_2_temb=enable_2_temb)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # [关键修改]: 强制将 VAE Decoder 移动到 GPU 并常驻
        # 这一步是为了防止 Loss 计算时的 Tensor 设备不匹配
        if hasattr(self.pipe, "vae_decoder") and self.pipe.vae_decoder is not None:
            self.pipe.vae_decoder.to(device)
            self.pipe.vae_decoder.requires_grad_(False) # 确保冻结
            print(f"VAE Decoder moved to {device} and locked.")

        # =========================================================
        # [最佳实践] 在这里加载 LPIPS，并永久挂载到 pipe 上
        # =========================================================
        print(f"Loading LPIPS model to {device}...")
        # 1. 初始化 (使用 alex 通常比 vgg 快且显存占用小，适合训练)
        lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # 2. 冻结参数 (绝对不要把 LPIPS 的参数加入到 optimizer 中)
        lpips_model.requires_grad_(False)
        lpips_model.eval()
        
        # 3. 挂载到 pipe 对象上，作为一个自定义属性
        # 这样在 TwinFlowLoss(pipe, ...) 里就能直接用 pipe.loss_fn_lpips 访问了
        self.pipe.loss_fn_lpips = lpips_model

        # Training mode
        self.pipe.scheduler.set_timesteps(1000, training=True)
        
        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Preset LoRA
        if preset_lora_path is not None:
            self.pipe.load_lora(getattr(self.pipe, preset_lora_model), preset_lora_path)
        
        if lora_base_model is not None and not task.endswith(":data_process"):
            if (not hasattr(self.pipe, lora_base_model)) or getattr(self.pipe, lora_base_model) is None:
                print(f"No {lora_base_model} models in the pipeline. We cannot patch LoRA on the model. If this occurs during the data processing stage, it is normal.")
                return
            model = self.add_custom_dual_lora(
                getattr(self.pipe, lora_base_model),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                enable_alpha1=enable_alpha1,
            )
            # TODO 这里好像会重复加载模型
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                # TODO check这里的lora权重加载和全量微调权重加载的不一致问题
                # state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(self.pipe, lora_base_model, model)
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "twinflow": lambda pipe, inputs_shared, inputs_posi, inputs_nega: TwinFlowLoss(pipe, **inputs_shared, **inputs_posi),
        }
        if task == "trajectory_imitation":
            # This is an experimental feature.
            # We may remove it in the future.
            self.loss_fn = TrajectoryImitationLoss()
            self.task_to_loss["trajectory_imitation"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)

    def add_custom_dual_lora(self, model, lora_rank, lora_alpha, lora_target_modules="all-linear", enable_alpha1=False):
        if lora_target_modules == "all-linear":
            patterns = ["all-linear"]
        else:
            # 将逗号分隔的字符串拆分为列表
            patterns = [module.strip() for module in lora_target_modules.split(",")]
        if enable_alpha1:
            replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=lora_alpha * lora_rank, alpha2=lora_alpha * lora_rank)
        else:
            print("forze noise branch ******************************************")
            replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=0, alpha2=lora_alpha * lora_rank) # 这里本体和分支都要finetune，也可以只tune条件分支
        return model
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["description"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "condition_image": data["condition_0"],
            "height": data["image"].shape[2],
            "width": data["image"].shape[3],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        if self.task == "trajectory_imitation":
            inputs_shared["cfg_scale"] = 2
            inputs_shared["teacher"] = self.pipe_teacher
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        # print("data shape: ", data["image"].shape)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        # print("inputs len: ", len(inputs))
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss
    
    @torch.no_grad()
    def test_model(self, data_sample, num_inference_steps=1, seed=42):
        """
        测试模型的生成能力，用于训练过程中的验证和训练结束后的测试
        :param data_sample: 数据样本，包含image和condition_0
        :param num_inference_steps: 推理步数
        :param seed: 随机种子
        """
        # 临时切换到评估模式
        train_state = self.training
        self.eval()
        
        # try:
        # 准备测试输入
        inputs_shared, inputs_posi, inputs_nega = self.get_pipeline_inputs(data_sample)
        
        # 确保输入数据在正确的设备上
        inputs_shared = self.transfer_data_to_device(inputs_shared, self.pipe.device, self.pipe.torch_dtype)
        inputs_posi = self.transfer_data_to_device(inputs_posi, self.pipe.device, self.pipe.torch_dtype)
        inputs_nega = self.transfer_data_to_device(inputs_nega, self.pipe.device, self.pipe.torch_dtype)
        
        # 从数据样本中获取图像尺寸信息
        height = inputs_shared.get("height", 512)
        width = inputs_shared.get("width", 512)

        # 获取原始的低质量图像（LQ）和高质量图像（GT）
        lq_image = self.pipe.vae_output_to_image(data_sample["condition_0"])    # 低质量图像
        gt_image = self.pipe.vae_output_to_image(data_sample["image"])          # 高质量图像

        
        # 使用管道进行推理生成
        with torch.no_grad():
            generated_image = self.pipe(
                prompt=inputs_posi.get("prompt", "test"),
                negative_prompt=inputs_nega.get("negative_prompt", ""),
                input_image=inputs_shared.get("input_image"),
                condition_image=inputs_shared.get("condition_image"),
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                rand_device=self.pipe.device,
                cfg_scale=inputs_shared.get("cfg_scale", 1.0),
                use_gradient_checkpointing=self.use_gradient_checkpointing,
                use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            )
        # switch to Training mode
        # self.pipe.scheduler.set_timesteps(1000, training=True)
        self.train()
        return lq_image, generated_image, gt_image
        
        # except Exception as e:
        #     print(f"Model testing failed: {e}")
        # finally:
        #     # 恢复原始训练状态
        #     self.train(train_state)


def z_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--dataset_cond_path", type=str, default=None, help="Path to condition.")
    parser.add_argument("--infer_steps", type=int, default=1)
    parser.add_argument("--infer_seed", type=int, default=42)
    parser.add_argument("--infer_num_samples", type=int, default=1)
    parser.add_argument("--deg_file_path", type=str, default='params.yml')
    parser.add_argument("--dataset_txt_paths", type=str, default='/GPFS/rhome/chenxinzhu/code/zimage/DiffSynth-Studio/diffsynth/extensions/realesrgan/gt_path.txt')
    parser.add_argument("--resolution_ori", type=int, default=512)
    parser.add_argument("--resolution_tgt", type=int, default=512)
    parser.add_argument("--highquality_dataset_txt_paths", type=str, default=None)
    parser.add_argument("--condition_timestep_zero", action="store_true", default=False, help="Whether to use zero timestep for condition branch.")
    parser.add_argument("--enable_alpha1", action="store_true", default=False, help="Whether to use alpha1 in DualLoRA.") # 默认为false，不然怎样都是true了
    parser.add_argument("--alpha", type=float, default=1.0, help="number of alpha/rank in lora.")
    parser.add_argument("--enable_2_temb", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    parser = z_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16", 
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    max_grad_norm = getattr(args, 'max_grad_norm', 0.0)  # 从命令行参数获取，默认为1.0
    dataset = PairedSROnlineTxtDataset(
       split="train", args=args,
    )
    dataset.load_from_cache = False
    # dataset = UnifiedDataset(
    #     base_path=args.dataset_base_path,
    #     metadata_path=args.dataset_metadata_path,
    #     repeat=args.dataset_repeat,
    #     data_file_keys=args.data_file_keys.split(","),
    #     main_data_operator=UnifiedDataset.default_image_operator(
    #         base_path=args.dataset_base_path,
    #         max_pixels=args.max_pixels,
    #         height=args.height,
    #         width=args.width,
    #         height_division_factor=16,
    #         width_division_factor=16,
    #     )
    # )
    model = ZImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.alpha,
        lora_checkpoint=args.lora_checkpoint,
        enable_alpha1=args.enable_alpha1,
        condition_timestep_zero=args.condition_timestep_zero,
        enable_2_temb=args.enable_2_temb,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device=accelerator.device,
    )

    # Initialize wandb if requested
    import wandb
    import time
    if accelerator.is_main_process:
        wandb_run = None
        wandb_init_args = {
            "project": getattr(args, "wandb_project", "z-image"),
            "name": time.strftime("%Y%m%d-%H%M%S")
        }
        if hasattr(args, "wandb_entity") and args.wandb_entity:
            wandb_init_args["entity"] = args.wandb_entity
        if hasattr(args, "wandb_name") and args.wandb_name:
            wandb_init_args["name"] = args.wandb_name
        if hasattr(args, "wandb_group") and args.wandb_group:
            wandb_init_args["group"] = args.wandb_group
            
        wandb_run = wandb.init(**wandb_init_args)

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        args=args,
        accelerator=accelerator,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "trajectory_imitation": launch_training_task,
        "twinflow": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
