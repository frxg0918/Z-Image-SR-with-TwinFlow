import torch, os, argparse, accelerate, copy
from datasets_utils.dataset import PairedSROnlineTxtDataset
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.core import load_state_dict
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

    def _replace_module(parent, name_prefix=""):
        for name, module in list(parent.named_children()):
            full_name = f"{name_prefix}{name}"
            if isinstance(module, torch.nn.Linear):
                module.weight.data = to_fp8(module.weight.data)
                if module.bias is not None:
                    module.bias.data = to_fp8(module.bias.data)

                if any(p in full_name for p in patterns):
                    # 替换为 DualLoRALinear
                    new_module = DualLoRALinear(module, rank, alpha1, alpha2)
                    setattr(parent, name, new_module)
                    print(f"[lora] {full_name} -> DualLoRALinear")
                else: 
                    # 替换为自动精度转换的 FP8LinearWrapper
                    setattr(parent, name, module)
                    print(f"[cast] {full_name} -> Origin")
            else:
                _replace_module(module, name_prefix=full_name + ".")
    
    _replace_module(model)

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
            y = self._base_linear(x)  # (B, C_out)
            delta1 = self.lora_B1(self.lora_A1(x)) * self.scaling1
            delta2 = self.lora_B2(self.lora_A2(x)) * self.scaling2
            y1 = y + delta1
            y2 = y + delta2
            return torch.stack([y1, y2], dim=1)  # (B, 2, C_out)
        else:
            B, L2, _ = x.shape
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

            
class ZImageTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
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
        self.pipe = ZImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

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
                lora_rank=lora_rank
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
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
        }
        if task == "trajectory_imitation":
            # This is an experimental feature.
            # We may remove it in the future.
            self.loss_fn = TrajectoryImitationLoss()
            self.task_to_loss["trajectory_imitation"] = self.loss_fn
            self.pipe_teacher = copy.deepcopy(self.pipe)
            self.pipe_teacher.requires_grad_(False)

    def add_custom_dual_lora(self, model, lora_rank):
        patterns = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "w1",
            "w2",
            "w3"
        ]
        replace_linear_with_duallora(model, patterns, rank=lora_rank, alpha1=0, alpha2=lora_rank)
        return model
        
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["text"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "condition_image": data["conditioning_image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
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
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def z_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--dataset_cond_path", type=str, default=None, help="Path to condition.")
    parser.add_argument("--infer_steps", type=int, default=100)
    parser.add_argument("--infer_seed", type=int, default=42)
    parser.add_argument("--infer_num_samples", type=int, default=1)
    parser.add_argument("--deg_file_path", type=str, default='params.yml')
    parser.add_argument("--dataset_txt_paths", type=str, default='/GPFS/rhome/chenxinzhu/code/zimage/DiffSynth-Studio/diffsynth/extensions/realesrgan/gt_path.txt')
    parser.add_argument("--highquality_dataset_txt_paths", type=str, default=None)

    return parser


if __name__ == "__main__":
    parser = z_image_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16", 
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = PairedSROnlineTxtDataset(
       split="train", args=args,
    )
    model = ZImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
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
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
        "trajectory_imitation": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
