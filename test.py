# test.py
from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
from datasets_utils.dataset import PairedSROnlineTxtDataset
from diffsynth.core import load_state_dict
import torch
import os
from PIL import Image
import argparse
from train import replace_linear_with_duallora, LoraLinear, LinearFP8Wrapper, DualLoRALinear

def load_duallora_weights(model, lora_path):
    """
    Custom function to load DualLoRA weights from checkpoint
    """
    import safetensors
    import torch
    
    # Load the checkpoint
    if lora_path.endswith('.safetensors'):
        state_dict = safetensors.torch.load_file(lora_path)
    else:
        state_dict = torch.load(lora_path, map_location='cpu')
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Create a mapping from the model's named modules to find DualLoRA layers
    duallora_modules = {}
    for name, module in model.named_modules():
        if isinstance(module, DualLoRALinear):
            duallora_modules[name] = module
    
    # Load weights for each DualLoRA module
    loaded_count = 0
    for name, duallora_module in duallora_modules.items():
        # Check for the expected weight keys in the checkpoint
        lora_A1_key = f"{name}.lora_A1.weight"
        lora_B1_key = f"{name}.lora_B1.weight"
        lora_A2_key = f"{name}.lora_A2.weight"
        lora_B2_key = f"{name}.lora_B2.weight"
        
        if lora_A1_key in state_dict and lora_B1_key in state_dict:
            # Load first LoRA set
            duallora_module.lora_A1.weight.data = state_dict[lora_A1_key].to(device)
            duallora_module.lora_B1.weight.data = state_dict[lora_B1_key].to(device)
            loaded_count += 1
            
        if lora_A2_key in state_dict and lora_B2_key in state_dict:
            # Load second LoRA set
            duallora_module.lora_A2.weight.data = state_dict[lora_A2_key].to(device)
            duallora_module.lora_B2.weight.data = state_dict[lora_B2_key].to(device)
            loaded_count += 1
    
    print(f"Loaded DualLoRA weights for {loaded_count} layers")
    return loaded_count

def main():
    parser = argparse.ArgumentParser(description="Test Z-Image model loading and inference with DualLoRA")
    parser.add_argument("--model_paths", type=str, default=None, help="Path to model checkpoints")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--dataset_txt_paths", type=str, default='./datasets_utils/gt_path.txt', help="Path to dataset txt files")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Number of inference steps")
    parser.add_argument("--test_seed", type=int, default=42, help="Random seed for testing")
    parser.add_argument("--save_dir", type=str, default="./test_results", help="Directory to save test results")
    parser.add_argument("--test_samples", type=int, default=5, help="Number of test samples to run")
    parser.add_argument("--lora_rank", type=int, default=64, help="Rank for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0,w1,w2,w3,all_final_layer,adaLN", help="Target modules for LoRA, use 'all-linear' for DualLoRA")
    parser.add_argument("--lora_base_model", type=str, default="dit", help="Base model for LoRA")
    parser.add_argument("--model_id_with_origin_paths", type=str, default="Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors", help="Model ID with origin paths, e.g., 'Tongyi-MAI/Z-Image-Turbo'")
    parser.add_argument("--trainable_models", type=str, default=None, help="Trainable models")
    parser.add_argument("--task", type=str, default="twinflow", help="Task type")
    parser.add_argument("--height", type=int, default=512, help="Height of generated images")
    parser.add_argument("--width", type=int, default=512, help="Width of generated images")
    parser.add_argument("--lora_path", type=str, default="./runs/20260102-120518/models/step-94991.safetensors", help="Path to LoRA weights")  
    parser.add_argument("--deg_file_path", type=str, default='params.yml', help="Path to the degree file")
    parser.add_argument("--resolution_ori", type=int, default=512, help="Original resolution of images")
    parser.add_argument("--resolution_tgt", type=int, default=512, help="Original resolution of images")
    parser.add_argument("--highquality_dataset_txt_paths", type=str, default=None)
    parser.add_argument("--alpha1", type=float, default=1.0, help="Alpha1 value for DualLoRA")
    parser.add_argument("--condition_timestep_zero", action='store_true', help="Use zero timestep for condition branch")
    args = parser.parse_args()

    # 根据train.py中的模型加载逻辑进行初始化
    # 1. 解析模型配置
    model_configs = []
    if args.model_id_with_origin_paths:
        for item in args.model_id_with_origin_paths.split(','):
            parts = item.split(':')
            model_id = parts[0]
            origin_file_pattern = parts[1] if len(parts) > 1 else '*'
            model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))
    
    # 2. 加载tokenizer配置
    tokenizer_config = ModelConfig(model_id="Tongyi-MAI/Z-Image-Turbo", origin_file_pattern="tokenizer/") if args.tokenizer_path is None else ModelConfig(args.tokenizer_path)
    
    # 3. 创建pipeline
    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        condition_timestep_zero=args.condition_timestep_zero  # 根据train.py中的默认设置
    )
    
    # 4. 根据train.py中的逻辑设置训练模式
    pipe.scheduler.set_timesteps(1000, training=False)  # 推理模式
    
    # 5. 冻结不需要的模型（根据train.py中的逻辑）
    trainable_models_list = [] if args.trainable_models is None else args.trainable_models.split(",")
    pipe.freeze_except(trainable_models_list)
    
    # 6. 应用DualLoRA替换（根据train.py中的逻辑）
    if args.lora_base_model is not None:
        if hasattr(pipe, args.lora_base_model) and getattr(pipe, args.lora_base_model) is not None:
            print(f"Applying DualLoRA to {args.lora_base_model} model...")
            model_to_modify = getattr(pipe, args.lora_base_model)
            
            # 解析LoRA目标模块
            if args.lora_target_modules == "all-linear":
                patterns = ["all-linear"]
            else:
                patterns = [module.strip() for module in args.lora_target_modules.split(",")]
            
            # 应用DualLoRA替换
            if args.alpha1:
                replace_linear_with_duallora(model_to_modify, patterns, rank=args.lora_rank, alpha1=args.lora_rank, alpha2=args.lora_rank)
            else:
                replace_linear_with_duallora(model_to_modify, patterns, rank=args.lora_rank, alpha1=0, alpha2=args.lora_rank)
                
            print(f"DualLoRA applied successfully to {args.lora_base_model} model")
        else:
            print(f"No {args.lora_base_model} model in the pipeline. Skipping DualLoRA application.")
    
    # 7. 加载LoRA权重（根据train.py中的逻辑）
    if args.lora_base_model is not None:
        if hasattr(pipe, args.lora_base_model) and getattr(pipe, args.lora_base_model) is not None:
            print(f"Loading LoRA from: {args.lora_path}")
            # 使用自定义的DualLoRA加载函数
            loaded_count = load_duallora_weights(getattr(pipe, args.lora_base_model), args.lora_path)
            print(f"DualLoRA loaded successfully from {args.lora_path}, {loaded_count} layers updated")
        else:
            print(f"No {args.lora_base_model} model in the pipeline. Skipping LoRA loading.")
    
    # 加载测试数据集
    dataset = PairedSROnlineTxtDataset(
        split="train",  # 使用训练集
        args=args,
    )
    dataset.load_from_cache = False

    # 打乱数据集
    import random
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 测试模型
    for i in range(min(args.test_samples, len(dataset))):
        print(f"Testing sample {i+1}/{min(args.test_samples, len(dataset))}")
        
        # 使用打乱后的索引
        data_sample = dataset[indices[i]]
        
        try:
            # 从数据样本中获取图像尺寸信息
            height = data_sample["image"].size[1]
            width = data_sample["image"].size[0]
            
            # 获取原始的低质量图像（LQ）和高质量图像（GT）
            lq_image = data_sample["condition_0"]  # 低质量图像
            gt_image = data_sample["image"]        # 高质量图像
            
            # 使用pipeline进行推理生成
            generated_image = pipe(
                prompt=data_sample.get("description", "test"),
                negative_prompt="",
                input_image=None,  # 对于超分任务，可能不需要input_image
                condition_image=data_sample["condition_0"],  # 使用低质量图像作为条件
                height=height,
                width=width,
                num_inference_steps=args.num_inference_steps,
                seed=args.test_seed,
                rand_device=pipe.device,
                cfg_scale=1.0,  # 根据train.py中的默认设置
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
            
            # 保存测试结果
            sample_dir = os.path.join(args.save_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            
            lq_image.save(os.path.join(sample_dir, f"lq_image.jpg"))
            generated_image.save(os.path.join(sample_dir, f"generated_image.jpg"))
            gt_image.save(os.path.join(sample_dir, f"gt_image.jpg"))
            
            print(f"Saved test results for sample {i+1} in {sample_dir}")
            
        except Exception as e:
            print(f"Failed to test sample {i+1}: {e}")
            continue


if __name__ == "__main__":
    main()