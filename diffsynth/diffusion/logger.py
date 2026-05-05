import os, torch
from accelerate import Accelerator

import wandb
from copy import deepcopy
import json
import PIL.Image as Image
from torchvision import transforms

@torch.no_grad()
def get_validation_image(lq_image, result_image_list, gt_image):
    """
    在训练过程中验证模型，不干扰训练状态
    保存拼接的LQ-Result-GT图像
    """
    
    # 确保所有图像都是PIL格式
    if not isinstance(lq_image, Image.Image):
        lq_image = transforms.ToPILImage()(lq_image)
    if not isinstance(gt_image, Image.Image):
        gt_image = transforms.ToPILImage()(gt_image)
    if not isinstance(result_image_list[0], Image.Image):
        raise TypeError('result_image_list must be a list of PIL images')
    
    # 获取图像尺寸
    lq_width, lq_height = lq_image.size
    gt_width, gt_height = gt_image.size
    result_width, result_height = result_image_list[0].size
    
    # 创建一个新的图像，宽度为三个图像宽度之和，高度为最大高度
    total_width = lq_width + result_width * len(result_image_list) + gt_width
    max_height = max(lq_height, result_height, gt_height)
    
    # 创建拼接图像
    concatenated_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # 粘贴图像：从左到右为 LQ, Result, GT
    concatenated_image.paste(lq_image, (0, 0))
    concatenated_image.paste(gt_image, (lq_width, 0))
    for i, result_image in enumerate(result_image_list):
        concatenated_image.paste(result_image, (lq_width + gt_width + result_width * i, 0))
    
    return concatenated_image

class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, args = None, accelerator: Accelerator = None):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

        if accelerator.is_main_process:
            self.run_name = wandb.run.name
            self.output_dir = os.path.join(self.output_path, self.run_name)
            # os.makedirs(output_dir, exist_ok=True)
            self.validation_samples_dir = os.path.join(self.output_dir, "validation_samples")
            os.makedirs(self.validation_samples_dir, exist_ok=True)
            self.models_dir = os.path.join(self.output_dir, "models")
            os.makedirs(self.models_dir, exist_ok=True)

            # 如果提供了args，则保存配置
            if args is not None:
                self.save_args_config(args)
    def save_args_config(self, args):
        """
        保存命令行参数配置
        """
        # 将args对象转换为字典
        if hasattr(args, '__dict__'):
            args_dict = vars(args)
        else:
            args_dict = dict(args) if isinstance(args, dict) else {}
        
        # 保存配置到文件
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(args_dict, f, indent=2, default=str)
        
        # 如果有wandb，也记录到wandb
        if wandb.run is not None:
            wandb.run.config.update(args_dict)

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, loss_components=None, data=None):
        accelerator.wait_for_everyone()
        self.num_steps += 1
        image_path = None
        if save_steps is not None and self.num_steps % save_steps == 0 and accelerator.is_main_process:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
        if (self.num_steps-1) % 100 == 0 and accelerator.is_main_process:
            num_inference_steps=1
            seed = 42

            # 从data中只取第一个样本进行validation
            if isinstance(data, dict) and len(data) > 0:
                # 创建一个只包含第一个样本的数据字典
                first_sample = {}
                for key, value in data.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0:
                        first_sample[key] = value[0] if len(value) > 0 else value
                    elif isinstance(value, torch.Tensor) and value.size(0) > 0:
                        first_sample[key] = value[0:1]  # 保持tensor维度
                    else:
                        first_sample[key] = value
                data_to_use = first_sample
            elif isinstance(data, (list, tuple)) and len(data) > 0:
                # 如果data是列表或元组，取第一个元素
                data_to_use = data[0]
            else:
                # 如果无法确定结构，使用原始数据
                data_to_use = data

            num_inference_steps_list=[1, 4]
            # 1. 准备工作：解包模型，保存模式
            raw_model = accelerator.unwrap_model(model)
            was_training = raw_model.training
            raw_model.eval() # 开启验证模式

            # 2. 隔离 Scheduler：复制一份 pipe 用于验证，绝对安全
            # 如果显存吃紧，也可以用 try-finally 结构在原 pipe 上改，但记得去掉 training=True
            # val_pipe = deepcopy(raw_model.pipe) 

            result_image_list = []
            lq_image, gt_image = None, None # 初始化防止报错

            try:
                for num_inference_steps in num_inference_steps_list:
                    # 使用 val_pipe 的 scheduler 设置步数
                    # val_pipe.scheduler.set_timesteps(num_inference_steps)
                    
                    # 假设 test_model 内部使用 raw_model.pipe，这里可能需要临时替换
                    # 或者直接传入 steps 让 test_model 内部去 set，但要确保它用的是 val_pipe
                    # 这里为了演示，假设 test_model 逻辑不变，我们还是用 try-finally 恢复原 pipe 状态更简单
                    
                    # --- 方案 B：如果不做 deepcopy，就在原 pipe 上操作，但要严谨恢复 ---
                    # raw_model.pipe.scheduler.set_timesteps(1000)
                    
                    lq_image, image, gt_image = raw_model.test_model(
                        data_sample=data_to_use, 
                        num_inference_steps=num_inference_steps, 
                        seed=seed
                    )
                    result_image_list.append(image)

                # 3. 循环结束后，统一生成拼图
                if result_image_list and lq_image is not None:
                    concat_image = get_validation_image(lq_image, result_image_list, gt_image)
                    # 注意：文件名里的 list 转 string 可能会包含空格，建议处理一下格式
                    steps_str = "-".join(map(str, num_inference_steps_list))
                    image_path = os.path.join(self.validation_samples_dir, f"test_valid_{self.num_steps}_step_{steps_str}.jpg")
                    # 保存 concat_image ...
                    concat_image.save(image_path)

            finally:
                # 4. 【至关重要】恢复环境
                # A. 恢复 Scheduler (如果你用的方案 B)
                # 注意：检查你的调度器是否有 set_timesteps(1000) 这种用法，通常训练用不到 set_timesteps
                # 如果训练本身不依赖 set_timesteps 设置的状态，这一步其实可以省略；
                # 但如果依赖，请确保参数正确（去掉 training=True）
                try:
                    raw_model.pipe.scheduler.set_timesteps(1000, training=True)
                except TypeError:
                    # 某些 scheduler 可能需要其他参数，或者根本不需要恢复
                    pass
                    
                # B. 恢复模型训练模式
                if was_training:
                    print("恢复模型训练模式...")
                    raw_model.train()
                raw_model.train()
                    
        # Log to wandb if available
        if accelerator.is_main_process and wandb.run is not None:
            log_dict = {
                "validation_sample": wandb.Image(image_path) if image_path is not None else None,
                "step": self.num_steps
            }
            # Add loss components if available
            if loss_components is not None:
                log_dict.update(loss_components)
            wandb.log(log_dict)



    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        
        # [修复] 同样移出来
        state_dict = accelerator.get_state_dict(model)
        
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)

            path = os.path.join(self.models_dir, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        
        # [修复] get_state_dict 必须由所有进程调用，不能只在 main_process 里调用
        # 这一步涉及到多卡之间的通信和权重汇聚
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model) 

        if accelerator.is_main_process:
            # 只有主进程负责处理业务逻辑（如过滤前缀）和写入磁盘
            # 注意：这里我们使用已经 unwrap 过的 model 来调用 export 方法
            # 如果 export_trainable_state_dict 是 model 自身的方法，需要用 unwrapped_model
            state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            
            os.makedirs(self.models_dir, exist_ok=True)
            path = os.path.join(self.models_dir, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
