from .base_pipeline import BasePipeline
import torch

from torch import nn
from copy import deepcopy
from typing import List, Union
from collections import OrderedDict
import numpy as np
from icecream import ic

DEBUG = True
@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """
    Step the EMA model parameters towards the current model parameters.
    
    Args:
        ema_model (nn.Module): The exponential moving average model (Teacher).
        model (nn.Module): The current training model (Student).
        decay (float): The decay rate for the moving average.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            # param_ema = decay * param_ema + (1 - decay) * param_curr
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)

def logit_normal_timestep_sampler(device, max_timestep_boundary=1000.0):
    """
    return sigmoid(randn()) * max_timestep_boundary
    """
    normal_samples = torch.randn((1,), dtype=torch.float32, device=device)
    t_01 = torch.sigmoid(normal_samples)
    timestep = t_01 * max_timestep_boundary
    return timestep

def uniform_timestep_sampler(device, max_timestep_boundary=1000.0):
    """
    return rand * max_timestep_boundary
    """
    samples = torch.rand((1,), dtype=torch.float32, device=device)
    timestep = samples * max_timestep_boundary
    return timestep

class RCGM(torch.nn.Module):
    """
    Recursive Consistent Generation Model (RCGM).

    This class implements the backbone for 'Any-step Generation via N-th Order 
    Recursive Consistent Velocity Field Estimation'. It serves as the foundation 
    for consistency training by estimating higher-order trajectories.

    References:
        - RCGM: https://github.com/LINs-lab/RCGM/blob/main/assets/paper.pdf
        - UCGM (Sampler): https://arxiv.org/abs/2505.07447 (Unified Continuous Generative Models)
    """
    def __init__(
        self,
        ema_decay_rate: float = 0.99, # Recomended: >=0.99 for estimate_order >=2
        estimate_order: int = 2,
        enhanced_ratio: float = 0.0,
        pipe: BasePipeline = None,
        **kwargs,
    ):
        super().__init__()

        self.emd = ema_decay_rate
        self.eso = estimate_order # N-th order estimation (RCGM paper)
        
        assert self.eso >= 1, "Only support estimate_order >= 1"

        self.cmd = 0
        self.mod = None # EMA Model container
        self.enr = enhanced_ratio # CFG Guidance ratio
        self.pipe = pipe

    def alpha_in(self, t): return t/1000.0          # Coefficient for noise z
    def gamma_in(self, t): return 1 - t/1000.0      # Coefficient for data x
    def alpha_to(self, t): return 1          # d(alpha)/dt
    def gamma_to(self, t): return -1         # d(gamma)/dt
    def l2_loss(self, pred, target):
        """Standard L2 (MSE) Loss flattened over spatial dimensions."""
        loss = (pred.float() - target.float()) ** 2
        return loss.flatten(1).mean(dim=1).to(pred.dtype)

    def loss_func(self, pred, target):
        return self.l2_loss(pred, target)

    @torch.no_grad()
    def get_refer_predc(
        self,
        rng_state: torch.Tensor,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: List[torch.Tensor],
    ):
        """
        Get reference predictions with and without conditions (Classifier-Free Guidance).
        Restores RNG state to ensure noise consistency between forward passes.
        """
        torch.cuda.set_rng_state(rng_state)
        # Unconditional forward (using empty condition 'e')
        refer_x, refer_z, refer_v, _ = self.forward(model, x_t, t, tt, **dict(c=e))
        
        torch.cuda.set_rng_state(rng_state)
        # Conditional forward (using condition 'c')
        predc_x, predc_z, predc_v, _ = self.forward(model, x_t, t, tt, **dict(c=c))
        
        return refer_x, refer_z, refer_v, predc_x, predc_z, predc_v

    @torch.no_grad()
    def enhance_target(
        self,
        target: torch.Tensor,
        ratio: float,
        pred_w_c: torch.Tensor,
        pred_wo_c: torch.Tensor,
    ):
        """
        Enhance the training target using Classifier-Free Guidance (CFG).
        Target' = Target + w * (Prediction_cond - Prediction_uncond)
        """
        target = target + ratio * (pred_w_c - pred_wo_c)
        return target

    @torch.no_grad()
    def prepare_inputs(
        self,
        model,
        # c: List[torch.Tensor] = None,
        inputs,
        max_timestep_boundary: int = 1000,
        min_timestep_boundary: int = 0,
        e2e=False,
    ):
        """
        Prepare inputs for Flow Matching training.
        Constructs x_t (noisy data) and target vector field.
        """
        pipe = self.pipe
        device = pipe.device
        dtype = pipe.torch_dtype
        bsz = inputs["input_latents"].shape[0]
        # TODO：确定这里的EPS
        EPS = 1e-3
        # ============================================================
        # 1. Logit-Normal 采样 (让 t 集中在 0.5 附近)
        # ============================================================
        # loc=0, scale=1.0 是标准设置。
        # Sigmoid 将 (-inf, inf) 映射到 (0, 1)
        # 然后映射到 [0, 1000]
        # TODO：确定这里的直接e2e，或是从中间随机一步到0
        if e2e:
            timestep = torch.full((1,), 1000.0, device=device, dtype=torch.float32)
            timestep = uniform_timestep_sampler(device)
            timestep = timestep.clamp_min(10.0)
            tt = torch.full((1,), 0.0, device=device, dtype=timestep.dtype)
        else:
            # 这里clamp到10.0，防止RCGM输入的x_t_m = x_t - 0.01 * target，出现一个不合理的noisy latents，虽然实际不起作用
            # timestep = logit_normal_timestep_sampler(device)
            timestep = uniform_timestep_sampler(device)
            timestep = timestep.clamp_min(10.0)
            # Aux time variable tt < t for consistency estimation
            # 这里的t和tt的选取中加入了EPS，是防止t和tt相等出现，除0导致loss出现nan
            # 这里EPS最起码是3，如果时间步归一化到1，那么EPS可能需要大于等于4，这里的精度挺麻烦的反正
            tt = (torch.rand_like(timestep)) * (timestep - EPS)
            tt = tt.to(timestep.dtype) # 数值范围可能因为这个，tt如果取到t，后面会除以0，loss出现nan
        noise = torch.randn_like(inputs["input_latents"])
        sigma = timestep / 1000.0
        sigma = sigma.view(-1,1,1,1)
        x_real_t = (1 - sigma) * inputs["input_latents"] + sigma * noise
        training_target = noise - inputs["input_latents"]
        # inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        # training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
        if DEBUG:
            ic(timestep)
            ic(tt)
        return x_real_t, noise, timestep, tt, training_target

    @torch.no_grad()
    def multi_fwd(
        self,
        rng_state: torch.Tensor,
        model,
        inputs,
        models,
        # x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        N: int,
    ):
        """
        Used to calculate the recursive consistency target.
        """
        pred = 0
        ts_float = [(t * (1 - i / (N)) + tt * (i / (N))) for i in range(N + 1)] # if N=2, t0=t, t1=0.5t+0.5tt, t2=tt
        ts = [val for val in ts_float]
        
        # Euler integration loop
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            if DEBUG:
                ic(f"t_c: {t_c}, t_n: {t_n}")
            torch.cuda.set_rng_state(rng_state)
            hx, hz, F_c, _ = self.forward(model, inputs, models, t_c, t_n)

            inputs["latents"] = self.alpha_in(t_n) * hz + self.gamma_in(t_n) * hx
            inputs["latents"] = inputs["latents"].to(F_c.dtype)
            pred = pred + F_c * (t_c - t_n)/1000.0
            
        return hx, hz, pred

    @torch.no_grad()
    def get_rcgm_target_bak(
        self,
        model,
        inputs,
        models,
        rng_state: torch.Tensor,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        # x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        N: int,
    ):
        """
        Calculates the RCGM consistency target using N-th order estimation.
        
        Ref: 'Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation'
        Uses a small temporal perturbation (Delta t = 0.01) to enforce local consistency.
        """
        x_t = inputs["latents"]

        # Delta t = 0.01 as mentioned in RCGM paper
        t_m = (t - 0.01).clamp_min(tt)
        x_t = x_t - target * 0.01 # First order step
        
        # N-step integration from t_m to tt
        inputs["latents"] = x_t
        _, _, Ft_tar = self.multi_fwd(rng_state, model, inputs, models, t_m, tt, N)
        
        # Weighting for boundary conditions near t=tt
        mask = t < (tt + 0.01)
        cof_l = torch.where(mask, torch.ones_like(t), 100 * (t - tt))
        cof_r = torch.where(mask, 1 / (t - tt), torch.ones_like(t) * 100)
        
        # Reconstruct velocity field target from integral
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - (Ft_tar).clamp(min=-1.0, max=1.0)
        return Ft_tar
    
    @torch.no_grad()
    def get_rcgm_target(
        self,
        model,
        inputs,
        models,
        rng_state: torch.Tensor,
        F_th_t: torch.Tensor,
        target: torch.Tensor,
        # x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        N: int,
    ):
        """
        Calculates the RCGM consistency target using N-th order estimation.
        
        Ref: 'Any-step Generation via N-th Order Recursive Consistent Velocity Field Estimation'
        Uses a small temporal perturbation (Delta t = 0.01) to enforce local consistency.
        """

        # Delta t = 0.01 as mentioned in RCGM paper
        t_m = (t - 10.0).clamp_min(tt)
        # print("t_m = ", t_m)
        x_t = inputs["latents"] - target * 0.01 # First order step
        
        # N-step integration from t_m to tt
        inputs["latents"] = x_t
        _, _, Ft_tar = self.multi_fwd(rng_state, model, inputs, models, t_m, tt, N)
        # print("F_th_t:", F_th_t) 这个数值范围对的
        # print("Ft_tar before detach:", Ft_tar) 这个数值范围不对
        
        # Weighting for boundary conditions near t=tt
        mask = t < (tt + 10.0)
        cof_l = torch.where(mask, torch.ones_like(t), 0.1 * (t - tt))
        cof_r = torch.where(mask, 1000.0 / (t - tt), torch.ones_like(t) * 100.0)
        
        # Reconstruct velocity field target from integral
        Ft_tar = (F_th_t * cof_l - Ft_tar * cof_r) - target
        Ft_tar = F_th_t.data - Ft_tar.clamp(min=-1.0, max=1.0)
        return Ft_tar

    @torch.no_grad()
    def update_ema(
        self,
        model: nn.Module,
    ):
        """Updates the EMA (Teacher) model."""
        if self.emd > 0.0 and self.emd < 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()
            update_ema(self.mod, model, decay=self.cmd)
            # Warmup logic for EMA decay
            self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
        elif self.emd == 0.0:
            self.mod = model
        elif self.emd == 1.0:
            self.mod = self.mod or deepcopy(model).requires_grad_(False).train()

    def forward(
        self,
        model,
        inputs,
        models,
        t: torch.Tensor,
        tt: Union[torch.Tensor, None] = None,
    ):
        """
        Forward pass.
        Returns:
            x_hat: Reconstructed data (x0)
            z_hat: Reconstructed noise (x1)
            F_t: Predicted velocity field v_t
            dent: Denominator (normalization term)
        """
        dent = -1 # dent = alpha(t)*gamma'(t) - gamma(t)*alpha'(t) for linear flow
        # 处理输入数据格式以适应DiffSynth框架
        # t_flat = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        # tt_flat = torch.ones(x_t.size(0), device=x_t.device) * tt.flatten() if tt is not None else torch.zeros(x_t.size(0), device=x_t.device)
        
        F_t = model(**models, **inputs, timestep=t, target_timestep=tt)
        t = torch.abs(t)
        
        # Invert flow to recover x and z
        x_t = inputs["latents"]
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent


def TwinFlowLoss(pipe: BasePipeline, **inputs):
    """
    实现TwinFlow损失函数，结合RCGM一致性训练和TwinFlow特定的对抗及修正损失
    
    Args:
        pipe: 基础管道对象
        **inputs: 输入参数字典，应包含input_latents等必要信息
        
    Returns:
        总损失值，包括RCGM基础损失、对抗损失和修正损失
    """
    # 初始化RCGM模块
    rcgm = RCGM(
        ema_decay_rate=0.99,
        estimate_order=2,
        enhanced_ratio=0.0,
        pipe=pipe,
    )   
    
    # 准备模型
    model = pipe.model_fn
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    
    if model is None:
        raise ValueError("无法找到模型用于训练")
    
    # -----------------------------------------------------------
    # loss any
    # -----------------------------------------------------------

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)) # 这边手动设定1000步，防止因为测试时validation导致scheduler被重置
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)) # 
    
    if DEBUG:
        ic(max_timestep_boundary, min_timestep_boundary)
    # TwinFlow特定参数
    using_twinflow = inputs.get("using_twinflow", True)
    using_twinflow = True
    
    # 准备输入
    inputs["latents"], z, t, tt, target = rcgm.prepare_inputs(model, inputs, max_timestep_boundary, min_timestep_boundary) # 0-1000
    
    # 获取随机状态以保证一致性
    rng_state = torch.cuda.get_rng_state()
    
    # 前向传播获得模型预测
    _, _, F_th_t, _ = rcgm.forward(model, inputs, models, t, tt)
    # print("F_th_t:", F_th_t)
    
    # -----------------------------------------------------------
    # 1. RCGM基础损失 (L_base) any
    # -----------------------------------------------------------
    # 当t-10<tt时，以下方法实际返回的是training_target
    rcgm_target = rcgm.get_rcgm_target(
        model, inputs, models, rng_state, F_th_t, target.clone(), t, tt, rcgm.eso,
    )
    # print("rcgm_target:", rcgm_target)
    # loss_base = rcgm.loss_func(F_th_t, rcgm_target).mean()
    loss_base = torch.nn.functional.mse_loss(F_th_t.float(), rcgm_target.float())
    weighting_base = (torch.tan((1 - (t.abs() - tt.abs())/1000.0) * np.pi / 2.5) + 1).flatten()
    # raise Exception("RCGM损失计算完成") # 到这里时对的
    # if DEBUG:
    #     print("RCGM loss computed.")
    # if DEBUG:
    #     print("loss_base: ", loss_base.item())
    if torch.isnan(loss_base) :
        raise Exception("RCGM损失计算错误, 出现nan")
    
    # 如果不使用TwinFlow，则只返回RCGM损失
    if not using_twinflow:
        # 返回总损失和各分项损失的字典
        loss_components = {
            "total_loss": 0,
            "loss_base": loss_base.item(),
            "loss_adv": 0,
            "loss_rectify": 0
        }
        return (loss_base, loss_components)
    
    # -----------------------------------------------------------
    # loss e2e
    # -----------------------------------------------------------
    # 准备输入
    enable_e2e = True
    loss_e2e = torch.tensor(0.0, device=t.device)
    inputs["latents"], z, t_e2e, tt_e2e, target_e2e = rcgm.prepare_inputs(model, inputs, max_timestep_boundary, min_timestep_boundary, e2e=True) # directly 1000 -> 0
    
    # 获取随机状态以保证一致性
    rng_state = torch.cuda.get_rng_state()
    
    # 前向传播获得模型预测
    x_fake, _, F_fake, _ = rcgm.forward(model, inputs, models, t_e2e, tt_e2e)
    weighting_pixel = 1 - t_e2e/1000.0
    if enable_e2e:
        # -----------------------------------------------------------
        # 1. RCGM基础损失 (L_base) any
        # -----------------------------------------------------------
        # 当t-10<tt时，以下方法实际返回的是training_target
        rcgm_target_e2e = rcgm.get_rcgm_target(
            model, inputs, models, rng_state, F_fake, target_e2e.clone(), t_e2e, tt_e2e, rcgm.eso,
        )
        # print("rcgm_target:", rcgm_target)
        # loss_base = rcgm.loss_func(F_th_t, rcgm_target).mean()
        loss_e2e = torch.nn.functional.mse_loss(F_fake.float(), rcgm_target_e2e.float())
    
    # -----------------------------------------------------------
    # TwinFlow 特定损失
    # -----------------------------------------------------------
    
    # [可选 flowmatching 损失]
    enable_flowmatching = True
    loss_flowmatching = torch.tensor(0.0, device=t.device)
    weighting_mul = torch.tensor(0.0, device=t.device)
    if enable_flowmatching:
        # TODO 这里flow matching loss不知道可不可以选择重新采样新的noise和timestep，更正，可以采样新的noise和timestep
        noise_fm = torch.randn_like(inputs["input_latents"])
        timestep_fm = uniform_timestep_sampler(pipe.device)
        tt_fm = timestep_fm - torch.rand_like(timestep_fm) * timestep_fm * 0.05
        ic(timestep_fm, tt_fm)
        sigma_fm = timestep_fm / 1000.0
        sigma_fm = sigma_fm.view(-1,1,1,1)
        x_real_t = (1 - sigma_fm) * inputs["input_latents"] + sigma_fm * noise_fm
        training_target = noise_fm - inputs["input_latents"]
        # x_real_t = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        # training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
        
        # 获取模型预测
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        inputs["latents"] = x_real_t
        v_real = pipe.model_fn(**models, **inputs, timestep=timestep_fm, target_timestep=tt_fm)
        
        # Flow matching loss计算
        loss_flowmatching = torch.nn.functional.mse_loss(v_real.float(), training_target.float())
        weighting_mul = (torch.tan((1 - (timestep_fm.abs() - tt_fm.abs())/1000.0) * np.pi / 2.5) + 1).flatten()

    # 生成假样本 x_fake = x_real_t - t*v_real
    # print("x_real_t:", x_real_t.shape)
    # print("v_real:", v_real.shape)
    # print("timestep:", timestep.shape)
    # x_real_t: torch.Size([1, 16, 96, 96])
    # v_real: torch.Size([1, 16, 96, 96])
    # timestep: torch.Size([1])

    # TODO 这里据原文所说，可以采样新的noise，另外这里是不是可以任意采时间步都一步出图，不是非要纯噪
    if x_fake is None:
        noise = torch.randn_like(inputs["input_latents"])
        timestep = t
        # 用下面这个时间步的话是纯噪声一步出假样本
        temp_t = torch.full_like(timestep, 1000.0)
        sigma_temp = temp_t / 1000.0 
        sigma_temp = sigma_temp.view(-1,1,1,1)
        inputs["latents"] = sigma_temp * noise + (1 - sigma_temp) * inputs["input_latents"]
        if DEBUG:
            print("temp_t:", temp_t)
        temp_t_zero = torch.full_like(timestep, 0.0)
        if DEBUG:
            print("calculating x_fake #######################################")
        F_fake = pipe.model_fn(**models, **inputs, timestep=temp_t, target_timestep=temp_t_zero) # 直接从temp_t一步到0
        # scheduler的方法都不精准的，都是用的预定义的时间步和sigma
        # x_fake = pipe.scheduler.step(F_fake, temp_t, noise, to_final=True)
        x_fake = inputs["latents"] - sigma_temp * F_fake
    else:
        F_fake = F_fake
        x_fake = x_fake
    # -----------------------------------------------------------
    # [核心修改] 2. 像素域损失 (Pixel MSE + LPIPS)
    # -----------------------------------------------------------
    loss_pixel_mse = torch.tensor(0.0, device=t.device)
    loss_pixel_lpips = torch.tensor(0.0, device=t.device)
    
    # 获取 GT 图像 (像素域)
    # 假设 Dataset 返回的 "input_image" 已经是 Tensor [B, 3, H, W] 且范围在 [-1, 1]
    gt_pixel = inputs.get("gt_rgb", None) # 对应 Dataset 中的 GT Key
    # gt_pixel = None
    
    if gt_pixel is not None:
        # 解码 x_fake (Latent -> Pixel)
        
        # VAE Decode
        # 注意: 这里的 x_fake 带有梯度，必须保留梯度流过 Decoder
        # Flux/SD VAE 通常不需要额外的 scaling_factor，如果你的模型需要 (如 SDXL 0.13025)，请在此处乘上
        # pred_pixel = pipe.vae_decoder(x_fake / pipe.vae.config.scaling_factor) 
        pred_pixel = pipe.vae_decoder(x_fake)
        
        # 确保尺寸一致 (防止 VAE 下采样取整导致的 1px 误差)
        # if pred_pixel.shape[-2:] != gt_pixel.shape[-2:]:
        #     pred_pixel = F.interpolate(pred_pixel, size=gt_pixel.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2.1 计算 MSE Loss TODO 已经有latent的rcgm，flowmatching做约束，pixel mse可能不需要，糊的问题可能是因为这个
        loss_pixel_mse = torch.nn.functional.mse_loss(pred_pixel.float(), gt_pixel.float())
        
        # 2.2 计算 LPIPS Loss
        # LPIPS 期望输入范围 [-1, 1]
        # 确保 pred_pixel 和 gt_pixel 都在 [-1, 1]
        # 如果你的 VAE 输出是 [0, 1]，需要转换: img = img * 2 - 1
        # 假设这里已经是 [-1, 1]
        
        # 确保 LPIPS 模型在正确的 dtype (通常 float32 比较稳，bf16 可能不稳定)
        # 如果 pipe.loss_fn_lpips 是 fp32，输入也转 fp32
        loss_pixel_lpips = pipe.loss_fn_lpips(pred_pixel.float(), gt_pixel.float()).mean()
        
        # if DEBUG:
        #     print(f"Pixel MSE: {loss_pixel_mse.item():.4f}, LPIPS: {loss_pixel_lpips.item():.4f}")
            
    else:
        pass
        # print("Warning: 'input_image' not found in inputs, skipping pixel losses.")


    # Loss_adv 计算开始 -----------------------------------------------------------------------------------------------
    # 采样新的时间步和噪声
    # timestep_id_2 = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    # timestep_2 = pipe.scheduler.timesteps[timestep_id_2].to(dtype=pipe.torch_dtype, device=pipe.device)
    # timestep_2 = logit_normal_timestep_sampler(pipe.device)
    timestep_2 = uniform_timestep_sampler(pipe.device)
    timestep_2 = 0.96 * timestep_2 + 20.0 # 防止假数据流采样到0，导致时间步处理走了真数据流逻辑
    z_fake = torch.randn_like(x_fake)
    ic(timestep_2)

    # 添加噪声到假样本
    sigma_timestep_2 = timestep_2 / 1000.0
    sigma_timestep_2 = sigma_timestep_2.view(-1,1,1,1)
    # x_t_fake = pipe.scheduler.add_noise(x_fake.detach(), z_fake, timestep_2)
    x_t_fake = (1 - sigma_timestep_2) * x_fake.detach() + sigma_timestep_2 * z_fake
    
    # 计算假样本的velocity
    timestep_negative = -1.0 * timestep_2
    inputs["latents"] = x_t_fake
    F_th_t_fake = pipe.model_fn(**models, **inputs, timestep=timestep_negative)
    
    # 对抗损失 loss_adv = d(v_fake, z_fake-x_fake)
    target_adv = z_fake - x_fake.detach()
    loss_adv = torch.nn.functional.mse_loss(F_th_t_fake.float(), target_adv.float())
    # if DEBUG:
    #     print("Adversarial loss computed.")
    # Loss_adv 计算结束 -----------------------------------------------------------------------------------------------
    
    # 修正损失计算 -------------------------------------------------------------------------------------------------
    # delta_v = v_fake - model(x_fake_t', t')
    with torch.no_grad():
        noise_new = torch.randn_like(inputs["input_latents"])
        # 限制 timestep_new 在 20 到 980 之间 (实际值)
        timestep_new = uniform_timestep_sampler(pipe.device)
        timestep_new = 0.96 * timestep_new + 20.0 # 从0到1000缩放到20到980之间
        # valid_timesteps = pipe.scheduler.timesteps[(pipe.scheduler.timesteps >= 20) & (pipe.scheduler.timesteps <= 980)]
        # if len(valid_timesteps) > 0:
        #     timestep_new_idx = torch.randint(0, len(valid_timesteps), (1,))
        #     timestep_new = valid_timesteps[timestep_new_idx].to(dtype=pipe.torch_dtype, device=pipe.device)
        # else:
        #     # 如果没有在范围内的时间步，使用原始方法
        #     timestep_new_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        #     timestep_new = pipe.scheduler.timesteps[timestep_new_id].to(dtype=pipe.torch_dtype, device=pipe.device)

        
        if DEBUG:
            ic(timestep_new)
        
        # x_tnew_fake = pipe.scheduler.add_noise(x_fake, noise_new, timestep_new)
        sigma_timestep_new = timestep_new / 1000.0
        sigma_timestep_new = sigma_timestep_new.view(-1,1,1,1)
        x_tnew_fake = (1 - sigma_timestep_new) * x_fake + sigma_timestep_new * noise_new
        inputs["latents"] = x_tnew_fake
        fake_v = pipe.model_fn(**models, **inputs, timestep= -1.0 * timestep_new)
        real_v = pipe.model_fn(**models, **inputs, timestep=timestep_new)
        F_grad = fake_v - real_v

        real_pred_x0 = x_tnew_fake - sigma_timestep_new * real_v
        weighting_rectify = torch.clamp(1.0 / (real_pred_x0 - x_fake).abs().mean().detach(), max=10.0)

    sg_target = (F_fake - F_grad).detach()
    
    # loss_rectify = d(v_real, sg_target)
    loss_rectify = torch.nn.functional.mse_loss(F_fake.float(), sg_target.float())
    # if DEBUG:
    #     print("Rectification loss computed.")
    # 修正损失计算结束 -------------------------------------------------------------------------------------------------
    
    # 总损失 
    # total_loss = 1.0 * loss_base + 0.25 * loss_adv + 0.25 * loss_rectify
    # total_loss = 1.0 * loss_base + 1.0 * loss_adv + 1.0 * loss_rectify + 2.0 * loss_pixel_lpips + 1.0 * loss_pixel_mse
    # total_loss = 1.0 * loss_base + 1.0 * loss_adv + 1.0 * loss_rectify + 1.0 * loss_pixel_lpips + 0.5 * loss_pixel_mse
    weighting_base = 1.0 * 1
    weighting_mul = 1.0 * 0.25
    weighting_rectify = 0.5 * torch.tensor(0.5)
    weighting_pixel = torch.tensor(1.0) * weighting_pixel
    weighting_e2e = 1.0 * 0.25
    weighting_adv = 1.0 * 0.25
    total_loss = weighting_base * loss_base + \
        weighting_e2e * loss_e2e + \
        weighting_adv * loss_adv + \
            weighting_rectify * loss_rectify + \
                weighting_mul * loss_flowmatching + \
            weighting_pixel * (1.0 * loss_pixel_lpips + 1.0 * loss_pixel_mse)
    # total_loss = 1.0 * loss_base + \
    #     0.25 * loss_adv + \
    #     0.25 * loss_rectify
    
    # 返回总损失和各分项损失的字典
    loss_components = {
        "total_loss": total_loss.item(),
        "loss_base": loss_base.item(),
        "loss_adv": loss_adv.item(),
        "loss_rectify": loss_rectify.item(),
        "loss_mse": loss_pixel_mse.item(),
        "loss_lpips": loss_pixel_lpips.item(),
        "loss_flowmatching": loss_flowmatching.item(),
        "loss_e2e": loss_e2e.item(),
        "weighting_rectify": weighting_rectify.item(),
    }
    
    return (total_loss, loss_components)

# ... rest of the file remains unchanged
def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
