import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.nn import RMSNorm
from ..core.attention import attention_forward
from ..core.gradient import gradient_checkpoint_forward

from icecream import ic

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
DEBUG = False


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(torch.bfloat16))
        return t_emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    # TODO checkthis 加入slice_info
    def forward(self, x, slice_info=None):
        if slice_info is not None:
            return self.w2(self._forward_silu_gating(self.w1(x, slice_info), self.w3(x, slice_info)), slice_info)
        else:
            return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

class Attention(torch.nn.Module):

    def __init__(self, q_dim, num_heads, head_dim, kv_dim=None, bias_q=False, bias_kv=False, bias_out=False):
        super().__init__()
        dim_inner = head_dim * num_heads
        kv_dim = kv_dim if kv_dim is not None else q_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(q_dim, dim_inner, bias=bias_q)
        self.to_k = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_v = torch.nn.Linear(kv_dim, dim_inner, bias=bias_kv)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim_inner, q_dim, bias=bias_out)])

        self.norm_q = RMSNorm(head_dim, eps=1e-5)
        self.norm_k = RMSNorm(head_dim, eps=1e-5)
    
    #[修改] 增加attn_mask, slice_info参数
    def forward(self, hidden_states, freqs_cis, attn_mask=None, slice_info=None):
        if slice_info is not None:
            query = self.to_q(hidden_states, slice_info)
            key = self.to_k(hidden_states, slice_info)
            value = self.to_v(hidden_states, slice_info)
        else:
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.num_heads, -1))
        key = key.unflatten(-1, (self.num_heads, -1))
        value = value.unflatten(-1, (self.num_heads, -1))

        # Apply Norms
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)  # todo

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Compute joint attention
        # [修改]: 将 attn_mask 传入 attention_forward
        # 假设底层的 attention_forward 支持 attention_mask 参数 (标准的 diffusers/torch 都支持)
        hidden_states = attention_forward(
            query,
            key,
            value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            attn_mask=None,
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        if slice_info is not None:
            output = self.to_out[0](hidden_states, slice_info)
        else:
            output = self.to_out[0](hidden_states)
        if len(self.to_out) > 1:  # dropout
            output = self.to_out[1](output)

        return output


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            q_dim=dim,
            num_heads=n_heads,
            head_dim=dim // n_heads,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        # TODO 需要去掉adaLN的sequential结构，重写loadstatedict逻辑
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        slice_info: Tuple[int, int, int] = None, # [新增] 传入 (L_noise, L_cond, L_text)
    ):
        # 解包长度信息
        # 假设 batch 内长度一致 (padding 后)，如果不一致需要更复杂的 mask 逻辑
        if slice_info is not None:
            L_noise, L_cond, L_text = slice_info 
            total_len = x.shape[1]
            assert total_len == L_noise + L_cond + L_text, "sequence length error"    
        
        if self.modulation:
            assert adaln_input is not None
            # ... (adaln_input 处理逻辑不变) ...
            
            adaln_slice_info = None
            if adaln_input.ndim == 3 and adaln_input.shape[1] == 2:
                # 显式告诉 MaskedLoRA: 前1个是 Base，后1个加 LoRA
                adaln_slice_info = (1, 1, 0)
            # [B, 4*dim] 或 [B, 2, 4*dim]
            adaln_output = self.adaLN_modulation(adaln_input) 
            
            # [核心逻辑修改开始] -------------------------------------------------
            if adaln_output.ndim == 3: # 双分支模式
                B, num_branches, _ = adaln_output.shape
                
                # 1. 拆分参数为 Noise Set 和 Cond Set
                scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_output.chunk(4, dim=-1)
                
                # params: [B, 1, Dim]
                s_msa_noise, s_msa_cond = scale_msa[:, 0:1], scale_msa[:, 1:2]
                g_msa_noise, g_msa_cond = gate_msa[:, 0:1], gate_msa[:, 1:2]
                s_mlp_noise, s_mlp_cond = scale_mlp[:, 0:1], scale_mlp[:, 1:2]
                g_mlp_noise, g_mlp_cond = gate_mlp[:, 0:1], gate_mlp[:, 1:2]
                
                # 2. 获取长度
                if slice_info is not None:
                    L_noise, L_cond, L_text = slice_info
                else:
                    # Fallback: 如果没有传长度，依然假设对称 (仅用于旧逻辑兼容)
                    total = x.shape[1]
                    L_noise, L_cond, L_text = total // 2, total // 2, 0 
                
                # 3. 精准广播 (Exact Broadcasting)
                # Noise Token 使用 Noise 参数
                # Cond Token 使用 Cond 参数
                # Text Token 使用 Cond 参数 (通常文本属于条件控制侧)
                
                def make_param(p_noise, p_cond):
                    part_n = p_noise.expand(-1, L_noise, -1)
                    part_c = p_cond.expand(-1, L_cond, -1)
                    part_t = p_noise.expand(-1, L_text, -1)
                    return torch.cat([part_n, part_c, part_t], dim=1)

                scale_msa = make_param(s_msa_noise, s_msa_cond)
                gate_msa  = make_param(g_msa_noise, g_msa_cond)
                scale_mlp = make_param(s_mlp_noise, s_mlp_cond)
                gate_mlp  = make_param(g_mlp_noise, g_mlp_cond)
                
            else:
                # 单分支逻辑 (不变)
                scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_output.chunk(4, dim=-1)
                # unsqueeze to [B, 1, Dim] for broadcasting
                scale_msa, gate_msa, scale_mlp, gate_mlp = \
                    scale_msa.unsqueeze(1), gate_msa.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

            # 激活函数
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            
            # [核心逻辑修改结束] -------------------------------------------------

            # 定义 slice_info 用于传入 MaskedLoRA
            slice_info = slice_info if adaln_output.ndim == 3 else None

            # Attention
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                slice_info=slice_info # [新增]
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN
            ffn_out = self.feed_forward(
                self.ffn_norm1(x) * scale_mlp,
                slice_info=slice_info # [新增]
            )
            x = x + gate_mlp * self.ffn_norm2(ffn_out)

        else:
            # 无 modulation 分支 (Context Refiner)
            # 这一路通常不需要 LoRA (或者全量微调)
            # 假设不需要 LoRA
            attn_out = self.attention(
                self.attention_norm1(x),
                freqs_cis=freqs_cis,
                attn_mask=attn_mask,
                slice_info=None,
            )
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x
        # if self.modulation:
        #     assert adaln_input is not None
        #     # adaln_input shape: [B, adaln_dim] for single branch or [B, 2, adaln_dim] for dual branch
        #     # 处理 adaln_input 是张量列表的情况
        #     if isinstance(adaln_input, list):
        #         if len(adaln_input) == 2:
        #             # 双分支情况：沿维度1堆叠得到 [B, 2, adaln_dim]
        #             adaln_input = torch.stack(adaln_input, dim=1)  # [B, 2, adaln_dim]
        #         else:
        #             # 单分支情况
        #             adaln_input = adaln_input[0]  # [B, adaln_dim]
            
        #     # 应用 adaLN 调制
        #     # print("adaln_input shape: ", adaln_input.shape)
        #     adaln_output = self.adaLN_modulation(adaln_input)  # [B, 4*dim] 或 [B, 2, 4*dim]
            
        #     # 获取序列长度
        #     total_seq_len = x.shape[1]
            
        #     if adaln_output.ndim == 3:
        #         # 双分支情况：[B, 2, 4*dim] -> split to 4 parts of [B, 2, dim] each
        #         B, num_branches, _ = adaln_output.shape
        #         assert num_branches == 2, "双分支情况下 adaln_output 的分支数必须为2"
                
        #         # Split into 4 parts: [B, 2, dim] each
        #         scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_output.chunk(4, dim=-1)
                
        #         # 获取每个分支的序列长度
        #         seq_len_per_branch = total_seq_len // 2
                
        #         # broadcast to [B, 2*L, dim] each, following the reference code pattern
        #         # after replace img_mod's linear, [B, 2, 3*dim] each
        #         # if img_mod_attn.ndim == 3:
        #         #     assert img_mod_attn.shape[1] == 2 # 这里是2 是因为img_mod用两套参数区别对待了temb输入
        #         #     # broadcast to [B, 2L, 3*dim]
        #         #     img_mod_attn = torch.repeat_interleave(img_mod_attn, repeats=L, dim=1)
                
        #         # 对于双分支情况，扩展到整个序列长度
        #         scale_msa = torch.repeat_interleave(scale_msa, repeats=seq_len_per_branch, dim=1)  # [B, 2*L, dim]
        #         gate_msa = torch.repeat_interleave(gate_msa, repeats=seq_len_per_branch, dim=1)    # [B, 2*L, dim]
        #         scale_mlp = torch.repeat_interleave(scale_mlp, repeats=seq_len_per_branch, dim=1)  # [B, 2*L, dim]
        #         gate_mlp = torch.repeat_interleave(gate_mlp, repeats=seq_len_per_branch, dim=1)    # [B, 2*L, dim]
        #     else:
        #         # 单分支情况：[B, 4*dim] -> split to 4 parts of [B, dim] each
        #         scale_msa, gate_msa, scale_mlp, gate_mlp = adaln_output.chunk(4, dim=-1)
        #         # unsqueeze to match x shape [B, 1, dim] then will be broadcast
        #         scale_msa = scale_msa  # [B, dim]
        #         gate_msa = gate_msa    # [B, dim]
        #         scale_mlp = scale_mlp  # [B, dim]
        #         gate_mlp = gate_mlp    # [B, dim]

        #     gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        #     scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        #     # 注意力块
        #     # adaln_output shape:  torch.Size([1, 2, 15360])
        #     # scale_msa shape:  torch.Size([1, 2080, 3840])
        #     # self.attention_norm1(x) shape:  torch.Size([1, 2080, 3840])
        #     # x shape:  torch.Size([1, 2080, 3840])scale_msa shape:  torch.Size([1, 2048, 3840])
        #     # print("adaln_output shape: ", adaln_output.shape)
        #     # print("scale_msa shape: ", scale_msa.shape)
        #     # print("self.attention_norm1(x) shape: ", self.attention_norm1(x).shape)  
        #     # print("x shape: ", x.shape)
        #     if adaln_output.ndim == 3:
        #         # For dual branch, apply modulation to the entire sequence
        #         #[修改] 传入attn_mask参数
        #         attn_out = self.attention(
        #             self.attention_norm1(x) * scale_msa,
        #             freqs_cis=freqs_cis,
        #             attn_mask=attn_mask,
        #         )
        #         x = x + gate_msa * self.attention_norm2(attn_out)
        #     else:
        #         # For single branch, unsqueeze scale to match x shape [B, 1, dim]
        #         attn_out = self.attention(
        #             self.attention_norm1(x) * scale_msa.unsqueeze(1),
        #             freqs_cis=freqs_cis,
        #             attn_mask=attn_mask,
        #         )
        #         x = x + gate_msa * self.attention_norm2(attn_out)

        #     # FFN块
        #     if adaln_output.ndim == 3:
        #         # For dual branch
        #         x = x + gate_mlp * self.ffn_norm2(
        #             self.feed_forward(
        #                 self.ffn_norm1(x) * scale_mlp,
        #             )
        #         )
        #     else:
        #         # For single branch
        #         x = x + gate_mlp * self.ffn_norm2(
        #             self.feed_forward(
        #                 self.ffn_norm1(x) * scale_mlp.unsqueeze(1),
        #             )
        #         )
        # else:
        #     # 注意力块
        #     attn_out = self.attention(
        #         self.attention_norm1(x),
        #         freqs_cis=freqs_cis,
        #         attn_mask=attn_mask,
        #     )
        #     x = x + self.attention_norm2(attn_out)

        #     # FFN块
        #     x = x + self.ffn_norm2(
        #         self.feed_forward(
        #             self.ffn_norm1(x),
        #         )
        #     )

        # return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        # c shape: [B, adaln_dim] for single branch or [B, 2, adaln_dim] for dual branch
        # 处理 c 是张量列表的情况
        if isinstance(c, list):
            if len(c) == 2:
                # 双分支情况：沿维度1堆叠得到 [B, 2, adaln_dim]
                c = torch.stack(c, dim=1)  # [B, 2, adaln_dim]
            else:
                # 单分支情况
                c = c[0]  # [B, adaln_dim]
        
        if DEBUG:
            print("c shape: ", c.shape) # [1,256]
        if c.ndim == 3:  # 双分支情况
            if DEBUG:
                print("c shape: ", c.shape) # torch.Size([1, 2, 256])
            B_total, num_branches, adaln_dim = c.shape
            if DEBUG:
                print("x shape: ", x.shape) # torch.Size([1, 1056, 3840])
            total_seq_len = x.shape[1]
            seq_len_per_branch = total_seq_len // 2  # 每个分支的长度
            
            # # 将 x 分为噪声和条件两部分
            # x_noise = x[:, :seq_len_per_branch, :]  # 前半部分
            # x_cond = x[:, seq_len_per_branch:, :]   # 后半部分
            
            # # 分别处理噪声分支和条件分支
            # # 噪声分支
            # noise_c = c[:, 0, :]  # [B_total, adaln_dim]
            # noise_scale_shift = self.adaLN_modulation(noise_c)  # [B_total, hidden_size]
            # if DEBUG:
            #     print("noise_c shape: ", noise_c.shape) # torch.Size([1, 256])
            #     print("noise_scale_shift shape: ", noise_scale_shift.shape) # torch.Size([1, 3840])
            
            # # 对噪声分支应用调制
            # noise_scale = 1.0 + noise_scale_shift.unsqueeze(1)  # [B, 1, hidden_size]
            # x_noise = self.norm_final(x_noise) * noise_scale   # [B, seq_len_per_branch, hidden_size]
            
            # # 条件分支
            # cond_c = c[:, 1, :]   # [B_total, adaln_dim]
            # cond_scale_shift = self.adaLN_modulation(cond_c)   # [B_total, hidden_size]
            # if DEBUG:
            #     print("cond_c shape: ", cond_c.shape) # torch.Size([1, 256])
            #     print("cond_scale_shift shape: ", cond_scale_shift.shape) # torch.Size([1, 3840])
            
            # # 对条件分支应用调制
            # cond_scale = 1.0 + cond_scale_shift.unsqueeze(1)   # [B, 1, hidden_size]
            # x_cond = self.norm_final(x_cond) * cond_scale      # [B, seq_len_per_branch, hidden_size]
            
            scale_shift = self.adaLN_modulation(c)
            scale = 1.0 + scale_shift
            # print("scale shape: ", scale.shape)
            # print(" x shape: ", x.shape)
            # scale shape:  torch.Size([1, 2, 3840])
            # x shape:  torch.Size([1, 2080, 3840])
            scale = torch.repeat_interleave(scale, repeats=seq_len_per_branch, dim=1)
            # print("scale shape: ", scale.shape) torch.Size([1, 2080, 3840])
            x = self.norm_final(x) * scale
            
        else:  # 单分支情况 - 原有逻辑
            if DEBUG:
                print("c shape: ", c.shape) # [1,256]
            scale = 1.0 + self.adaLN_modulation(c)  # [B, hidden_size]
            if DEBUG:
                print("scale shape: ", scale.shape) # [1,3840]
                print("x shape: ", x.shape) # [1, 1056, 3840]
            x = self.norm_final(x) * scale.unsqueeze(1)  # [B, 1, hidden_size]
            if DEBUG:
                print("x shape: ", x.shape) # [1, 1056, 3840]
        
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = 256.0):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)  # complex64
                freqs_cis.append(freqs_cis_i)

            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i][index])
        return torch.cat(result, dim=-1)


class ZImageDiT(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["ZImageTransformerBlock"]

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        enable_2_temb=True, # TODO 检查参数传递
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.enable_2_temb = enable_2_temb
        if self.enable_2_temb:
            self.t_embedder_2 = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024) # 第二个时间嵌入层
            print(f"DiT的第二个时间步嵌入层设置为{self.enable_2_temb}")
            # [方案一核心]: 特征融合层 (Concat -> Linear)
            # 输入: t_emb (1024) + t_emb_2 (1024) = 2048
            # 输出: fused_emb (1024)
            # 注意: 这里用 mid_size (1024) 还是 out_size (256)? 
            # 查看 TimestepEmbedder 代码，输出是 out_size (t_embed_dim)，通常是 256。
            # 所以输入是 t_embed_dim * 2，输出是 t_embed_dim
            # t_embed_dim = min(dim, ADALN_EMBED_DIM)
            # self.time_fusion = nn.Linear(t_embed_dim * 2, t_embed_dim, bias=True)

            # [关键]: 零初始化 (Zero-Initialization)
            # 确保初始状态下 time_fusion 输出全 0，模型等价于原始 DiT
            # nn.init.zeros_(self.time_fusion.weight)
            # nn.init.zeros_(self.time_fusion.bias)
            
            # 注意：t_embedder_2 本身不需要 Zero-Init 了，因为有 time_fusion 把关
            # 但为了加速收敛，可以用 t_embedder 的权重初始化它 (可选)，在后面的load state dict中实现了
            # with torch.no_grad():
            #     self.t_embedder_2.load_state_dict(self.t_embedder.state_dict())
        else:
            print("没有加入第二个时间步嵌入层")
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t: List[torch.Tensor],
        cap_feats: List[torch.Tensor],
        condition_x: Optional[List[torch.Tensor]] = None, 
        patch_size=2,
        f_patch_size=1,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        target_timestep=None,  # NEW: target timestep
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        # print("bsz: ", bsz) # 由于现在输入有gt有lq,所以不能以pil的形式叠加batch,只能bs=1
        device = x[0].device
 
        # t = t * self.t_scale
        # t = self.t_embedder(t)
        # adaln_input = t

        # NEW: Implement dual timestep embedding ---------------------------------------------------------------------
        # TODO 检查是否正确
        adaln_input = []
        cond_t = t[1]
        noise_t = t[0]
        noise_tt = target_timestep[0]
        cond_tt = target_timestep[1]
        # ori_dtype = noise_t.dtype
        # print("original dtype: ", ori_dtype) # torch.float32
        ori_dtype = torch.bfloat16
        calc_dtype = torch.float32
        t_high = noise_t.to(dtype=calc_dtype)
        cond_t_high = cond_t.to(dtype=calc_dtype)
        with torch.autocast(device_type=device.type, enabled=False):
            # TODO：使用原始时间步还是绝对值
            if self.enable_2_temb:
                if False:
                    t_emb = self.t_embedder(t_high * self.t_scale)
                    cond_t_emb = self.t_embedder(cond_t_high * self.t_scale)
                    noise_tt_high = noise_tt.to(dtype=calc_dtype)
                    cond_tt_high = cond_tt.to(dtype=calc_dtype)

                    if DEBUG:
                        print("timestep fusion DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG")
                        print("noise_tt_high: ", noise_tt_high)
                    t_emb_2_input = (noise_tt_high * self.t_scale - t_high * self.t_scale)
                    t_emb_2_cond_input = (cond_tt_high * self.t_scale - cond_t_high * self.t_scale)
                    t_emb_2 = self.t_embedder_2(t_emb_2_input)
                    t_emb_2_cond = self.t_embedder_2(t_emb_2_cond_input)
                    
                    # print("t_emb_2: ", t_emb_2.shape)
                    combined_noise = torch.cat([t_emb, t_emb_2], dim=-1) # [B, 2*Dim]
                    combined_cond = torch.cat([cond_t_emb, t_emb_2_cond], dim=-1)
                    combined_noise = combined_noise
                    combined_cond = combined_cond
                    
                    # 融合并残差连接: Output = Base + Gate(Concat(Base, Trend))
                    # 初始时 time_fusion 输出 0，Output = Base，完美退化
                    fused_delta_noise = self.time_fusion(combined_noise)
                    adaln_input_noise = (t_emb + fused_delta_noise)
                    adaln_input.append(adaln_input_noise)
                    fused_delta_cond = self.time_fusion(combined_cond)
                    adaln_input_cond = (cond_t_emb + fused_delta_cond)
                    adaln_input.append(adaln_input_cond)
                else: # TODO 检查：这里对噪声分支和条件分支分别输入t和tt
                    adaln_input_noise = self.t_embedder(t_high * self.t_scale)
                    cond_tt_high = cond_tt.to(dtype=calc_dtype)
                    adaln_input_cond = self.t_embedder_2(cond_tt_high * self.t_scale)
                    adaln_input = [adaln_input_noise, adaln_input_cond]
                
                adaln_input = torch.stack(adaln_input, dim=1).to(device=device, dtype=ori_dtype)
            else:
                noise_tt_high = noise_tt.to(dtype=calc_dtype)
                cond_tt_high = cond_tt.to(dtype=calc_dtype)
                t_embedder_input = (t_high * self.t_scale + noise_tt_high * self.t_scale) / 2
                adaln_input_noise = self.t_embedder(t_embedder_input)
                t_embedder_input_cond = (cond_t_high * self.t_scale + cond_tt_high * self.t_scale) / 2
                adaln_input_cond = self.t_embedder(t_embedder_input_cond)
                adaln_input = torch.stack([adaln_input_noise, adaln_input_cond], dim=1).to(device=device, dtype=ori_dtype)
        # # 确保 adaln_input 与模型的 dtype 一致
        # adaln_input = torch.stack(adaln_input, dim=1).to(device=device, dtype=ori_dtype)

        # END NEW ---------------------------------------------------------------------

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)
        # 记录 噪声 部分的长度
        # 注意: x 是 list of tensors, 我们先取第一个计算长度 (假设 batch padding 一致)
        L_noise_per_sample = len(x[0])
        print(f"L_noise_per_sample: {L_noise_per_sample}")

        # ===================== NEW: concat condition on seq_len and duplicate pos ids =====================
        if condition_x is not None:
            # patchify condition images too (cap parts are redundant; we only use condition image tokens)
            (
                x_ctrl,
                _cap_feats2,
                x_size2,
                x_pos_ids2,
                _cap_pos_ids2,
                x_inner_pad_mask2,
                _cap_inner_pad_mask2,
            ) = self.patchify_and_embed(condition_x, cap_feats, patch_size, f_patch_size)
            # print("x_ctrl len shape: ", len(x_ctrl), x_ctrl[0].shape)
            L_cond_per_sample = len(x_ctrl[0])
            print("L_cond_per_sample: ", L_cond_per_sample)
            pH = pW = patch_size
            pF = f_patch_size

            new_x = []
            new_pos = []
            new_pad_mask = []

            for i in range(len(x)):
                F, H, W = x_size[i]
                F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW
                ori_len = F_tokens * H_tokens * W_tokens
                print("ori_len: ", ori_len)
                # take ONLY real tokens (remove per-branch padding), then concat
                x_main_ori = x[i][:ori_len]
                x_ctrl_ori = x_ctrl[i][:ori_len]

                # Qwen-like: pos ids for control are duplicated from main
                pos_main_ori = x_pos_ids[i][:ori_len]
                x_cat = torch.cat([x_main_ori, x_ctrl_ori], dim=0)            # (2*ori_len, dim_in)
                pos_cat = torch.cat([pos_main_ori, pos_main_ori], dim=0)      # duplicate positions

                # re-pad to SEQ_MULTI_OF (pad only at the end, no pads in the middle)
                pad_len = (-len(x_cat)) % SEQ_MULTI_OF
                if pad_len > 0:
                    x_cat = torch.cat([x_cat, x_cat[-1:].repeat(pad_len, 1)], dim=0)
                    pad_pos = (
                        self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=x_cat.device)
                        .flatten(0, 2)
                        .repeat(pad_len, 1)
                    )
                    pos_cat = torch.cat([pos_cat, pad_pos], dim=0)

                pad_mask = torch.cat(
                    [
                        torch.zeros((len(pos_cat) - pad_len,), dtype=torch.bool, device=x_cat.device),
                        torch.ones((pad_len,), dtype=torch.bool, device=x_cat.device),
                    ],
                    dim=0,
                )

                new_x.append(x_cat)
                new_pos.append(pos_cat)
                new_pad_mask.append(pad_mask)

            x = new_x
            x_pos_ids = new_pos
            x_inner_pad_mask = new_pad_mask
            # adaln_input = torch.cat([adaln_input, adaln_input], dim=0)
        # ===================== END NEW =====================
        # print("x len shape: ", len(x), x[0].shape)
        # x len shape:  1 torch.Size([1024, 64])
        # x_ctrl len shape:  1 torch.Size([1024, 64])
        # x len shape:  1 torch.Size([2048, 64])

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        print("x_item_seqlens: ", x_item_seqlens)
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        # print("x shape: ", x.shape) # x shape:  torch.Size([2048, 64])
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x) # 对于这种层，输入要求和duallora不匹配，只能正常微调
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token.to(dtype=x.dtype, device=x.device)
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)

        # 原来的attn_mask逻辑
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1
        # # ================= NEW: 提前构建 Image-Only Attention Mask =================
        # # 因为 noise_refiner 只有图像，我们需要一个只包含图像部分的 Mask
        # # 假设 x_batch 已经是 [B, Image_Seq_Len, Dim]
        
        # # 这里的 Image_Seq_Len = Noise_Len + Cond_Len
        # img_max_len = x.shape[1] 
        
        # img_attn_mask = torch.zeros(
        #     (bsz, 1, img_max_len, img_max_len), 
        #     dtype=torch.bool, 
        #     device=device
        # )
        
        # for i in range(bsz):
        #     curr_len = x_item_seqlens[i] # 当前样本图像总长
        #     img_attn_mask[i, 0, :curr_len, :curr_len] = True # 基础可见性
            
        #     # 应用 "Noise 看 Cond, Cond 不看 Noise" 逻辑
        #     if condition_x is not None:
        #         noise_len = curr_len // 2 # 假设是对半分
        #         # Cond (Row) 不能看 Noise (Col)
        #         # Row: [noise_len : curr_len]
        #         # Col: [0 : noise_len]
        #         img_attn_mask[i, 0, noise_len:curr_len, 0:noise_len] = False
        # x_attn_mask = img_attn_mask
        # # ================= END NEW =================

        # print("[INFO] Using Noise Refiner") # 序列长度2048
        # [构造 Slice Info] 用于 Noise Refiner
        # Refiner 只看到 Noise + Cond，没有 Text
        # slice_info = (L_noise, L_cond, 0)
        # 注意: 这里假设 Batch 内长度是对齐的 (即 L_noise_per_sample 是一样的)
        slice_info_refiner = (L_noise_per_sample, L_cond_per_sample, 0)
        for layer in self.noise_refiner:
            x = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=x,
                attn_mask=x_attn_mask,
                freqs_cis=x_freqs_cis,
                adaln_input=adaln_input,
                slice_info=slice_info_refiner, # [传参]
            )

        # ================= Process Caption =================
        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)
        L_text_per_sample = cap_item_seqlens[0] # 假设对齐

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token.to(dtype=x.dtype, device=x.device)
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        # print("[INFO] Using Context Refiner")
        for layer in self.context_refiner:
            cap_feats = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=cap_feats,
                attn_mask=cap_attn_mask,
                freqs_cis=cap_freqs_cis,
                slice_info=None # [传参]
            )

        # ================= Unified Construction =================
        # 顺序: [Noise, Cond, Text]
        # x_batch 已经是 [Noise, Cond]
        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)

        # 原来的attn_mask
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
        # # ================= NEW: 构建特殊的 Attention Mask =================
        # # 我们需要构建一个 [B, 1, L, L] 的掩码
        # # True 表示可见 (Attend), False 表示屏蔽 (Mask)
        
        # unified_attn_mask = torch.zeros(
        #     (bsz, 1, unified_max_item_seqlen, unified_max_item_seqlen), 
        #     dtype=torch.bool, 
        #     device=device
        # ) # 初始化为全False (全屏蔽)

        # for i in range(bsz):
        #     total_len = unified_item_seqlens[i] # 当前样本总有效长度
        #     x_len = x_item_seqlens[i]           # 图像部分长度 (Noise + Cond)
            
        #     # 1. 基础掩码：有效长度内全部可见 (对角矩阵块)
        #     # 这是一个标准的 Causal Mask 或者 Full Mask，这里我们先设为 Full
        #     unified_attn_mask[i, 0, :total_len, :total_len] = True
            
        #     # 2. 应用特殊逻辑：Condition 分支不能看 Noise 分支
        #     if condition_x is not None:
        #         # 图像部分包含了 [Noise, Condition]
        #         # x_len 应该能被2整除
        #         noise_len = x_len // 2
        #         cond_len = x_len // 2
                
        #         # 定义各个区域在序列中的范围
        #         # Sequence Layout: [Noise (0~N), Condition (N~2N), Caption (2N~End)]
                
        #         # Condition Query 的范围: [noise_len : x_len]
        #         # Noise Key 的范围:       [0 : noise_len]
                
        #         # [核心逻辑]: Condition 不能看 Noise
        #         # Mask[Row, Col] = False -> Row 不能看 Col
        #         unified_attn_mask[i, 0, noise_len:x_len, 0:noise_len] = False
                
        #         # [可选逻辑]: 如果你想让 Caption 也能看所有，上面已经覆盖了
        #         # [可选逻辑]: 如果你想让 Caption 只能看 Condition 不能看 Noise? 
        #         # 通常 Caption 看全图，保持默认即可。
            
        #     # 注意：padding 部分保持为 False (初始化时已为 0)
            
        # # ================= END NEW =================

        # print("[INFO] Using Layers") # 序列长度2080
        # [构造 Slice Info] 用于 Main Layers
        # slice_info = (L_noise, L_cond, L_text)
        slice_info_layers = (L_noise_per_sample, L_cond_per_sample, L_text_per_sample)
        for layer in self.layers:
            unified = gradient_checkpoint_forward(
                layer,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=unified,
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_cis,
                adaln_input=adaln_input,
                slice_info=slice_info_layers, # [传参]
            )

        # ================= Final Layer (Output Optimization) =================
        # [优化]: 只取 Noise 部分进 Final Layer
        # unified 结构: [Noise, Cond, Text]
        # Noise 区间: [0 : L_noise]
        
        # 1. 切片：只取 Noise 部分
        # 这一步极其重要，它规避了上面提到的 AdaLN 错位风险
        noise_x = unified[:, :L_noise_per_sample, :] 

        # 2. 准备 AdaLN 输入
        # 我们只需要 Noise 分支的时间步信息 (Index 0)
        # adaln_input 原本是 [B, 2, Dim]
        adaln_input_noise = adaln_input[:, 0, :] # 取出 Noise 的 emb

        # 3. 输入 Final Layer
        # 此时 final_x 长度纯净，adaln_input 也是单分支的
        # FinalLayer 内部会走 "else: 单分支" 的逻辑
        # scale = 1 + linear(adaln_input_noise) -> [B, Dim] -> broadcast to [B, 1, Dim]
        noise_out = self.all_final_layer[f"{patch_size}-{f_patch_size}"](noise_x, adaln_input_noise)
        noise_out = list(noise_out.unbind(dim=0))

        # 注意: x_size 存储的是图像原始尺寸，unpatchify 需要这个
        x = self.unpatchify(noise_out, x_size, patch_size, f_patch_size)


        # 统一全要过final layer，原始的逻辑 ——————————————————————————————
        # unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        # unified = list(unified.unbind(dim=0))
        # x = self.unpatchify(unified, x_size, patch_size, f_patch_size)
        # 统一全要过final layer，原始的逻辑 ——————————————————————————————
        
        return x, {}
    
    # TODO 检查一下这里用t_emb_1初始化t_emb_2是否正确
    def load_state_dict(self, state_dict, strict=True, **kwargs):
        if self.enable_2_temb:
            # 检查state_dict中是否包含新增模块的参数
            t_embedder_2_keys = [key for key in state_dict.keys() if "t_embedder_2" in key]
            # time_fusion_keys = [key for key in state_dict.keys() if "time_fusion" in key]
            
            # 如果state_dict包含这些模块的参数，需要确保当前模型也具有相应的结构
            if len(t_embedder_2_keys) > 0:
                # 检查是否存在模块，如果不存在则创建
                if not hasattr(self, 't_embedder_2'):
                    t_embed_dim = min(self.dim, ADALN_EMBED_DIM)
                    self.t_embedder_2 = TimestepEmbedder(t_embed_dim, mid_size=1024)
                # if not hasattr(self, 'time_fusion'):
                #     t_embed_dim = min(self.dim, ADALN_EMBED_DIM)
                #     self.time_fusion = nn.Linear(t_embed_dim * 2, t_embed_dim, bias=True)
                #     nn.init.zeros_(self.time_fusion.weight)
                #     nn.init.zeros_(self.time_fusion.bias)
            
            # 执行正常的加载过程
            missing, unexpected = super().load_state_dict(state_dict, strict=False, **kwargs)

            print(f"缺失的键: {missing}")
            print(f"未预期的键: {unexpected}")
            
            # 检查是否还有未加载的新模块参数
            missing_t_embedder_2 = [key for key in missing if "t_embedder_2" in key]
            # missing_time_fusion = [key for key in missing if "time_fusion" in key]
            
            if len(missing_t_embedder_2) > 0:
                print("检测到缺失的新模块参数，可能是结构不匹配造成的。")
                # 如果仍有缺失的参数，尝试更灵活的加载方式
                # 只处理基础缺失，对于结构变化的问题，依赖训练时的LoRA设置
                pass
            else:
                print("加载没有问题")
                return missing, unexpected
            
            # 2. 处理新加的层 (仅在启用双时间步时)
            # [关键步骤] 实体化：把 meta tensor 变成实际占内存的 tensor
            # 必须先 to_empty(device='cpu')，否则无法初始化数值
            self.t_embedder_2.to_empty(device="cuda")
            # self.time_fusion.to_empty(device='cpu')

            # 3. 初始化权重
            # A. t_embedder_2: 直接复制 t_embedder 的参数作为起点
            self.t_embedder_2.load_state_dict(self.t_embedder.state_dict(), strict=True, **kwargs)
            
            # B. time_fusion: 零初始化 (保证初始状态下无影响)
            # torch.nn.init.zeros_(self.time_fusion.weight)
            # torch.nn.init.zeros_(self.time_fusion.bias)

            # 使用极小的高斯噪声，既不破坏预训练性能，又能打破对称性
            # torch.nn.init.normal_(self.time_fusion.weight, mean=0.0, std=1e-4) 
            # if self.time_fusion.bias is not None:
            #     torch.nn.init.zeros_(self.time_fusion.bias) # bias 可以保持为 0，只要 weight 不为 0 即可

            print("Successfully initialized t_embedder_2 and time_fusion.")

        else:
            missing, unexpected = super().load_state_dict(state_dict, strict=True, **kwargs)
            return missing, unexpected
