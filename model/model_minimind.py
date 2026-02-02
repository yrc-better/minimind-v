# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

"""
MiniMind 核心模型实现（注释版）

包含内容
- MiniMindConfig：模型与训练超参配置（含 RoPE、KV Cache、Flash Attention、MoE 等）
- RMSNorm / RoPE：归一化与旋转位置编码
- Attention：支持 GQA/MQA、KV cache、PyTorch 2.x SDPA(Flash) 的自注意力
- FeedForward / MOEFeedForward：标准 FFN 与 Mixture-of-Experts 前馈层（含路由与辅助损失）
- MiniMindModel：解码器主体（多层 Block 堆叠）
- MiniMindForCausalLM：HF 风格的 CausalLM 封装（logits、loss、past_key_values）

本文件重点注释：
- 每个类/函数的职责、输入输出形状
- 关键 PyTorch/Transformer 机制（view/transpose、register_buffer、KV cache、cross_entropy、SDPA、scatter_add）
- 复杂实现（RoPE 预计算、GQA 的 repeat_kv、MoE 路由与推理加速）
"""

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    '''
    配置类：集中管理 MiniMind 的所有超参数。
    
    你会在训练/推理脚本中构造该配置，然后交给 MiniMindModel / MiniMindForCausalLM。
    关键字段：
    - hidden_size / num_attention_heads / num_hidden_layers：Transformer 主体规模
    - rope_theta / rope_scaling：RoPE 参数（rope_scaling 可用于长上下文扩展）
    - flash_attn：是否优先使用 PyTorch 2.x 的 scaled_dot_product_attention
    - use_moe + (n_routed_experts, num_experts_per_tok, aux_loss_alpha, ...)：MoE 相关配置
    '''
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.01,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    '''
    RMSNorm（Root Mean Square LayerNorm 的变体）
    
    与 LayerNorm 的区别：不减去均值，只按 RMS 做缩放。
    公式：y = w * x / sqrt(mean(x^2) + eps)
    其中 w 为可训练缩放参数（nn.Parameter）。
    '''
    def __init__(self, dim: int, eps: float = 1e-5):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        '''
        RMSNorm 的核心归一化：按最后一维计算 RMS 并缩放。
        
        torch.rsqrt：计算 1/sqrt(x)，比先 sqrt 再取倒数更数值稳定/更高效。
        '''
        # torch.rsqrt(x)：计算 1/sqrt(x)；这里用于 RMSNorm 的归一化缩放，数值更稳。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        '''
        前向传播：定义 `RMSNorm` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    '''
    预计算 RoPE 所需的 cos/sin 表。
    
    参数：
    - dim：每个 head 的维度（通常 hidden_size / num_attention_heads）
    - end：最大位置长度（max_position_embeddings）
    - rope_base：RoPE 的基数（theta）
    - rope_scaling：可选的缩放策略（用于长上下文扩展，如 YaRN）
    
    返回：
    - freqs_cos, freqs_sin：形状约为 [end, dim]（具体取决于实现）
    '''
    # 生成 RoPE 的逆频率 inv_freq：不同维度对应不同旋转频率（偶数维/奇数维成对）。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # 可选 RoPE 缩放：用于长上下文扩展（例如 YaRN/NTK 相关思路），调整频率分布。
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    '''
    对 Query/Key 应用 RoPE 旋转位置编码。
    
    输入：
    - q, k：形状通常为 [B, T, n_heads, head_dim] 或相近布局
    - cos, sin：来自 precompute_freqs_cis 的表（按位置切片）
    - unsqueeze_dim：为了对齐维度做广播
    
    返回：
    - q_embed, k_embed：应用 RoPE 后的 q/k
    '''
    def rotate_half(x):
        '''
        函数 `rotate_half`：具体逻辑见函数体注释。
        '''
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # RoPE 应用：q*cos + rotate(q)*sin（k 同理）；unsqueeze 用于对齐维度做广播。
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    # RoPE 应用：q*cos + rotate(q)*sin（k 同理）；unsqueeze 用于对齐维度做广播。
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # 通过 expand+reshape 复制 KV 头：避免真实拷贝数据（更省显存），对齐到 Q 头数。
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    '''
    多头自注意力层（Decoder Self-Attention）
    
    特性：
    - 支持 GQA/MQA：Q 头数可能大于 K/V 头数，通过 repeat_kv 复制 K/V 头对齐
    - 支持 RoPE：对 Q/K 应用旋转位置编码
    - 支持 KV cache：推理阶段可拼接历史 K/V，显著加速自回归生成
    - 支持 PyTorch 2.x SDPA：使用 F.scaled_dot_product_attention 走 Flash/高效路径（可选）
    '''
    def __init__(self, args: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        '''
        前向传播：定义 `Attention` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        bsz, seq_len, _ = x.shape
        # 线性投影得到 Q/K/V：把 hidden_states 映射到多头空间（后续会 reshape 成 [B,T,heads,head_dim]）。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # view：仅重排张量形状不拷贝数据；这里把最后一维拆成 (heads, head_dim)。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # view：仅重排张量形状不拷贝数据；这里把最后一维拆成 (heads, head_dim)。
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # view：仅重排张量形状不拷贝数据；这里把最后一维拆成 (heads, head_dim)。
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        # 对 Q/K 施加 RoPE：把位置信息编码进注意力的相位旋转里。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        # KV cache：把历史 K/V 与当前步的 K/V 在序列维拼接，用于自回归推理加速。
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            # transpose：交换维度以适配注意力实现（通常需要 [B, heads, T, head_dim]）。
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            # transpose：交换维度以适配注意力实现（通常需要 [B, heads, T, head_dim]）。
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # PyTorch 2.x SDPA：可能走 Flash Attention/更高效 kernel；is_causal=True 自动加因果 mask。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 慢速注意力路径：显式计算 QK^T/softmax，再与 V 相乘（当 SDPA 不满足条件时）。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果 mask：用上三角填充 -inf，禁止注意力看到未来 token（自回归）。
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            if attention_mask is not None:
                # attention_mask：把 [B,T] 扩展到可广播形状，加到 scores 上屏蔽 padding。
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax：把注意力分数归一化为概率分布（在 float 上算更稳定，再转回原 dtype）。
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    '''
    前馈网络（FFN），实现类似 LLaMA 的 SwiGLU 结构：
    act(gate_proj(x)) * up_proj(x) -> down_proj -> dropout
    
    其中 act 通常是 SiLU（也可能根据 config.hidden_act 选择其他激活）。
    '''
    def __init__(self, config: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        '''
        前向传播：定义 `FeedForward` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        # SwiGLU 风格：act(gate_proj(x)) * up_proj(x)，再 down_proj 回 hidden_size。
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    '''
    MoE 路由器（Gate）
    
    输入 token 表示 -> 计算每个 token 分配到各个专家(expert)的分数 -> 取 top-k 专家。
    可选：
    - norm_topk_prob：是否把 top-k 权重重新归一化
    - aux_loss_alpha + seq_aux：负载均衡辅助损失（防止路由坍塌，提升专家利用率）
    '''
    def __init__(self, config: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        '''
        函数 `reset_parameters`：具体逻辑见函数体注释。
        '''
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        '''
        前向传播：定义 `MoEGate` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        # Gate 打分：对每个 token 计算路由 logits（到每个 expert 的分数）。
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            # softmax 得到路由概率：每个 token 在所有专家上的分配权重。
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # topk：为每个 token 选择概率最大的 k 个专家（稀疏路由）。
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                # aux_loss：负载均衡正则项，鼓励不同专家都被使用，避免路由坍塌。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                # aux_loss：负载均衡正则项，鼓励不同专家都被使用，避免路由坍塌。
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    '''
    MoE 前馈层：由多个专家(FeedForward) + Gate 组成。
    
    训练阶段：
    - 为每个 token 选择 top-k 专家
    - 把 token 复制 k 份分别送入对应专家
    - 按 gate 权重加权求和得到输出
    推理阶段：
    - 使用 moe_infer：按专家分组、排序、scatter_add_ 回写，减少 Python 循环与切片开销
    '''
    def __init__(self, config: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        '''
        前向传播：定义 `MOEFeedForward` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        # 先由 Gate 选择 top-k 专家及权重；aux_loss 用于训练时的负载均衡。
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # repeat_interleave：把每个 token 复制 k 份，分别送入对应专家（训练阶段便于实现）。
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        '''
        MoE 推理加速路径（eval/inference 常用）
        
        思路：
        - 把 token 按选中的 expert id 排序/分组
        - 每个 expert 批量处理属于自己的 token（避免逐 token 循环）
        - 用 scatter_add_ 把加权后的输出累加回原 token 位置
        '''
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # scatter_add_：把专家输出按 token 索引累加回去（同一 token 的多个专家贡献会相加）。
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    '''
    Transformer Block（Decoder Block）
    
    结构：
    1) RMSNorm -> Self-Attention -> 残差
    2) RMSNorm -> FFN 或 MoE FFN -> 残差
    '''
    def __init__(self, layer_id: int, config: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        '''
        前向传播：定义 `MiniMindBlock` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    '''
    MiniMind 解码器主体：Embedding + N 层 Block + Norm
    
    - 预先用 precompute_freqs_cis 计算 RoPE 的 cos/sin 并注册为 buffer（register_buffer），避免每次 forward 重算。
    - forward 支持 past_key_values / use_cache：用于自回归生成的 KV cache。
    '''
    def __init__(self, config: MiniMindConfig):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        # register_buffer：把张量注册为 buffer（随模型保存/转移设备，但不参与训练更新）。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        # register_buffer：把张量注册为 buffer（随模型保存/转移设备，但不参与训练更新）。
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        '''
        前向传播：定义 `MiniMindModel` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # start_pos：已缓存的历史长度（KV cache 的序列长度），用于对齐 RoPE 位置切片。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 按 start_pos 切片 RoPE cos/sin：确保当前 token 的绝对位置正确（与 cache 对齐）。
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        # 汇总 MoE 的 aux_loss：仅对使用 MOEFeedForward 的层求和（否则为 0）。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    '''
    HF 风格的 CausalLM 封装
    
    - 内部包含 MiniMindModel + lm_head（输出词表 logits）
    - 绑定权重：embed_tokens.weight 与 lm_head.weight 共享（weight tying，减少参数并通常更稳）
    - 若提供 labels：计算 next-token 交叉熵（shift logits/labels 一位）
    - 输出 CausalLMOutputWithPast，并额外挂载 aux_loss（MoE 辅助损失）
    '''
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        '''
        构造函数：初始化模块的参数与子模块（nn.Linear/nn.Embedding 等）。
        '''
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重绑定（weight tying）：让输入 embedding 与输出 lm_head 共享权重，减少参数并常见于 LLaMA/GPT。
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        '''
        前向传播：定义 `MiniMindForCausalLM` 的核心计算逻辑（支持训练与推理/KV cache 等）。
        '''
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        # logits_to_keep：可只保留最后 N 个位置的 logits（推理省显存）；训练一般保留整段。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # next-token 训练：logits 去掉最后一位，labels 去掉第一位（对齐为“预测下一个 token”）。
            # contiguous：确保张量在内存中连续，便于 view/reshape 与高效 kernel。
            shift_logits = logits[..., :-1, :].contiguous()
            # contiguous：确保张量在内存中连续，便于 view/reshape 与高效 kernel。
            shift_labels = labels[..., 1:].contiguous()
            # next-token 训练：logits 去掉最后一位，labels 去掉第一位（对齐为“预测下一个 token”）。
            # cross_entropy：对词表做分类的损失；ignore_index=-100 表示这些位置不计入 loss（常用于 mask prompt/pad）。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
