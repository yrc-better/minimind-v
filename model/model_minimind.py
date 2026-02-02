"""
这个文件是 MiniMind 的核心实现（配置 + Transformer 解码器 + MoE 组件 + CausalLM 包装）。

你可以直接用它替换/对照原始实现进行阅读与调试：
- 配置：MiniMindConfig
- 组件：RMSNorm / RoPE / Attention / FFN / MoE
- 主体：MiniMindModel（解码器堆叠）
- 封装：MiniMindForCausalLM（Hugging Face 风格输出 + loss）

本注释版遵循：对“原脚本每一行可执行代码”在其上一行添加中文解释注释。
"""

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# [原脚本 L5] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from transformers import PretrainedConfig


# [原脚本 L8] 定义类 `MiniMindConfig`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MiniMindConfig(PretrainedConfig):
    # [原脚本 L9] 赋值：计算右侧表达式并保存到变量 `model_type`，供后续使用。
    model_type = "minimind"

    # [原脚本 L11] 函数/方法签名续行：继续列出参数或结束括号。
    def __init__(
            # [原脚本 L12] 函数/方法签名续行：继续列出参数或结束括号。
            self,
            # [原脚本 L13] 函数/方法签名参数：定义参数 `dropout` 的类型与默认值（仅声明，不会在此处执行计算）。
            dropout: float = 0.0,
            # [原脚本 L14] 函数/方法签名参数：定义参数 `bos_token_id` 的类型与默认值（仅声明，不会在此处执行计算）。
            bos_token_id: int = 1,
            # [原脚本 L15] 函数/方法签名参数：定义参数 `eos_token_id` 的类型与默认值（仅声明，不会在此处执行计算）。
            eos_token_id: int = 2,
            # [原脚本 L16] 函数/方法签名参数：定义参数 `hidden_act` 的类型与默认值（仅声明，不会在此处执行计算）。
            hidden_act: str = 'silu',
            # [原脚本 L17] 函数/方法签名参数：定义参数 `hidden_size` 的类型与默认值（仅声明，不会在此处执行计算）。
            hidden_size: int = 512,
            # [原脚本 L18] 函数/方法签名参数：定义参数 `intermediate_size` 的类型与默认值（仅声明，不会在此处执行计算）。
            intermediate_size: int = None,
            # [原脚本 L19] 函数/方法签名参数：定义参数 `max_position_embeddings` 的类型与默认值（仅声明，不会在此处执行计算）。
            max_position_embeddings: int = 32768,
            # [原脚本 L20] 函数/方法签名参数：定义参数 `num_attention_heads` 的类型与默认值（仅声明，不会在此处执行计算）。
            num_attention_heads: int = 8,
            # [原脚本 L21] 函数/方法签名参数：定义参数 `num_hidden_layers` 的类型与默认值（仅声明，不会在此处执行计算）。
            num_hidden_layers: int = 8,
            # [原脚本 L22] 函数/方法签名参数：定义参数 `num_key_value_heads` 的类型与默认值（仅声明，不会在此处执行计算）。
            num_key_value_heads: int = 2,
            # [原脚本 L23] 函数/方法签名参数：定义参数 `vocab_size` 的类型与默认值（仅声明，不会在此处执行计算）。
            vocab_size: int = 6400,
            # [原脚本 L24] 函数/方法签名参数：定义参数 `rms_norm_eps` 的类型与默认值（仅声明，不会在此处执行计算）。
            rms_norm_eps: float = 1e-05,
            # [原脚本 L25] 函数/方法签名参数：定义参数 `rope_theta` 的类型与默认值（仅声明，不会在此处执行计算）。
            rope_theta: int = 1000000.0,
            # [原脚本 L26] 函数/方法签名参数：定义参数 `inference_rope_scaling` 的类型与默认值（仅声明，不会在此处执行计算）。
            inference_rope_scaling: bool = False,
            # [原脚本 L27] 函数/方法签名参数：定义参数 `flash_attn` 的类型与默认值（仅声明，不会在此处执行计算）。
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            # [原脚本 L32] 函数/方法签名参数：定义参数 `use_moe` 的类型与默认值（仅声明，不会在此处执行计算）。
            use_moe: bool = False,
            # [原脚本 L33] 函数/方法签名参数：定义参数 `num_experts_per_tok` 的类型与默认值（仅声明，不会在此处执行计算）。
            num_experts_per_tok: int = 2,
            # [原脚本 L34] 函数/方法签名参数：定义参数 `n_routed_experts` 的类型与默认值（仅声明，不会在此处执行计算）。
            n_routed_experts: int = 4,
            # [原脚本 L35] 函数/方法签名参数：定义参数 `n_shared_experts` 的类型与默认值（仅声明，不会在此处执行计算）。
            n_shared_experts: int = 1,
            # [原脚本 L36] 函数/方法签名参数：定义参数 `scoring_func` 的类型与默认值（仅声明，不会在此处执行计算）。
            scoring_func: str = 'softmax',
            # [原脚本 L37] 函数/方法签名参数：定义参数 `aux_loss_alpha` 的类型与默认值（仅声明，不会在此处执行计算）。
            aux_loss_alpha: float = 0.01,
            # [原脚本 L38] 函数/方法签名参数：定义参数 `seq_aux` 的类型与默认值（仅声明，不会在此处执行计算）。
            seq_aux: bool = True,
            # [原脚本 L39] 函数/方法签名参数：定义参数 `norm_topk_prob` 的类型与默认值（仅声明，不会在此处执行计算）。
            norm_topk_prob: bool = True,
            # [原脚本 L40] 函数/方法签名续行：继续列出参数或结束括号。
            **kwargs
    # [原脚本 L41] 函数/方法签名结束：后面开始进入函数体。
    ):
        # [原脚本 L42] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__(**kwargs)
        # [原脚本 L43] 设置对象属性：将计算/配置结果保存到 `self.dropout`，以便后续 forward/推理时使用。
        self.dropout = dropout
        # [原脚本 L44] 设置对象属性：将计算/配置结果保存到 `self.bos_token_id`，以便后续 forward/推理时使用。
        self.bos_token_id = bos_token_id
        # [原脚本 L45] 设置对象属性：将计算/配置结果保存到 `self.eos_token_id`，以便后续 forward/推理时使用。
        self.eos_token_id = eos_token_id
        # [原脚本 L46] 设置对象属性：将计算/配置结果保存到 `self.hidden_act`，以便后续 forward/推理时使用。
        self.hidden_act = hidden_act
        # [原脚本 L47] 设置对象属性：将计算/配置结果保存到 `self.hidden_size`，以便后续 forward/推理时使用。
        self.hidden_size = hidden_size
        # [原脚本 L48] 设置对象属性：将计算/配置结果保存到 `self.intermediate_size`，以便后续 forward/推理时使用。
        self.intermediate_size = intermediate_size
        # [原脚本 L49] 设置对象属性：将计算/配置结果保存到 `self.max_position_embeddings`，以便后续 forward/推理时使用。
        self.max_position_embeddings = max_position_embeddings
        # [原脚本 L50] 设置对象属性：将计算/配置结果保存到 `self.num_attention_heads`，以便后续 forward/推理时使用。
        self.num_attention_heads = num_attention_heads
        # [原脚本 L51] 设置对象属性：将计算/配置结果保存到 `self.num_hidden_layers`，以便后续 forward/推理时使用。
        self.num_hidden_layers = num_hidden_layers
        # [原脚本 L52] 设置对象属性：将计算/配置结果保存到 `self.num_key_value_heads`，以便后续 forward/推理时使用。
        self.num_key_value_heads = num_key_value_heads
        # [原脚本 L53] 设置对象属性：将计算/配置结果保存到 `self.vocab_size`，以便后续 forward/推理时使用。
        self.vocab_size = vocab_size
        # [原脚本 L54] 设置对象属性：将计算/配置结果保存到 `self.rms_norm_eps`，以便后续 forward/推理时使用。
        self.rms_norm_eps = rms_norm_eps
        # [原脚本 L55] 设置对象属性：将计算/配置结果保存到 `self.rope_theta`，以便后续 forward/推理时使用。
        self.rope_theta = rope_theta
        # [原脚本 L56] 设置对象属性：将计算/配置结果保存到 `self.inference_rope_scaling`，以便后续 forward/推理时使用。
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        # [原脚本 L58] 设置对象属性：将计算/配置结果保存到 `self.rope_scaling`，以便后续 forward/推理时使用。
        self.rope_scaling = {
            # [原脚本 L59] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "beta_fast": 32,
            # [原脚本 L60] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "beta_slow": 1,
            # [原脚本 L61] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "factor": 16,
            # [原脚本 L62] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "original_max_position_embeddings": 2048,
            # [原脚本 L63] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "attention_factor": 1.0,
            # [原脚本 L64] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            "type": "yarn"
        # [原脚本 L65] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        } if self.inference_rope_scaling else None
        # [原脚本 L66] 设置对象属性：将计算/配置结果保存到 `self.flash_attn`，以便后续 forward/推理时使用。
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        # [原脚本 L71] 设置对象属性：将计算/配置结果保存到 `self.use_moe`，以便后续 forward/推理时使用。
        self.use_moe = use_moe
        # [原脚本 L72] 设置对象属性：将计算/配置结果保存到 `self.num_experts_per_tok`，以便后续 forward/推理时使用。
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        # [原脚本 L73] 设置对象属性：将计算/配置结果保存到 `self.n_routed_experts`，以便后续 forward/推理时使用。
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        # [原脚本 L74] 设置对象属性：将计算/配置结果保存到 `self.n_shared_experts`，以便后续 forward/推理时使用。
        self.n_shared_experts = n_shared_experts  # 共享专家
        # [原脚本 L75] 设置对象属性：将计算/配置结果保存到 `self.scoring_func`，以便后续 forward/推理时使用。
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        # [原脚本 L76] 设置对象属性：将计算/配置结果保存到 `self.aux_loss_alpha`，以便后续 forward/推理时使用。
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        # [原脚本 L77] 设置对象属性：将计算/配置结果保存到 `self.seq_aux`，以便后续 forward/推理时使用。
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        # [原脚本 L78] 设置对象属性：将计算/配置结果保存到 `self.norm_topk_prob`，以便后续 forward/推理时使用。
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

# [原脚本 L85] 导入外部库/模块，后续将使用其提供的函数与类型。
import math
# [原脚本 L86] 导入外部库/模块，后续将使用其提供的函数与类型。
import torch
# [原脚本 L87] 导入外部库/模块，后续将使用其提供的函数与类型。
import torch.nn.init as init
# [原脚本 L88] 导入外部库/模块，后续将使用其提供的函数与类型。
import torch.nn.functional as F
# [原脚本 L89] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from torch import nn
# [原脚本 L90] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from transformers.activations import ACT2FN
# [原脚本 L91] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from typing import Optional, Tuple, List, Union
# [原脚本 L92] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
# [原脚本 L93] 从外部库导入模块/类/函数，供本文件实现模型与训练/推理逻辑使用。
from transformers.modeling_outputs import CausalLMOutputWithPast


# [原脚本 L96] 定义类 `RMSNorm`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class RMSNorm(torch.nn.Module):
    # [原脚本 L97] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, dim: int, eps: float = 1e-5):
        # [原脚本 L98] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L99] 设置对象属性：将计算/配置结果保存到 `self.eps`，以便后续 forward/推理时使用。
        self.eps = eps
        # [原脚本 L100] 设置对象属性：将计算/配置结果保存到 `self.weight`，以便后续 forward/推理时使用。
        self.weight = nn.Parameter(torch.ones(dim))

    # [原脚本 L102] 定义函数/方法 `_norm`：封装可复用的计算步骤。
    def _norm(self, x):
        # [原脚本 L103] 返回：把函数/方法的输出值返回给调用者。
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    # [原脚本 L105] 定义函数/方法 `forward`：封装可复用的计算步骤。
    def forward(self, x):
        # [原脚本 L106] 返回：把函数/方法的输出值返回给调用者。
        return self.weight * self._norm(x.float()).type_as(x)


# [原脚本 L109] 函数/方法签名参数：定义参数 `dim` 的类型与默认值（仅声明，不会在此处执行计算）。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         # [原脚本 L110] 函数/方法签名参数：定义参数 `rope_scaling` 的类型与默认值（仅声明，不会在此处执行计算）。
                         rope_scaling: Optional[dict] = None):
    # [原脚本 L111] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    # [原脚本 L112] 条件分支：仅当条件成立时执行下面的缩进代码块。
    if rope_scaling is not None:
        # [原脚本 L113] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            # [原脚本 L114] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            # [原脚本 L115] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        # [原脚本 L116] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        )
        # [原脚本 L117] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            # [原脚本 L119] 赋值：计算右侧表达式并保存到变量 `inv_dim`，供后续使用。
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            # [原脚本 L120] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # [原脚本 L121] 赋值：计算右侧表达式并保存到变量 `ramp`，供后续使用。
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            # [原脚本 L122] 赋值：计算右侧表达式并保存到变量 `freqs`，供后续使用。
            freqs = freqs * (1 - ramp + ramp / factor)

    # [原脚本 L124] 赋值：计算右侧表达式并保存到变量 `t`，供后续使用。
    t = torch.arange(end, device=freqs.device)
    # [原脚本 L125] 赋值：计算右侧表达式并保存到变量 `freqs`，供后续使用。
    freqs = torch.outer(t, freqs).float()
    # [原脚本 L126] 赋值：计算右侧表达式并保存到变量 `freqs_cos`，供后续使用。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    # [原脚本 L127] 赋值：计算右侧表达式并保存到变量 `freqs_sin`，供后续使用。
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    # [原脚本 L128] 返回：把函数/方法的输出值返回给调用者。
    return freqs_cos, freqs_sin


# [原脚本 L131] 定义函数/方法 `apply_rotary_pos_emb`：封装可复用的计算步骤。
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # [原脚本 L132] 定义函数/方法 `rotate_half`：封装可复用的计算步骤。
    def rotate_half(x):
        # [原脚本 L133] 返回：把函数/方法的输出值返回给调用者。
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # [原脚本 L135] 赋值：计算右侧表达式并保存到变量 `q_embed`，供后续使用。
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    # [原脚本 L136] 赋值：计算右侧表达式并保存到变量 `k_embed`，供后续使用。
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    # [原脚本 L137] 返回：把函数/方法的输出值返回给调用者。
    return q_embed, k_embed


# [原脚本 L140] 函数/方法签名参数：定义参数 `x` 的类型与默认值（仅声明，不会在此处执行计算）。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # [原脚本 L141] 函数/方法签名续行：继续列出参数或结束括号。
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # [原脚本 L142] 函数/方法签名续行：继续列出参数或结束括号。
    bs, slen, num_key_value_heads, head_dim = x.shape
    # [原脚本 L143] 函数/方法签名续行：继续列出参数或结束括号。
    if n_rep == 1:
        # [原脚本 L144] 函数/方法签名续行：继续列出参数或结束括号。
        return x
    # [原脚本 L145] 函数/方法签名续行：继续列出参数或结束括号。
    return (
        # [原脚本 L146] 函数/方法签名续行：继续列出参数或结束括号。
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    # [原脚本 L147] 函数/方法签名续行：继续列出参数或结束括号。
    )


# [原脚本 L150] 函数/方法签名续行：继续列出参数或结束括号。
class Attention(nn.Module):
    # [原脚本 L151] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, args: MiniMindConfig):
        # [原脚本 L152] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L153] 设置对象属性：将计算/配置结果保存到 `self.num_key_value_heads`，以便后续 forward/推理时使用。
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        # [原脚本 L154] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        assert args.num_attention_heads % self.num_key_value_heads == 0
        # [原脚本 L155] 设置对象属性：将计算/配置结果保存到 `self.n_local_heads`，以便后续 forward/推理时使用。
        self.n_local_heads = args.num_attention_heads
        # [原脚本 L156] 设置对象属性：将计算/配置结果保存到 `self.n_local_kv_heads`，以便后续 forward/推理时使用。
        self.n_local_kv_heads = self.num_key_value_heads
        # [原脚本 L157] 设置对象属性：将计算/配置结果保存到 `self.n_rep`，以便后续 forward/推理时使用。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # [原脚本 L158] 设置对象属性：将计算/配置结果保存到 `self.head_dim`，以便后续 forward/推理时使用。
        self.head_dim = args.hidden_size // args.num_attention_heads
        # [原脚本 L159] 设置对象属性：将计算/配置结果保存到 `self.q_proj`，以便后续 forward/推理时使用。
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        # [原脚本 L160] 设置对象属性：将计算/配置结果保存到 `self.k_proj`，以便后续 forward/推理时使用。
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # [原脚本 L161] 设置对象属性：将计算/配置结果保存到 `self.v_proj`，以便后续 forward/推理时使用。
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # [原脚本 L162] 设置对象属性：将计算/配置结果保存到 `self.o_proj`，以便后续 forward/推理时使用。
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        # [原脚本 L163] 设置对象属性：将计算/配置结果保存到 `self.attn_dropout`，以便后续 forward/推理时使用。
        self.attn_dropout = nn.Dropout(args.dropout)
        # [原脚本 L164] 设置对象属性：将计算/配置结果保存到 `self.resid_dropout`，以便后续 forward/推理时使用。
        self.resid_dropout = nn.Dropout(args.dropout)
        # [原脚本 L165] 设置对象属性：将计算/配置结果保存到 `self.dropout`，以便后续 forward/推理时使用。
        self.dropout = args.dropout
        # [原脚本 L166] 设置对象属性：将计算/配置结果保存到 `self.flash`，以便后续 forward/推理时使用。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    # [原脚本 L169] 函数/方法签名续行：继续列出参数或结束括号。
    def forward(self,
                # [原脚本 L170] 函数/方法签名参数：定义参数 `x` 的类型与默认值（仅声明，不会在此处执行计算）。
                x: torch.Tensor,
                # [原脚本 L171] 函数/方法签名参数：定义参数 `position_embeddings` 的类型与默认值（仅声明，不会在此处执行计算）。
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                # [原脚本 L172] 函数/方法签名参数：定义参数 `past_key_value` 的类型与默认值（仅声明，不会在此处执行计算）。
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                # [原脚本 L173] 函数/方法签名续行：继续列出参数或结束括号。
                use_cache=False,
                # [原脚本 L174] 函数/方法签名参数：定义参数 `attention_mask` 的类型与默认值（仅声明，不会在此处执行计算）。
                attention_mask: Optional[torch.Tensor] = None):
        # [原脚本 L175] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        bsz, seq_len, _ = x.shape
        # [原脚本 L176] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # [原脚本 L177] 赋值：计算右侧表达式并保存到变量 `xq`，供后续使用。
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        # [原脚本 L178] 赋值：计算右侧表达式并保存到变量 `xk`，供后续使用。
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        # [原脚本 L179] 赋值：计算右侧表达式并保存到变量 `xv`，供后续使用。
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # [原脚本 L181] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        cos, sin = position_embeddings
        # [原脚本 L182] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # kv_cache实现
        # [原脚本 L185] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if past_key_value is not None:
            # [原脚本 L186] 赋值：计算右侧表达式并保存到变量 `xk`，供后续使用。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            # [原脚本 L187] 赋值：计算右侧表达式并保存到变量 `xv`，供后续使用。
            xv = torch.cat([past_key_value[1], xv], dim=1)
        # [原脚本 L188] 赋值：计算右侧表达式并保存到变量 `past_kv`，供后续使用。
        past_kv = (xk, xv) if use_cache else None

        # [原脚本 L190] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        xq, xk, xv = (
            # [原脚本 L191] 张量形状/内存布局操作：调整维度或确保连续内存布局，以便后续高效计算。
            xq.transpose(1, 2),
            # [原脚本 L192] 张量形状/内存布局操作：调整维度或确保连续内存布局，以便后续高效计算。
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            # [原脚本 L193] 张量形状/内存布局操作：调整维度或确保连续内存布局，以便后续高效计算。
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        # [原脚本 L194] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        )

        # [原脚本 L196] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            # [原脚本 L197] 赋值：计算右侧表达式并保存到变量 `output`，供后续使用。
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        # [原脚本 L198] 条件分支（否则）：当前面条件都不成立时执行。
        else:
            # [原脚本 L199] 赋值：计算右侧表达式并保存到变量 `scores`，供后续使用。
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # [原脚本 L200] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
            scores[:, :, :, -seq_len:] += torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)

            # [原脚本 L202] 条件分支：仅当条件成立时执行下面的缩进代码块。
            if attention_mask is not None:
                # [原脚本 L203] 赋值：计算右侧表达式并保存到变量 `extended_attention_mask`，供后续使用。
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # [原脚本 L204] 赋值：计算右侧表达式并保存到变量 `extended_attention_mask`，供后续使用。
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                # [原脚本 L205] 赋值：计算右侧表达式并保存到变量 `scores`，供后续使用。
                scores = scores + extended_attention_mask

            # [原脚本 L207] 赋值：计算右侧表达式并保存到变量 `scores`，供后续使用。
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # [原脚本 L208] 赋值：计算右侧表达式并保存到变量 `scores`，供后续使用。
            scores = self.attn_dropout(scores)
            # [原脚本 L209] 赋值：计算右侧表达式并保存到变量 `output`，供后续使用。
            output = scores @ xv

        # [原脚本 L211] 赋值：计算右侧表达式并保存到变量 `output`，供后续使用。
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        # [原脚本 L212] 赋值：计算右侧表达式并保存到变量 `output`，供后续使用。
        output = self.resid_dropout(self.o_proj(output))
        # [原脚本 L213] 返回：把函数/方法的输出值返回给调用者。
        return output, past_kv


# [原脚本 L216] 定义类 `FeedForward`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class FeedForward(nn.Module):
    # [原脚本 L217] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, config: MiniMindConfig):
        # [原脚本 L218] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L219] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if config.intermediate_size is None:
            # [原脚本 L220] 赋值：计算右侧表达式并保存到变量 `intermediate_size`，供后续使用。
            intermediate_size = int(config.hidden_size * 8 / 3)
            # [原脚本 L221] 赋值：计算右侧表达式并保存到变量 `config.intermediate_size`，供后续使用。
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        # [原脚本 L222] 设置对象属性：将计算/配置结果保存到 `self.gate_proj`，以便后续 forward/推理时使用。
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # [原脚本 L223] 设置对象属性：将计算/配置结果保存到 `self.down_proj`，以便后续 forward/推理时使用。
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # [原脚本 L224] 设置对象属性：将计算/配置结果保存到 `self.up_proj`，以便后续 forward/推理时使用。
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # [原脚本 L225] 设置对象属性：将计算/配置结果保存到 `self.dropout`，以便后续 forward/推理时使用。
        self.dropout = nn.Dropout(config.dropout)
        # [原脚本 L226] 设置对象属性：将计算/配置结果保存到 `self.act_fn`，以便后续 forward/推理时使用。
        self.act_fn = ACT2FN[config.hidden_act]

    # [原脚本 L228] 定义函数/方法 `forward`：封装可复用的计算步骤。
    def forward(self, x):
        # [原脚本 L229] 返回：把函数/方法的输出值返回给调用者。
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# [原脚本 L232] 定义类 `MoEGate`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MoEGate(nn.Module):
    # [原脚本 L233] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, config: MiniMindConfig):
        # [原脚本 L234] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L235] 设置对象属性：将计算/配置结果保存到 `self.config`，以便后续 forward/推理时使用。
        self.config = config
        # [原脚本 L236] 设置对象属性：将计算/配置结果保存到 `self.top_k`，以便后续 forward/推理时使用。
        self.top_k = config.num_experts_per_tok
        # [原脚本 L237] 设置对象属性：将计算/配置结果保存到 `self.n_routed_experts`，以便后续 forward/推理时使用。
        self.n_routed_experts = config.n_routed_experts

        # [原脚本 L239] 设置对象属性：将计算/配置结果保存到 `self.scoring_func`，以便后续 forward/推理时使用。
        self.scoring_func = config.scoring_func
        # [原脚本 L240] 设置对象属性：将计算/配置结果保存到 `self.alpha`，以便后续 forward/推理时使用。
        self.alpha = config.aux_loss_alpha
        # [原脚本 L241] 设置对象属性：将计算/配置结果保存到 `self.seq_aux`，以便后续 forward/推理时使用。
        self.seq_aux = config.seq_aux

        # [原脚本 L243] 设置对象属性：将计算/配置结果保存到 `self.norm_topk_prob`，以便后续 forward/推理时使用。
        self.norm_topk_prob = config.norm_topk_prob
        # [原脚本 L244] 设置对象属性：将计算/配置结果保存到 `self.gating_dim`，以便后续 forward/推理时使用。
        self.gating_dim = config.hidden_size
        # [原脚本 L245] 设置对象属性：将计算/配置结果保存到 `self.weight`，以便后续 forward/推理时使用。
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        # [原脚本 L246] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        self.reset_parameters()

    # [原脚本 L248] 函数/方法签名参数：定义参数 `None` 的类型与默认值（仅声明，不会在此处执行计算）。
    def reset_parameters(self) -> None:
        # [原脚本 L249] 函数/方法签名续行：继续列出参数或结束括号。
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    # [原脚本 L251] 函数/方法签名续行：继续列出参数或结束括号。
    def forward(self, hidden_states):
        # [原脚本 L252] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        bsz, seq_len, h = hidden_states.shape
        # [原脚本 L253] 赋值：计算右侧表达式并保存到变量 `hidden_states`，供后续使用。
        hidden_states = hidden_states.view(-1, h)
        # [原脚本 L254] 赋值：计算右侧表达式并保存到变量 `logits`，供后续使用。
        logits = F.linear(hidden_states, self.weight, None)
        # [原脚本 L255] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.scoring_func == 'softmax':
            # [原脚本 L256] 赋值：计算右侧表达式并保存到变量 `scores`，供后续使用。
            scores = logits.softmax(dim=-1)
        # [原脚本 L257] 条件分支（否则）：当前面条件都不成立时执行。
        else:
            # [原脚本 L258] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # [原脚本 L260] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # [原脚本 L262] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.top_k > 1 and self.norm_topk_prob:
            # [原脚本 L263] 赋值：计算右侧表达式并保存到变量 `denominator`，供后续使用。
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # [原脚本 L264] 赋值：计算右侧表达式并保存到变量 `topk_weight`，供后续使用。
            topk_weight = topk_weight / denominator

        # [原脚本 L266] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.training and self.alpha > 0.0:
            # [原脚本 L267] 赋值：计算右侧表达式并保存到变量 `scores_for_aux`，供后续使用。
            scores_for_aux = scores
            # [原脚本 L268] 赋值：计算右侧表达式并保存到变量 `aux_topk`，供后续使用。
            aux_topk = self.top_k
            # [原脚本 L269] 赋值：计算右侧表达式并保存到变量 `topk_idx_for_aux_loss`，供后续使用。
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            # [原脚本 L270] 条件分支：仅当条件成立时执行下面的缩进代码块。
            if self.seq_aux:
                # [原脚本 L271] 赋值：计算右侧表达式并保存到变量 `scores_for_seq_aux`，供后续使用。
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                # [原脚本 L272] 赋值：计算右侧表达式并保存到变量 `ce`，供后续使用。
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # [原脚本 L273] 选择最大的 k 个元素（这里用于 MoE 路由选择 top-k 专家）。
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                # [原脚本 L274] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    # [原脚本 L275] 选择最大的 k 个元素（这里用于 MoE 路由选择 top-k 专家）。
                    seq_len * aux_topk / self.n_routed_experts)
                # [原脚本 L276] 赋值：计算右侧表达式并保存到变量 `aux_loss`，供后续使用。
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            # [原脚本 L277] 条件分支（否则）：当前面条件都不成立时执行。
            else:
                # [原脚本 L278] 赋值：计算右侧表达式并保存到变量 `mask_ce`，供后续使用。
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # [原脚本 L279] 赋值：计算右侧表达式并保存到变量 `ce`，供后续使用。
                ce = mask_ce.float().mean(0)
                # [原脚本 L280] 赋值：计算右侧表达式并保存到变量 `Pi`，供后续使用。
                Pi = scores_for_aux.mean(0)
                # [原脚本 L281] 赋值：计算右侧表达式并保存到变量 `fi`，供后续使用。
                fi = ce * self.n_routed_experts
                # [原脚本 L282] 赋值：计算右侧表达式并保存到变量 `aux_loss`，供后续使用。
                aux_loss = (Pi * fi).sum() * self.alpha
        # [原脚本 L283] 条件分支（否则）：当前面条件都不成立时执行。
        else:
            # [原脚本 L284] 赋值：计算右侧表达式并保存到变量 `aux_loss`，供后续使用。
            aux_loss = scores.new_zeros(1).squeeze()
        # [原脚本 L285] 返回：把函数/方法的输出值返回给调用者。
        return topk_idx, topk_weight, aux_loss


# [原脚本 L288] 定义类 `MOEFeedForward`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MOEFeedForward(nn.Module):
    # [原脚本 L289] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, config: MiniMindConfig):
        # [原脚本 L290] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L291] 设置对象属性：将计算/配置结果保存到 `self.config`，以便后续 forward/推理时使用。
        self.config = config
        # [原脚本 L292] 设置对象属性：将计算/配置结果保存到 `self.experts`，以便后续 forward/推理时使用。
        self.experts = nn.ModuleList([
            # [原脚本 L293] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            FeedForward(config)
            # [原脚本 L294] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
            for _ in range(config.n_routed_experts)
        # [原脚本 L295] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        ])
        # [原脚本 L296] 设置对象属性：将计算/配置结果保存到 `self.gate`，以便后续 forward/推理时使用。
        self.gate = MoEGate(config)
        # [原脚本 L297] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if config.n_shared_experts > 0:
            # [原脚本 L298] 设置对象属性：将计算/配置结果保存到 `self.shared_experts`，以便后续 forward/推理时使用。
            self.shared_experts = nn.ModuleList([
                # [原脚本 L299] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
                FeedForward(config)
                # [原脚本 L300] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
                for _ in range(config.n_shared_experts)
            # [原脚本 L301] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            ])

    # [原脚本 L303] 定义函数/方法 `forward`：封装可复用的计算步骤。
    def forward(self, x):
        # [原脚本 L304] 赋值：计算右侧表达式并保存到变量 `identity`，供后续使用。
        identity = x
        # [原脚本 L305] 赋值：计算右侧表达式并保存到变量 `orig_shape`，供后续使用。
        orig_shape = x.shape
        # [原脚本 L306] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        # [原脚本 L308] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        topk_idx, topk_weight, aux_loss = self.gate(x)
        # [原脚本 L309] 赋值：计算右侧表达式并保存到变量 `x`，供后续使用。
        x = x.view(-1, x.shape[-1])
        # [原脚本 L310] 赋值：计算右侧表达式并保存到变量 `flat_topk_idx`，供后续使用。
        flat_topk_idx = topk_idx.view(-1)
        # [原脚本 L311] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.training:
            # [原脚本 L312] 赋值：计算右侧表达式并保存到变量 `x`，供后续使用。
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # [原脚本 L313] 赋值：计算右侧表达式并保存到变量 `y`，供后续使用。
            y = torch.empty_like(x, dtype=x.dtype)
            # [原脚本 L314] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
            for i, expert in enumerate(self.experts):
                # [原脚本 L315] 选择最大的 k 个元素（这里用于 MoE 路由选择 top-k 专家）。
                expert_out = expert(x[flat_topk_idx == i])
                # [原脚本 L316] 条件分支：仅当条件成立时执行下面的缩进代码块。
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                # [原脚本 L317] 条件分支（否则）：当前面条件都不成立时执行。
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            # [原脚本 L318] 赋值：计算右侧表达式并保存到变量 `y`，供后续使用。
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            # [原脚本 L319] 赋值：计算右侧表达式并保存到变量 `y`，供后续使用。
            y = y.view(*orig_shape)
        # [原脚本 L320] 条件分支（否则）：当前面条件都不成立时执行。
        else:
            # [原脚本 L321] 赋值：计算右侧表达式并保存到变量 `y`，供后续使用。
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        # [原脚本 L322] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if self.config.n_shared_experts > 0:
            # [原脚本 L323] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
            for expert in self.shared_experts:
                # [原脚本 L324] 赋值：计算右侧表达式并保存到变量 `y`，供后续使用。
                y = y + expert(identity)
        # [原脚本 L325] 设置对象属性：将计算/配置结果保存到 `self.aux_loss`，以便后续 forward/推理时使用。
        self.aux_loss = aux_loss
        # [原脚本 L326] 返回：把函数/方法的输出值返回给调用者。
        return y

    # [原脚本 L328] 装饰器：用于修改/标记下面的函数行为（例如 no_grad 表示推理阶段不记录梯度）。
    @torch.no_grad()
    # [原脚本 L329] 定义函数/方法 `moe_infer`：封装可复用的计算步骤。
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # [原脚本 L330] 赋值：计算右侧表达式并保存到变量 `expert_cache`，供后续使用。
        expert_cache = torch.zeros_like(x)
        # [原脚本 L331] 赋值：计算右侧表达式并保存到变量 `idxs`，供后续使用。
        idxs = flat_expert_indices.argsort()
        # [原脚本 L332] 赋值：计算右侧表达式并保存到变量 `tokens_per_expert`，供后续使用。
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # [原脚本 L333] 赋值：计算右侧表达式并保存到变量 `token_idxs`，供后续使用。
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        # [原脚本 L338] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
        for i, end_idx in enumerate(tokens_per_expert):
            # [原脚本 L339] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            # [原脚本 L340] 条件分支：仅当条件成立时执行下面的缩进代码块。
            if start_idx == end_idx:
                # [原脚本 L341] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
                continue
            # [原脚本 L342] 赋值：计算右侧表达式并保存到变量 `expert`，供后续使用。
            expert = self.experts[i]
            # [原脚本 L343] 赋值：计算右侧表达式并保存到变量 `exp_token_idx`，供后续使用。
            exp_token_idx = token_idxs[start_idx:end_idx]
            # [原脚本 L344] 赋值：计算右侧表达式并保存到变量 `expert_tokens`，供后续使用。
            expert_tokens = x[exp_token_idx]
            # [原脚本 L345] 赋值：计算右侧表达式并保存到变量 `expert_out`，供后续使用。
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # [原脚本 L346] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # [原脚本 L347] 张量形状/内存布局操作：调整维度或确保连续内存布局，以便后续高效计算。
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        # [原脚本 L349] 返回：把函数/方法的输出值返回给调用者。
        return expert_cache


# [原脚本 L352] 定义类 `MiniMindBlock`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MiniMindBlock(nn.Module):
    # [原脚本 L353] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, layer_id: int, config: MiniMindConfig):
        # [原脚本 L354] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L355] 设置对象属性：将计算/配置结果保存到 `self.num_attention_heads`，以便后续 forward/推理时使用。
        self.num_attention_heads = config.num_attention_heads
        # [原脚本 L356] 设置对象属性：将计算/配置结果保存到 `self.hidden_size`，以便后续 forward/推理时使用。
        self.hidden_size = config.hidden_size
        # [原脚本 L357] 设置对象属性：将计算/配置结果保存到 `self.head_dim`，以便后续 forward/推理时使用。
        self.head_dim = config.hidden_size // config.num_attention_heads
        # [原脚本 L358] 设置对象属性：将计算/配置结果保存到 `self.self_attn`，以便后续 forward/推理时使用。
        self.self_attn = Attention(config)

        # [原脚本 L360] 设置对象属性：将计算/配置结果保存到 `self.layer_id`，以便后续 forward/推理时使用。
        self.layer_id = layer_id
        # [原脚本 L361] 设置对象属性：将计算/配置结果保存到 `self.input_layernorm`，以便后续 forward/推理时使用。
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # [原脚本 L362] 设置对象属性：将计算/配置结果保存到 `self.post_attention_layernorm`，以便后续 forward/推理时使用。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # [原脚本 L363] 设置对象属性：将计算/配置结果保存到 `self.mlp`，以便后续 forward/推理时使用。
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    # [原脚本 L365] 定义函数/方法 `forward`：封装可复用的计算步骤。
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # [原脚本 L366] 赋值：计算右侧表达式并保存到变量 `residual`，供后续使用。
        residual = hidden_states
        # [原脚本 L367] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        hidden_states, present_key_value = self.self_attn(
            # [原脚本 L368] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            self.input_layernorm(hidden_states), position_embeddings,
            # [原脚本 L369] KV cache（缓存历史 token 的 K/V，用于推理加速）。
            past_key_value, use_cache, attention_mask
        # [原脚本 L370] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        )
        # [原脚本 L371] 赋值：计算右侧表达式并保存到变量 `hidden_states +`，供后续使用。
        hidden_states += residual
        # [原脚本 L372] 赋值：计算右侧表达式并保存到变量 `hidden_states`，供后续使用。
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        # [原脚本 L373] 返回：把函数/方法的输出值返回给调用者。
        return hidden_states, present_key_value


# [原脚本 L376] 定义类 `MiniMindModel`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MiniMindModel(nn.Module):
    # [原脚本 L377] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, config: MiniMindConfig):
        # [原脚本 L378] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__()
        # [原脚本 L379] 设置对象属性：将计算/配置结果保存到 `self.config`，以便后续 forward/推理时使用。
        self.config = config
        # [原脚本 L380] 设置对象属性：将计算/配置结果保存到 `self.vocab_size, num_hidden_layers`，以便后续 forward/推理时使用。
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # [原脚本 L381] 设置对象属性：将计算/配置结果保存到 `self.embed_tokens`，以便后续 forward/推理时使用。
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # [原脚本 L382] 设置对象属性：将计算/配置结果保存到 `self.dropout`，以便后续 forward/推理时使用。
        self.dropout = nn.Dropout(config.dropout)
        # [原脚本 L383] 设置对象属性：将计算/配置结果保存到 `self.layers`，以便后续 forward/推理时使用。
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # [原脚本 L384] 设置对象属性：将计算/配置结果保存到 `self.norm`，以便后续 forward/推理时使用。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # [原脚本 L386] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    # [原脚本 L387] 赋值：计算右侧表达式并保存到变量 `end`，供后续使用。
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    # [原脚本 L388] 赋值：计算右侧表达式并保存到变量 `rope_scaling`，供后续使用。
                                                    rope_scaling=config.rope_scaling)
        # [原脚本 L389] 设置对象属性：将计算/配置结果保存到 `self.register_buffer("freqs_cos", freqs_cos, persistent`，以便后续 forward/推理时使用。
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        # [原脚本 L390] 设置对象属性：将计算/配置结果保存到 `self.register_buffer("freqs_sin", freqs_sin, persistent`，以便后续 forward/推理时使用。
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # [原脚本 L392] 函数/方法签名续行：继续列出参数或结束括号。
    def forward(self,
                # [原脚本 L393] 函数/方法签名参数：定义参数 `input_ids` 的类型与默认值（仅声明，不会在此处执行计算）。
                input_ids: Optional[torch.Tensor] = None,
                # [原脚本 L394] 函数/方法签名参数：定义参数 `attention_mask` 的类型与默认值（仅声明，不会在此处执行计算）。
                attention_mask: Optional[torch.Tensor] = None,
                # [原脚本 L395] 函数/方法签名参数：定义参数 `past_key_values` 的类型与默认值（仅声明，不会在此处执行计算）。
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                # [原脚本 L396] 函数/方法签名参数：定义参数 `use_cache` 的类型与默认值（仅声明，不会在此处执行计算）。
                use_cache: bool = False,
                # [原脚本 L397] 函数/方法签名续行：继续列出参数或结束括号。
                **kwargs):
        # [原脚本 L398] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        batch_size, seq_length = input_ids.shape
        # [原脚本 L399] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if hasattr(past_key_values, 'layers'): past_key_values = None
        # [原脚本 L400] 赋值：计算右侧表达式并保存到变量 `past_key_values`，供后续使用。
        past_key_values = past_key_values or [None] * len(self.layers)
        # [原脚本 L401] 赋值：计算右侧表达式并保存到变量 `start_pos`，供后续使用。
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # [原脚本 L403] 赋值：计算右侧表达式并保存到变量 `hidden_states`，供后续使用。
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # [原脚本 L405] 赋值：计算右侧表达式并保存到变量 `position_embeddings`，供后续使用。
        position_embeddings = (
            # [原脚本 L406] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            self.freqs_cos[start_pos:start_pos + seq_length],
            # [原脚本 L407] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            self.freqs_sin[start_pos:start_pos + seq_length]
        # [原脚本 L408] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        )

        # [原脚本 L410] 赋值：计算右侧表达式并保存到变量 `presents`，供后续使用。
        presents = []
        # [原脚本 L411] 循环：按序遍历可迭代对象，对每个元素执行下面的代码块。
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            # [原脚本 L412] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
            hidden_states, present = layer(
                # [原脚本 L413] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
                hidden_states,
                # [原脚本 L414] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
                position_embeddings,
                # [原脚本 L415] 赋值：计算右侧表达式并保存到变量 `past_key_value`，供后续使用。
                past_key_value=past_key_value,
                # [原脚本 L416] 赋值：计算右侧表达式并保存到变量 `use_cache`，供后续使用。
                use_cache=use_cache,
                # [原脚本 L417] 赋值：计算右侧表达式并保存到变量 `attention_mask`，供后续使用。
                attention_mask=attention_mask
            # [原脚本 L418] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            )
            # [原脚本 L419] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            presents.append(present)

        # [原脚本 L421] 赋值：计算右侧表达式并保存到变量 `hidden_states`，供后续使用。
        hidden_states = self.norm(hidden_states)

        # [原脚本 L423] 赋值：计算右侧表达式并保存到变量 `aux_loss`，供后续使用。
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        # [原脚本 L424] 返回：把函数/方法的输出值返回给调用者。
        return hidden_states, presents, aux_loss


# [原脚本 L427] 定义类 `MiniMindForCausalLM`：封装一组相关的数据与方法（这里用于配置/模型组件）。
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    # [原脚本 L428] 赋值：计算右侧表达式并保存到变量 `config_class`，供后续使用。
    config_class = MiniMindConfig

    # [原脚本 L430] 定义函数/方法 `__init__`：封装可复用的计算步骤。
    def __init__(self, config: MiniMindConfig = None):
        # [原脚本 L431] 设置对象属性：将计算/配置结果保存到 `self.config`，以便后续 forward/推理时使用。
        self.config = config or MiniMindConfig()
        # [原脚本 L432] 调用父类构造函数，初始化基类状态（如 transformers 的配置/模型基类）。
        super().__init__(self.config)
        # [原脚本 L433] 设置对象属性：将计算/配置结果保存到 `self.model`，以便后续 forward/推理时使用。
        self.model = MiniMindModel(self.config)
        # [原脚本 L434] 设置对象属性：将计算/配置结果保存到 `self.lm_head`，以便后续 forward/推理时使用。
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # [原脚本 L435] 设置对象属性：将计算/配置结果保存到 `self.model.embed_tokens.weight`，以便后续 forward/推理时使用。
        self.model.embed_tokens.weight = self.lm_head.weight

    # [原脚本 L437] 函数/方法签名续行：继续列出参数或结束括号。
    def forward(self,
                # [原脚本 L438] 函数/方法签名参数：定义参数 `input_ids` 的类型与默认值（仅声明，不会在此处执行计算）。
                input_ids: Optional[torch.Tensor] = None,
                # [原脚本 L439] 函数/方法签名参数：定义参数 `attention_mask` 的类型与默认值（仅声明，不会在此处执行计算）。
                attention_mask: Optional[torch.Tensor] = None,
                # [原脚本 L440] 函数/方法签名参数：定义参数 `labels` 的类型与默认值（仅声明，不会在此处执行计算）。
                labels: Optional[torch.Tensor] = None,
                # [原脚本 L441] 函数/方法签名参数：定义参数 `past_key_values` 的类型与默认值（仅声明，不会在此处执行计算）。
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                # [原脚本 L442] 函数/方法签名参数：定义参数 `use_cache` 的类型与默认值（仅声明，不会在此处执行计算）。
                use_cache: bool = False,
                # [原脚本 L443] 函数/方法签名参数：定义参数 `logits_to_keep` 的类型与默认值（仅声明，不会在此处执行计算）。
                logits_to_keep: Union[int, torch.Tensor] = 0,
                # [原脚本 L444] 函数/方法签名续行：继续列出参数或结束括号。
                **args):
        # [原脚本 L445] 多变量赋值：将右侧表达式的多个返回值拆包赋给左侧多个变量。
        hidden_states, past_key_values, aux_loss = self.model(
            # [原脚本 L446] 赋值：计算右侧表达式并保存到变量 `input_ids`，供后续使用。
            input_ids=input_ids,
            # [原脚本 L447] 赋值：计算右侧表达式并保存到变量 `attention_mask`，供后续使用。
            attention_mask=attention_mask,
            # [原脚本 L448] 赋值：计算右侧表达式并保存到变量 `past_key_values`，供后续使用。
            past_key_values=past_key_values,
            # [原脚本 L449] 赋值：计算右侧表达式并保存到变量 `use_cache`，供后续使用。
            use_cache=use_cache,
            # [原脚本 L450] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
            **args
        # [原脚本 L451] 执行一条普通语句：实现当前模块的具体计算/逻辑（建议结合上下文注释一起看）。
        )
        # [原脚本 L452] 赋值：计算右侧表达式并保存到变量 `slice_indices`，供后续使用。
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # [原脚本 L453] 赋值：计算右侧表达式并保存到变量 `logits`，供后续使用。
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # [原脚本 L455] 赋值：计算右侧表达式并保存到变量 `loss`，供后续使用。
        loss = None
        # [原脚本 L456] 条件分支：仅当条件成立时执行下面的缩进代码块。
        if labels is not None:
            # [原脚本 L457] 赋值：计算右侧表达式并保存到变量 `shift_logits`，供后续使用。
            shift_logits = logits[..., :-1, :].contiguous()
            # [原脚本 L458] 赋值：计算右侧表达式并保存到变量 `shift_labels`，供后续使用。
            shift_labels = labels[..., 1:].contiguous()
            # [原脚本 L459] 赋值：计算右侧表达式并保存到变量 `loss`，供后续使用。
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        # [原脚本 L461] 赋值：计算右侧表达式并保存到变量 `output`，供后续使用。
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        # [原脚本 L462] 赋值：计算右侧表达式并保存到变量 `output.aux_loss`，供后续使用。
        output.aux_loss = aux_loss
        # [原脚本 L463] 返回：把函数/方法的输出值返回给调用者。
        return output
