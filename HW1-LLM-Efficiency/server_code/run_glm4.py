import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math
from torch import nn
import torch.nn.functional as F
import subprocess
import time
import numpy as np
import pandas as pd
import random

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import GenerationConfig

try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10, is_flash_attn_2_available
    if is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
        print("flash_attn is available")
    else:
        print("flash_attn is not available")
except:
    pass


@dataclass
class Config:
    hidden_size: int = 4096
    ffn_hidden_size: int = 13696
    kv_channels: int = 128
    num_layers: int = 40
    num_attention_heads: int = 32
    multi_query_attention: bool = True
    multi_query_group_num: int = 2
    padded_vocab_size: int = 151552
    seq_length: int = 8192
    layernorm_epsilon: float = 0.00000015625
    torch_dtype: float = "bfloat16"

    add_qkv_bias: bool = True # Means that the qkv linear layers have bias terms.
    post_layer_norm: bool = True # At the end of all layers, there is an additional RMSNorm.
    add_bias_linear: bool = False  # The linear layers in the FFN do not have bias terms.

    is_encoder_decoder: bool = False
    
    
    use_cache: bool = True
    use_return_dict: bool = True
    
    _attn_implementation: str = "eager"
    # _attn_implementation: str = "flash_attention_2"
    
    
    

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        # 这种每隔两个维度选择一个值的方式是因为 RoPE 中每两个相邻的维度共享一个频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        # 将 inv_freq 注册为模型的缓冲区参数，定义一个不可训练的参数（即缓冲区）
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        # base -> 10000缩放
        base = base * self.rope_ratio
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)



# TODO: Implement the RMSNorm class.
class RMSNorm(torch.nn.Module):
    """
    RMSNorm implementation.
    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.
    """
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        # self.normalized_shape = normalized_shape
        self.eps = eps
        # hugging face: weight not scale
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        
    def forward(self, hidden_states: torch.Tensor):
        # Compute RMS for normalization
        # rms = hidden_states.norm(2, dim=-1, keepdim=True) / (hidden_states.shape[-1] ** 0.5)
        rms_inv = torch.rsqrt((hidden_states.to(torch.float32) * hidden_states.to(torch.float32)).mean(-1, keepdim=True) + self.eps)
        # Normalize hidden states using RMS and apply scaling
        return (self.weight * hidden_states * rms_inv).type_as(hidden_states)


# TODO: Implement the Attention class.
class Attention(torch.nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.is_causal = True

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        # sqrt(dk) scaling
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # query, key, value layer [batch, number_heads, sequence_length, hidden_size_per_head] = [b, np, sq, hn]
        
        # query_layer: [b, np, sq, hn]
        # key_layer: [b, np, sk, hn]
        # value_layer: [b, np, sk, hn]
        # attention_mask: [b, 1, sq, sk]
        # attention_scores: [b, np, sq, sk]
        
        # learn from "attention mask" -> output size = [batch, number_heads, sequence_length_q, sequence_length_k] 
        output_size = (query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))
        attention_scores = torch.einsum("bnqh,bnkh->bnqk", query_layer, key_layer) / self.norm_factor

        # attention scores and attention mask [batch, number_heads, sequence_length, sequence_length]
        if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
            attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                        device=attention_scores.device, dtype=torch.bool)
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))

        # compute attention probabilities
        # attention_probs [b, np, sq, sk]
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)
        # no attention drop out?
        # value_layer: [b, np, sk, hn]
        # context_layer: [b, np, sq, hn]
        context_layer = torch.einsum("bnqk,bnkh->bnqh", attention_probs, value_layer)
        # context_layer: [b, np, sq, hn] -> [b, sq, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [b, sq, np, hn] -> [b, sq, (np * hn)]
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # context_layer [batch, sequence_length, hidden_size] = [b, sq, np*hn]
        return context_layer


class FlashAttention2(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 如果版本大于等于 2.10，则不使用 top-left mask
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, query_states, key_states, value_states, attention_mask=None):
        # query, key, value layer [batch, number_heads, sequence_length, hidden_size_per_head] = [b, np, sq, hn]
        # [b, np, sq, hn] -> [b, sq, np, hn]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        batch_size, query_length = query_states.shape[:2]
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1
        # dropout = self.config.attention_dropout if self.training else 0.0
        dropout = 0.0
        # Contains at least one padding token in the sequence
        # if attention_mask is not None:
        #     # 根据 attention_mask 去除 qkv 中的 padding
        #     query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
        #         query_states, key_states, value_states, attention_mask, query_length
        #     )

        #     cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        #     max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        #     # flash_attn_varlen_func 是 FlashAttention 的核心函数，它支持通过调整序列长度来减少内存占用，并进行加速计算。
        #     attn_output_unpad = flash_attn_varlen_func(
        #         query_states,
        #         key_states,
        #         value_states,
        #         cu_seqlens_q=cu_seqlens_q,
        #         cu_seqlens_k=cu_seqlens_k,
        #         max_seqlen_q=max_seqlen_in_batch_q,
        #         max_seqlen_k=max_seqlen_in_batch_k,
        #         dropout_p=dropout,
        #         softmax_scale=None,
        #         causal=causal,
        #     )
        #     # 对于变长序列，计算的结果会是没有填充的，使用 pad_input 将其恢复成原来的形状
        #     attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        # else:
        #     attn_output = flash_attn_func(
        #         query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
        #     )
        attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
            )
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size_per_partition).contiguous()
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads_per_partition, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class SdpaAttention(Attention):
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             is_causal=True,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        else:
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                             attention_mask,
                                                                             dropout_p=self.config.attention_dropout if self.training else 0.0)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer



CORE_ATTENTION_CLASSES = {
    "eager": Attention,
    "sdpa": SdpaAttention,
    "flash_attention_2": FlashAttention2
}


# TODO: Implement the AttentionBlock class.
class AttentionBlock(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(AttentionBlock, self).__init__()

        self.dtype = dtype
        # 每个注意力头的维度
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )

        # hidden_size -> qkv_hidden_size (3 in 1)
        self.query_key_value = nn.Linear(
            config.hidden_size, 
            self.qkv_hidden_size, 
            bias=config.add_bias_linear or config.add_qkv_bias, # Note  
            device=device,
            dtype=dtype
        )
        # huggingface name: fc_out => dense
        self.dense = nn.Linear(
            self.projection_size, 
            config.hidden_size, 
            bias=config.add_bias_linear, 
            device=device,
            dtype=dtype
        )
    
        # self.core_attention = Attention(config)
        # self.core_attention = FlashAttention2(config)
        
        self.core_attention = CORE_ATTENTION_CLASSES[config._attn_implementation](config)
        

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):

        mixed_x_layer = self.query_key_value(hidden_states)
        # query, key, value layer:  [b, sq, h=(np * hn)]
        # context_layer [batch, sequence_length, hidden_size]
        
        # self.multi_query_attention = True
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [
                self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        # qkv: [b, sq, h] -> [b, sq, np, hn]
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        )
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        value_layer = value_layer.view(
            value_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
        )
        # qkv: [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]]

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # TODO
        # apply kv cache and use_cache
        # kv_cache concat in squence dim
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=2)
            value_layer = torch.cat((cache_v, value_layer), dim=2)
        # huggingface
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None
            
        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1] + (self.num_attention_heads_per_partition,) + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1] + (self.num_attention_heads_per_partition,) + value_layer.size()[3:]
            )
        # context_layer: [b, sq, h=(np * hn)]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)
        

        # TODO
        output = self.dense(context_layer)
        new_kv_cache = kv_cache
        # =================
        # Output. [sequence_length, batch, hidden size] = [sq, b, h]
        # =================
        return output, new_kv_cache



def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
    
    
 

# TODO: Implement the MLP class.
class MLP(torch.nn.Module):
    """MLP.
    MLP will take the input with h hidden state, project it to 4*h => ?????
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        super(MLP, self).__init__()
        # Config.ffn_hidden_size??
        # -> huggingface name:  dense_h_to_4h & dense_4h_to_h
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size * 2, bias=Config.add_bias_linear, device=device, dtype=dtype)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=Config.add_bias_linear, device=device, dtype=dtype)
        
        def swiglu(x):
            # x1, x2 = x.chunk(2, dim=-1)
            x1, x2 = torch.chunk(x, 2, dim=-1)
            # x1 使用 SiLU 激活，x2 直接相乘
            return F.silu(x1) * x2

        self.activation_func = swiglu

    def forward(self, hidden_states):
        # hidden_states: [sequence_length, batch, hidden size]
        inner_hidden_states = self.dense_h_to_4h(hidden_states)
        # inner_hidden_states: [sequence_length, batch, 4 * hidden size]
        output = self.dense_4h_to_h(self.activation_func(inner_hidden_states))
        return output


# TODO: Implement the Layer class. (transformer layer
class Layer(torch.nn.Module):
    def __init__(self, config, device):
        super(Layer, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.self_attention = AttentionBlock(config, device=device, dtype=self.dtype)
        
        config.rmsnorm = True
        LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm 
        # self.layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=self.dtype)
        
        # huggingface: input_layernorm & post_attention_layernorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=self.dtype)
        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=self.dtype)
        
        
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [sequence_length, batch, hidden size]
        
        # pre normalization for self-attention
        pre_norm_hidden_states = self.input_layernorm(hidden_states)
        
        self_attention_output, new_kv_cache = self.self_attention(
            pre_norm_hidden_states, attention_mask, rotary_pos_emb, kv_cache=kv_cache, use_cache=use_cache
        )
        
        residual_connection = hidden_states + self_attention_output

        # post normalization for self-attention
        attention2ffn = self.post_attention_layernorm(residual_connection)
        
        ffn_output = self.mlp(attention2ffn)
        # !!! 实现逻辑需要和huggingface一模一样才行
        output = ffn_output + residual_connection
    
        # output: [sequence_length, batch, hidden size]
        return output, new_kv_cache

# TODO: Implement the Transformer class.
class Transformer(torch.nn.Module):
    def __init__(self, config, device):
        super(Transformer, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm
        
        config.rmsnorm = True
        LayerNormFunc = RMSNorm if config.rmsnorm else nn.LayerNorm 
        # hugggingface: final_layernorm
        # Final layer norm before output.
        self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device, dtype=self.dtype)
        
        self.layers = nn.ModuleList([Layer(config, device=device) for i in range(self.num_layers)])
        
        
    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):

        # new_kv_caches is a tuple
        # length: num_layers, each element is a tuple of length 2 (key, value cache)
        # key shape: [batch, multi_query_group_num, seq_len, kv_channels]
        
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        new_kv_caches = () if use_cache else None

        
        all_hidden_states = () if output_hidden_states else None
        
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer = self.layers[index]
            hidden_states, kv_cache = layer(
                hidden_states,
                attention_mask,
                rotary_pos_emb,
                kv_cache=kv_caches[index],
                use_cache=use_cache
            )
            
            if use_cache:
                new_kv_caches = new_kv_caches + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # At the end of all layers, there is an additional RMSNorm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        
        return hidden_states, new_kv_caches


class GLM4(torch.nn.Module):
    def __init__(self, config, device):
        super(GLM4, self).__init__()
        self.dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
        self.word_embedding = torch.nn.Embedding(config.padded_vocab_size, config.hidden_size, dtype=self.dtype, device=device)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.seq_length = config.seq_length

        self.model = Transformer(config, device=device)
        self.output_layer = nn.Linear(config.hidden_size, config.padded_vocab_size, bias=False, dtype=self.dtype, device=device)
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=1,
                                              original_impl=True,
                                              device=device, dtype=self.dtype)

    def word_embedding_forward(self, input_ids):
        return self.word_embedding(input_ids)

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        if self.config._attn_implementation == "flash_attention_2":
            if padding_mask is not None and not padding_mask.all():
                return padding_mask
            return None
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def forward(self, input_ids, position_ids = None,
                past_key_values=None, full_attention_mask=None, attention_mask=None,
                use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        inputs_embeds = self.word_embedding_forward(input_ids)
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)
        hidden_states, presents = self.model(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache
        )
        if presents is not None and type(presents) is torch.Tensor:
            presents = presents.split(1, dim=0)
            presents = list(presents)
            presents = [list(x.squeeze(0).split(1, dim=0)) for x in presents]
            presents = [tuple([x.squeeze(0) for x in y]) for y in presents]
            presents = tuple(presents)

        return hidden_states, presents




class ChatGLMForConditionalGeneration(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    
    def __init__(self, config, device=None):
        
        pretrain_config = PretrainedConfig(is_decoder=True, is_encoder_decoder=False)
        super().__init__(pretrain_config)

        self.max_sequence_length = 2500
        self.transformer = GLM4(config, device=device)
        self.config = config
        
        _supports_flash_attn_2 = True
        _supports_sdpa = True

    def _update_model_kwargs_for_generation(
            self,
            outputs,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[:, -1:]
        lm_logits = self.transformer.output_layer(hidden_states)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return output

        return CausalLMOutputWithPast(
            logits=lm_logits,
            past_key_values=transformer_outputs[1],
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )


def get_glm4_param_name():
    huggingface_model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    model_dict = huggingface_model.state_dict()
    return model_dict.keys()

def convert_ckpt():
    huggingface_model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4-9b-chat",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    model_dict = huggingface_model.state_dict()
    new_model_dict = {}
    # model_keys_set = set(model_dict.keys())
    # new_model_keys_set = set()
    for k, v in model_dict.items():
        new_k = k
        # embedding: 'transformer.embedding.word_embeddings.weight' => 'transformer.word_embedding.weight'
        if 'transformer.embedding.word_embeddings.weight' in k:
            new_k = k.replace('transformer.embedding.word_embeddings.weight', 'transformer.word_embedding.weight')
            new_model_dict[new_k] = v
            # new_model_keys_set.add(new_k)
            continue
            
        # 'transformer.encoder.layers.0.input_layernorm.weight',
        # 'transformer.encoder.layers.0.self_attention.query_key_value.weight',
        # 'transformer.encoder.layers.0.self_attention.query_key_value.bias',
        # 'transformer.encoder.layers.0.self_attention.dense.weight',
        # 'transformer.encoder.layers.0.post_attention_layernorm.weight',
        # 'transformer.encoder.layers.0.mlp.dense_h_to_4h.weight',
        # 'transformer.encoder.layers.0.mlp.dense_4h_to_h.weight',
        # =>
        # 'transformer.model.xxx'
        if 'transformer.encoder' in new_k:
            new_k = new_k.replace('transformer.encoder', 'transformer.model')
            new_model_dict[new_k] = v
            # new_model_keys_set.add(new_k)
        # 'transformer.output_layer.weight' and 'transformer.rotary_pos_emb.inv_freq' <=>
        else:
            new_model_dict[new_k] = v
            # new_model_keys_set.add(new_k)
            
    # check
    save_path = "/flash2/aml/public/xiaojs24"
    if not os.access(save_path, os.W_OK):
        print(f"No write permission for directory: {save_path}")
        raise PermissionError(f"Cannot write to {save_path}")

    # torch.save(new_model_dict, "glm4.pt")
    torch.save(new_model_dict, os.path.join(save_path, "glm4.pt"))
    del huggingface_model
    # torch.cuda.empty_cache()
    # for i in range(torch.cuda.device_count()):
    #     torch.cuda.set_device(i)
    torch.cuda.empty_cache()

def print_gpu_memory():
    # 调用 nvidia-smi 命令
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
    print(result.stdout)
    
    
def run_glm4(tokenizer, query, model, generation_config, test_config, device):
    # convert and save weight, if no "glm4.pt"
    save_path = "/flash2/aml/public/xiaojs24"
    if not os.path.exists(os.path.join(save_path, "glm4.pt")):
        print("Converting...")
        convert_ckpt()
    else:
        print("Weight file exists, skip convert.")
    # load weight
    # print_gpu_memory()
    # torch.cuda.empty_cache()
    # for i in range(torch.cuda.device_count()):
    #     torch.cuda.set_device(i)
    #     torch.cuda.empty_cache()
    # only 1 GPU

  
    print("begin load weight...")
    checkpoint = torch.load(os.path.join(save_path,"glm4.pt"), map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.cuda()  # Move model to GPU
    # print(model.state_dict().keys())

    # query -> input
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                        add_generation_prompt=True,
                                        tokenize=True,
                                        return_tensors="pt",
                                        return_dict=True
                                        )
    # -> parallel
    # inputs = inputs.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    num_tokens = inputs['input_ids'].shape[1]
    
    # run model
    # print(model)
    
    # scale_list = [1, 10, 100, 500, 1000]
    if test_config.test_times > 1:
        # pass
        time_used = 0
        memory_used = 0
        num_tokens_list = 0
        # for i, scale in enumerate(scale_list):
        #     # query scale
        #     print(f"Test with scale/n_words: {scale}, num_tokens: {num_tokens}")
        #     query_scaled = query * scale
        #     # query_scaled = generate_random_prompt(scale)
            
        #     # query -> input
        #     inputs_scaled = tokenizer.apply_chat_template([{"role": "user", "content": query_scaled}],
        #                                         add_generation_prompt=True,
        #                                         tokenize=True,
        #                                         return_tensors="pt",
        #                                         return_dict=True
        #                                         )
        #     # -> parallel
        #     # inputs = inputs.to(device)
        #     inputs_scaled = {key: value.to(device) for key, value in inputs_scaled.items()}
        #     num_tokens = inputs_scaled['input_ids'].shape[1]
        #     num_tokens_list[i] = num_tokens
            # 开始测试
        for _ in range(test_config.test_times):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize(device=device) # 确保cuda操作完成
            start_time = time.time()
            # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=False):
            with torch.no_grad():
                # parallel: model.generate -> model.module.generate
                outputs = model.generate(**inputs, generation_config=generation_config)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                # print(outputs[0])
                # print("Outputs: ", end="")
                # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            torch.cuda.synchronize(device=device) # 再次确保cuda操作完成
            end_time = time.time()
            # scale tokens 下，单次推理耗时
            num_tokens_list = num_tokens
            time_used += (end_time - start_time) / test_config.test_times # sec
            memory_used += torch.cuda.max_memory_allocated() / (1024 ** 2) / test_config.test_times  # 转为 MB
            
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(device=device)
        start_time = time.time()
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
        with torch.no_grad():
            # parallel: model.generate -> model.module.generate
            outputs = model.generate(**inputs, generation_config=generation_config)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # print(outputs[0])
            # print("Outputs: ", end="")
            # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        torch.cuda.synchronize(device=device)
        end_time = time.time()
        num_tokens_list = num_tokens
        time_used = end_time - start_time # sec
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转为 MB
        print("Outputs: ", end="")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        
    attn_type = Config._attn_implementation
    # print(f"Attention is {attn_type}")
    # print(f"Number of tokens: {num_tokens}")
    # print(f"Time elapsed: {time_used} seconds")
    # print(f"Memory used: {memory_used} MB")
    return attn_type, num_tokens_list, time_used, memory_used


# def generate_random_prompt(n_words, word_list=None):
#     # 默认单词列表
#     if word_list is None:
#         word_list = ["Who", "are", "you", "I", "am", "they", "this", "is", "a", "test", "hello", "world"]
    
#     # 随机生成指定数量的 tokens
#     prompt = " ".join(random.choices(word_list, k=n_words))
#     return prompt

def generate_n_tokens(model, input_ids, n=100):
    generated_text = input_ids
    for _ in range(n):
        outputs = model(generated_text)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_text = torch.cat([generated_text, next_token], dim=-1)
    return generated_text




if __name__ == "__main__":
    # set environment variable
    # 设置 GPU 编号，如果单机单卡指定一个，单机多卡指定多个 GPU 编号，注意这里是"visible"啊
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'  
    torch.cuda.set_device(6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cuda device count: {torch.cuda.device_count()}, device: {device}")
    random.seed(42)
    
    MODEL_PATH = "THUDM/glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    
    generation_config = GenerationConfig(
        eos_token_id=[151329,151336,151338],
        pad_token_id= 151329,
        do_sample= True,
        # do_sample= False, # for test flashattn 禁用随机性，确保生成结果一致
        temperature= 0.8,
        max_length= 100000,
        min_new_tokens= 2000,
        max_new_tokens= 2000+10,
        top_p= 0.8,
        top_k= 1,
        # top_k=None,            # 不限制候选集大小
        # top_p=1.0,             # 使用完整的分布生成
        transformers_version= "4.44.0")
    
    # Config._attn_implementation: str = "eager"  # 'sdpa', 'flash_attention_2'
    
    model = ChatGLMForConditionalGeneration(config=Config(), device=device).eval()
    # hf_param_name = get_glm4_param_name()
    # my_param_name = model.state_dict().keys()
    
    # print(hf_param_name)
    # print("\n\n=====================\n\n")
    # print(my_param_name)
    # print("\n\n=====================\n\n")
    
    # query = "你好"
    base_query = "Who are you?" * 10
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": base_query}],
                                    add_generation_prompt=True,
                                    tokenize=True,
                                    return_tensors="pt",
                                    return_dict=True
                                    )
    # -> parallel
    # inputs = inputs.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    input_text = "Once when I was six years old I saw a magnificent picture in a book, called True Stories from Nature, about the primeval forest." * 10
    input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda()
    # query = "太空探索为人类带来了无限可能。"
    # 多推理几次，取平均时间和空间
    # scale_list = [1, 10, 50, 100, 200, 500, 1000]
    # attn_type = ""

    # N_test = 1000
    
    class TestConfig:
        def __init__(self):
            self.test_times = 10
            # self.input_scale_list = [1, 10, 100, 500, 1000]
            self.input_scale_list = [10]
            self.output_length_list = [100, 500] #, 1000] # 2000, 5000, 10000]
            # self.output_length_list = [2000, 5000]

    test_config = TestConfig()
    
    for Config._attn_implementation in ['eager', 'sdpa', 'flash_attention_2']:
        print(f"Test Config._attn_implementation: {Config._attn_implementation}")
        
        if Config._attn_implementation == 'eager':#  or Config._attn_implementation == 'sdpa':
            _enable_flash = False
            _enable_math=True
            _enable_mem_efficient=False
        else:
            _enable_flash = True
            _enable_math=False
            _enable_mem_efficient=False
        
        
        model = ChatGLMForConditionalGeneration(config=Config(), device=device).eval()
        model.cuda()
        
        num_tokens_list = np.zeros(len(test_config.output_length_list))
        avg_time_used = np.zeros(len(test_config.output_length_list))
        avg_memory_used = np.zeros(len(test_config.output_length_list))
        
        
        # attn_type, num_tokens, time_used, memory_used = run_glm4(tokenizer, base_query, model, generation_config, test_config, device)
        
        # for i, output_length in enumerate(test_config.output_length_list):
        #     generation_config.min_new_tokens = output_length
        #     generation_config.max_new_tokens = output_length + 5
        #     print(f"Test output_length: {output_length}")
        #     # attn_type, num_tokens, time_used, memory_used = run_glm4(tokenizer, base_query, model, generation_config, test_config, device)
        #     generated_text = generate_n_tokens(model, inputs)
            
        #     num_tokens_list[i] = num_tokens
        #     avg_time_used[i] = time_used
        #     avg_memory_used[i] = memory_used
        
        
        # Measure inference time with FlashAttention
        with torch.backends.cuda.sdp_kernel(enable_flash=_enable_flash, enable_math=_enable_math, enable_mem_efficient=_enable_mem_efficient):
            torch.cuda.synchronize()
            start_time = time.time()
            # print(inputs)
            generated_text_flash = generate_n_tokens(model, input_ids, 1000)
            torch.cuda.synchronize()
            flash_inference_time = time.time() - start_time
        print(f'{Config._attn_implementation} Attention Inference Time: {flash_inference_time:.3f} seconds')
        
        # attn_type, num_tokens_list, time_used, memory_used = run_glm4(tokenizer, base_query, model, generation_config, TestConfig, device)
            
        # save as csv
        # save_path = "/flash2/aml/public/xiaojs24"
        # df_test_res = pd.DataFrame({
        #     "Attention": attn_type,
        #     "tokens": num_tokens_list,
        #     "time": time_used,
        #     "memory": memory_used
        # })
        # df_test_res.to_csv(os.path.join(save_path, f"glm4_{attn_type}_test.csv"), index=False)
        
        # print(f"Attention is {attn_type}")
        # for i, input_scale in enumerate(test_config.output_length_list):
        #     print(f"Test {i}: Number of tokens: {num_tokens_list[i]}")
        #     print(f"Average time elapsed: {avg_time_used[i]} seconds")
        #     print(f"Average memory used: {avg_memory_used[i]} MB")
