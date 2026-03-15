from transformers import PretrainedConfig
import torch

class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"

    def __init__(
            self,
            num_layers=40,
            padded_vocab_size=151552,
            hidden_size=4096,
            ffn_hidden_size=13696,
            kv_channels=128,
            num_attention_heads=32,
            seq_length=8192,
            hidden_dropout=0.0,
            classifier_dropout=None,
            attention_dropout=0.0,
            layernorm_epsilon= 0.00000015625, # 1e-5,
            rmsnorm=True,
            apply_residual_connection_post_layernorm=False,
            post_layer_norm=True,
            add_bias_linear=False,
            add_qkv_bias=True,
            bias_dropout_fusion=True,
            multi_query_attention=True,
            multi_query_group_num=2,
            rope_ratio=1,
            apply_query_key_layer_scaling=False,
            attention_softmax_in_fp32=False,
            fp32_residual_connection=False,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.rope_ratio = rope_ratio
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.torch_dtype = torch_dtype
        self.use_cache=use_cache
        super().__init__(**kwargs)
