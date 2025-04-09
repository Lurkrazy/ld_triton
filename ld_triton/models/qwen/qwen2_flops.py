
from dataclasses import dataclass


@dataclass
class Qwen2Config:
    GBS: int = 16
    SEQ_LEN: int = 2048
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    

def flops_fwd(config: Qwen2Config):
    head_dim = config.hidden_size // config.num_attention_heads
    mlp_flops = 6 * config.GBS * config.SEQ_LEN * config.hidden_size * config.intermediate_size
    self_attn_linear_flops = 4 * config.GBS * config.SEQ_LEN * config.hidden_size * config.num_attention_heads * head_dim \
                    + 4 * config.GBS * config.SEQ_LEN * config.hidden_size * config.num_key_value_heads * head_dim 
    self_attn_qkv_flops = 4 * config.GBS  * config.num_attention_heads * head_dim  * config.SEQ_LEN * config.SEQ_LEN
    
    # TFLOPS = FLOPS / 1e12
    mlp_tflops = mlp_flops / 1e12
    self_attn_linear_tflops = self_attn_linear_flops / 1e12
    self_attn_qkv_tflops = self_attn_qkv_flops / 1e12
    self_attn_tflops = self_attn_linear_tflops + self_attn_qkv_tflops
    total_tflops = mlp_tflops + self_attn_tflops

    mlp_tflops = mlp_tflops * config.num_hidden_layers
    self_attn_linear_tflops = self_attn_linear_tflops * config.num_hidden_layers
    self_attn_qkv_tflops = self_attn_qkv_tflops * config.num_hidden_layers
    self_attn_tflops = self_attn_tflops * config.num_hidden_layers
    total_tflops = total_tflops * config.num_hidden_layers

    print(f"Self Attention Linear TFLOPS: {self_attn_linear_tflops:.3f}")
    print(f"Self Attention QKV TFLOPS:    {self_attn_qkv_tflops:.3f}")
    print(f"Self Attention TFLOPS:        {self_attn_tflops:.3f}")
    print(f"MLP TFLOPS:                   {mlp_tflops:.3f}")
    print(f"Total TFLOPS:                 {total_tflops:.3f}")


def kv_cache_flops_fwd(config: Qwen2Config):
    head_dim = config.hidden_size // config.num_attention_heads
    mlp_flops = 6 * config.GBS * config.hidden_size * config.intermediate_size
    self_attn_linear_flops = 4 * config.GBS * config.hidden_size * config.num_attention_heads * head_dim \
                    + 4 * config.GBS * config.hidden_size * config.num_key_value_heads * head_dim 
    self_attn_qkv_flops = 4 * config.GBS  * config.num_attention_heads * head_dim  * config.SEQ_LEN
    
    # TFLOPS = FLOPS / 1e12
    mlp_tflops = mlp_flops / 1e12
    self_attn_linear_tflops = self_attn_linear_flops / 1e12
    self_attn_qkv_tflops = self_attn_qkv_flops / 1e12
    self_attn_tflops = self_attn_linear_tflops + self_attn_qkv_tflops
    total_tflops = mlp_tflops + self_attn_tflops

    mlp_tflops = mlp_tflops * config.num_hidden_layers
    self_attn_linear_tflops = self_attn_linear_tflops * config.num_hidden_layers
    self_attn_qkv_tflops = self_attn_qkv_tflops * config.num_hidden_layers
    self_attn_tflops = self_attn_tflops * config.num_hidden_layers
    total_tflops = total_tflops * config.num_hidden_layers

    print(f"kv_cache Self Attention Linear TFLOPS: {self_attn_linear_tflops:.3f}")
    print(f"kv_cache Self Attention QKV TFLOPS:    {self_attn_qkv_tflops:.3f}")
    print(f"kv_cache Self Attention TFLOPS:        {self_attn_tflops:.3f}")
    print(f"kv_cache MLP TFLOPS:                   {mlp_tflops:.3f}")
    print(f"kv_cache Total TFLOPS:                 {total_tflops:.3f}")

if __name__ == '__main__':
    print("==========================")
    print("Qwen/Qwen2.5-0.5B-Instruct")
    Qwen25_05B_Instruct = Qwen2Config(
        GBS=16,
        SEQ_LEN=2048,
        hidden_size=896,
        intermediate_size=4864,
        num_attention_heads=14,
        num_hidden_layers=24,
        num_key_value_heads=2
    )
    flops_fwd(Qwen25_05B_Instruct)
    print("==========================")
    print("Qwen/Qwen2.5-7B-Instruct-1M")
    Qwen25_7B_Instruct_1M = Qwen2Config(
        GBS=16,
        SEQ_LEN=2*1024,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_hidden_layers=28,
        num_key_value_heads=4
    )

    flops_fwd(Qwen25_7B_Instruct_1M)

    print("==========================")
    print("Qwen/Qwen2.5-14B-Instruct-1M")
    Qwen25_7B_Instruct_1M = Qwen2Config(
        GBS=16,
        SEQ_LEN=2*1024,
        hidden_size=5120,
        intermediate_size=13824,
        num_attention_heads=40,
        num_hidden_layers=48,
        num_key_value_heads=8
    )

    flops_fwd(Qwen25_7B_Instruct_1M)

    print("==========================")
    print("Qwen/Qwen2.5-72B-Instruct")

    Qwen25_72B_Instruct = Qwen2Config(
        GBS=16,
        SEQ_LEN=2*1024,
        hidden_size=8192,
        intermediate_size=29568,
        num_attention_heads=64,
        num_hidden_layers=80,
        num_key_value_heads=8
    )

    flops_fwd(Qwen25_72B_Instruct)
    kv_cache_flops_fwd(Qwen25_72B_Instruct)

    