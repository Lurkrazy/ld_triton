
import triton
import torch
from ld_triton.ops.linear.triton_linear import triton_linear


def mfu_linear(batch_size, SEQ_LEN, in_features, out_features, bias=True, dtype=torch.float16, hardware_tflops=25.3):
    factory_kwargs = {'device': 'cuda', 'dtype': dtype, 'requires_grad': False}
    input = torch.randn((batch_size, SEQ_LEN, in_features), **factory_kwargs)
    weight = torch.randn((out_features, in_features), **factory_kwargs)
    if bias:
        bias = torch.randn((out_features), **factory_kwargs)
    else:
        bias = None
    output = torch.nn.functional.linear(input, weight, bias)

    ms = triton.testing.do_bench(
        # lambda: torch.nn.functional.linear(input, weight, bias)
        lambda: triton_linear(input, weight, bias)
    )

    flops = 2 * batch_size * SEQ_LEN * in_features * out_features
    tflops = (flops * 1e-12) / (ms * 1e-3)
    print(
        f"Config: batch_size={batch_size}, SEQ_LEN={SEQ_LEN}, hidden_size={in_features}, intermediate_size={out_features}, dtype={dtype}"
    )
    print(f"FLOPs: {tflops:.3f} TFLOPs/s")
    print(f"MFU: {100 * tflops / hardware_tflops:.2f}%")


def mfu_mha():
    torch.nn.functional.multi_head_attention_forward()
    pass

if __name__ == "__main__":
    hardware_tflops = 0
    if torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3060":
        hardware_tflops = 25.3
    elif torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 3090":
        hardware_tflops = 71
    else:
        raise ValueError("Unsupported GPU, please set hardware_tflops manually")
    
    for dtype in [torch.float16]:
        # for batch_size in [1, 3, 8, 16]: # inference
        for batch_size in [16]: # inference
            # for SEQ_LEN in [1, 512, 1024, 2048]:
            for SEQ_LEN in [2048]:
                for in_features, out_features in [
                    # 0.5B
                    (896, 4864),
                    (4864, 896),
                    # 7B
                    (3584, 18944),
                    (18944, 3584),
                ]:
                    mfu_linear(batch_size, SEQ_LEN, in_features, out_features, dtype=dtype, hardware_tflops=hardware_tflops)
