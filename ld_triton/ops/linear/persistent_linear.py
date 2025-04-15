
import torch
import triton
import triton.language as tl


autotune_config = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                    num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                    num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                    num_warps=2),
    # Good config for fp8 inputs.
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                    num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                    num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                    num_warps=4)
]


@triton.autotune(
    configs=autotune_config,
    key=['M', 'N', 'K'],
)
@triton.jit
def _persistent_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    NUM_SMS,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # : tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    pid_m = 0
    pid_n = 0
    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            # pid_m = tile_id // num_pid_n
            # pid_n = tile_id % num_pid_n
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

        if ki == k_tiles - 1:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            c = accumulator.to(tl.float16)
            tl.store(c_ptrs, c, mask=c_mask)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        

class _persistent_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        shape = input.shape
        input = input.view(-1, shape[-1])
        M, K = input.shape
        N, K = weight.shape
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        dtype = input.dtype
        output = torch.empty((M, N), dtype=dtype, device=input.device)
        grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])), )
        _persistent_matmul_kernel[grid](
            input, weight, output, 
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            NUM_SMS,
        )

        input = input.view(*shape)
        output = output.view(*shape[:-1], N)
        return output
    

persistent_linear = _persistent_linear.apply


if __name__ == '__main__':
    M = 16
    in_features = 128
    out_features = 256
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
    input = torch.randn(M, in_features, requires_grad=True, **factory_kwargs)
    weight = torch.randn(out_features, in_features, requires_grad=True, **factory_kwargs)
    # bias = torch.randn(out_features, requires_grad=True, **factory_kwargs)
    bias = None

    output = torch.functional.F.linear(input, weight, bias)
    # doutput = torch.rand_like(output)
    # output.backward(doutput, retain_graph=True)
    # dinput, input.grad = input.grad.clone(), None
    # dweight, weight.grad = weight.grad.clone(), None
    # dbias, bias.grad = bias.grad.clone(), None
    
    persistent_output = persistent_linear(input, weight, bias)
    # print(f'output: {output}, persistent_linear: {persistent_output}')
    

    batch_size_and_seqlen = [
        # train
        (1, 2048), 
        # (1, 1024 * 1024), # long context
        (2, 2048),
        # inference
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (8, 1),
        (16, 1),
    ]
    weight_shapes = [
        # 0.5B
        (896, 4864),
        (4864, 896),
        # 3b
        (2048, 11008),
        (11008, 2048),
        # 7B
        (3584, 18944),
        (18944, 3584),
        # 14B
        (5120, 13824),
        (13824, 5120),
        # 32B
        (5120, 27648),
        (27648, 5120),
        # 72B
        (8192, 29696),
        (29696, 8192),
    ]
    
    hardware_tflops = 0
    hardware_gmemorys = torch.cuda.get_device_properties('cuda').total_memory / 1024 / 1024 / 1024
    hardware_name = torch.cuda.get_device_name(0)
    
    if hardware_name == "NVIDIA GeForce RTX 3060":
        hardware_tflops = 25.3
    elif hardware_name == "NVIDIA GeForce RTX 3090":
        hardware_tflops = 71
    else:
        raise ValueError("Unsupported GPU, please set hardware_tflops manually")
    
    

    for batch_size, seqlen in batch_size_and_seqlen:
        for shape in weight_shapes:
            flops = 2 * batch_size * seqlen * shape[0] * shape[1]
            gmemorys = input.dtype.itemsize * \
                      (batch_size * seqlen * shape[0] +  shape[1] * shape[0] + batch_size * seqlen * shape[1]) \
                      / 1024 / 1024 / 1024
            if gmemorys > hardware_gmemorys:
                print(f'hardware info: hardware_name: {hardware_name}, hardware_tflops: {hardware_tflops}, hardware_gmemroys: {hardware_gmemorys: .3f}')
                print(f'batch_size: {batch_size}, seqlen: {seqlen}, in_features: {shape[0]}, out_features: {shape[1]}, gmemorys: {gmemorys:.3f} GB, '
                      )
                continue
            input = torch.randn(batch_size * seqlen, shape[0], requires_grad=True, **factory_kwargs)
            weight = torch.randn(shape[1], shape[0], requires_grad=True, **factory_kwargs)
            output = torch.functional.F.linear(input, weight, bias)
            persistent_output = persistent_linear(input, weight, bias)
            
            rtol = 1e-0
            atol = 1e-0
            assert torch.allclose(output, persistent_output, rtol=rtol, atol=atol), f"Output mismatch: {output} != {persistent_output}"
            for func in [torch.functional.F.linear, persistent_linear]:
                ms = triton.testing.do_bench(lambda: func(input, weight, bias),)
                TFLOPS = (flops * 1e-12) / (ms * 1e-3)
                MFU = 100 *(TFLOPS / hardware_tflops)
                print(f'func: {func.__name__}, batch_size: {batch_size}, seqlen: {seqlen}, '
                    f'in_features: {shape[0]}, out_features: {shape[1]}, '
                    f'TFLOPS: {TFLOPS:.3f} TFLOPS/s, MFU: {MFU:.3f}%, ')
    # print(f'Time: {ms} ms')