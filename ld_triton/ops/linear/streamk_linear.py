
import torch
import triton
import triton.language as tl
import triton.profiler as proton

autotune_config = [
    triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=3,
                    num_warps=8),
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


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    BLOCK_SIZE_M = args["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = args["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = args["BLOCK_SIZE_K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}, BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}]"
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "tiles_per_update" in args:
        ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}, tiles_per_update={args['tiles_per_update']:02}]"
    if "C_ptr" in args:
        bytes_per_elem = args["C_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.autotune(
    configs=autotune_config,
    key=['M', 'N', 'K'],
    reset_to_zero=['C_ptr']
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def _streamk_matmul_kernel(
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
    sm_id = tl.program_id(axis=0) # sms id
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)

    if num_tiles % NUM_SMS != 0:
        if num_tiles < NUM_SMS:
            tiles_per_SM = num_tiles // NUM_SMS # - 1
            streamk_titles = num_tiles % NUM_SMS # + NUM_SMS
        else:
            tiles_per_SM = num_tiles // NUM_SMS - 1
            streamk_titles = num_tiles % NUM_SMS + NUM_SMS
        streamk_ktitles = streamk_titles * k_tiles
        streamk_h = streamk_ktitles % NUM_SMS
        if streamk_h != 0:
            if sm_id < streamk_h:
                streamk_ktitles_per_SM = streamk_ktitles // NUM_SMS + 1
                streamk_start_id = sm_id * streamk_ktitles_per_SM
            else:
                streamk_ktitles_per_SM = streamk_ktitles // NUM_SMS
                streamk_start_id = sm_id * streamk_ktitles_per_SM + streamk_h
        else:
            streamk_ktitles_per_SM = streamk_ktitles // NUM_SMS
            streamk_start_id = sm_id * streamk_ktitles_per_SM
        tile_id = streamk_titles + sm_id - NUM_SMS
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(streamk_start_id, streamk_start_id + streamk_ktitles_per_SM):
            pid_m = k // (k_tiles * num_pid_n)
            pid_n = (k // k_tiles) % num_pid_n
            pid_k = k % k_tiles

            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k  = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)

            a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
            b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            
            a = tl.load(a_ptrs, mask=offs_k_for_mask[None, :] < K - pid_k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k_for_mask[:, None] < K - pid_k * BLOCK_SIZE_K, other=0.0)
            acc = tl.dot(a, b, acc)

            if (k == streamk_start_id + streamk_ktitles_per_SM - 1 or pid_k == k_tiles - 1):
            # if (k == streamk_start_id + streamk_ktitles_per_SM - 1 or pid_k == k_tiles - 1):
                offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                c = acc.to(tl.float16)
                tl.atomic_add(c_ptrs, c, mask=c_mask)
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        tiles_per_SM = num_tiles // NUM_SMS
        streamk_titles = 0
        tile_id = sm_id - NUM_SMS
  
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
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            # group_id = tile_id // num_pid_in_group
            # first_pid_m = group_id * GROUP_SIZE_M
            # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            # pid_m = first_pid_m + (tile_id % group_size_m)
            # pid_n = (tile_id % num_pid_in_group) // group_size_m

            start_m = pid_m * BLOCK_SIZE_M
            start_n = pid_n * BLOCK_SIZE_N
            offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
            offs_am = tl.where(offs_am < M, offs_am, 0)
            offs_bn = tl.where(offs_bn < N, offs_bn, 0)
            # offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            # offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        
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
        

class _streamk_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        shape = input.shape
        input = input.view(-1, shape[-1])
        M, K = input.shape
        N, K = weight.shape
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        dtype = input.dtype
        output = torch.zeros((M, N), dtype=dtype, device=input.device)
        grid = (NUM_SMS, )
        _streamk_matmul_kernel[grid](
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
    

streamk_linear = _streamk_linear.apply


if __name__ == '__main__':
    torch.set_printoptions(profile='full')
    M = 16 * 8
    in_features = 16 * 8
    out_features = 16 * 8
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
    input = torch.randn(M, in_features, requires_grad=True, **factory_kwargs)
    weight = torch.randn(out_features, in_features, requires_grad=True, **factory_kwargs)
    # bias = torch.randn(out_features, requires_grad=True, **factory_kwargs)
    bias = None

    
    # doutput = torch.rand_like(output)
    # output.backward(doutput, retain_graph=True)
    # dinput, input.grad = input.grad.clone(), None
    # dweight, weight.grad = weight.grad.clone(), None
    # dbias, bias.grad = bias.grad.clone(), None

    def show_profile(precision, profile_name):
        import triton.profiler.viewer as proton_viewer
        metrics = ["time/ms"]
        if precision == 'fp8':
            metrics = ["tflop8/s"] + metrics
        elif precision == 'fp16':
            metrics = ["tflop16/s"] + metrics
        file_name = f"{profile_name}.hatchet"
        proton_viewer.parse(metrics, file_name, depth=100)

    # proton.start("matmul", hook="triton")
    output = torch.functional.F.linear(input, weight, bias)
    streamk_output = streamk_linear(input, weight, bias)

    # print(f"output: {output}")
    # print(f"streamk_output: {streamk_output}")
    rtol = 0.1
    atol = 0.1
    if not torch.allclose(output, streamk_output, rtol=rtol, atol=rtol):
        print(torch.isclose(output, streamk_output, rtol=rtol, atol=atol))
        print(output[[torch.isclose(output, streamk_output, rtol=rtol, atol=atol) != True]])
        print(streamk_output[[torch.isclose(output, streamk_output, rtol=rtol, atol=atol) != True]])
    assert torch.allclose(output, streamk_output, rtol=rtol, atol=rtol)
    # proton.finalize()
    # show_profile('fp16', "matmul")

    output = torch.functional.F.linear(input, weight, bias)
    streamk_output = streamk_linear(input, weight, bias)

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
            streamk_output = streamk_linear(input, weight, bias)
            
            rtol = 1e-1
            atol = 1e-1

            assert torch.allclose(output, streamk_output, rtol=rtol, atol=atol), f"Output mismatch: {output} != {streamk_output}"
            for func in [torch.functional.F.linear, streamk_linear]:
                ms = triton.testing.do_bench(lambda: func(input, weight, bias),)
                TFLOPS = (flops * 1e-12) / (ms * 1e-3)
                MFU = 100 *(TFLOPS / hardware_tflops)
                print(f'func: {func.__name__}, batch_size: {batch_size}, seqlen: {seqlen}, '
                    f'in_features: {shape[0]}, out_features: {shape[1]}, '
                    f'TFLOPS: {TFLOPS:.3f} TFLOPS/s, MFU: {MFU:.3f}%, ')
