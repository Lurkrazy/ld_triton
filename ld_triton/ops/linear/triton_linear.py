
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
def _triton_linear_kernel(
        A_ptr, B_ptr, C_ptr, bias_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if bias_ptr is not None:
        offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        bias_ptrs = bias_ptr + offs_bias
        bias = tl.load(bias_ptrs, mask=offs_bias < N, other=0.0)
        accumulator += bias

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def _triton_bias_kernel(
        output_ptr, input_ptr,
        n_rows, n_cols,
        input_row_stride,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_rows
    output_ptrs = output_ptr + pid
    input_ptrs = input_ptr + (pid + col_offsets * input_row_stride)

    input = tl.load(input_ptrs, mask=mask, other=0.0)
    output = tl.sum(input, axis=0)
    tl.store(output_ptrs, output)


class _triton_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        M, K = input.shape
        N, K = weight.shape
        output = torch.empty((M, N), device = input.device, dtype=input.dtype)
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        _triton_linear_kernel[grid](
            input, weight, output, bias,
            M, N, K, 
            input.stride(0), input.stride(1), 
            weight.stride(1), weight.stride(0), 
            output.stride(0), output.stride(1),
        )
        ctx.save_for_backward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        M, N = grad_output.shape
        if input.requires_grad:
            _, K = weight.shape
            grad_input = torch.empty((M, K), device=input.device, dtype=input.dtype)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),)
            _triton_linear_kernel[grid](
                grad_output, weight, grad_input, None,
                M, K, N, 
                grad_output.stride(0), grad_output.stride(1), 
                weight.stride(0), weight.stride(1), 
                grad_input.stride(0), grad_input.stride(1),
            )
        if weight.requires_grad:
            _, K = input.shape
            grad_weight = torch.empty((N, K), device=weight.device, dtype=weight.dtype)
            grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_N']),)
            _triton_linear_kernel[grid](
                grad_output, input, grad_weight, None,
                N, K, M, 
                grad_output.stride(1), grad_output.stride(0), 
                input.stride(0), input.stride(1), 
                grad_weight.stride(0), grad_weight.stride(1),
            )

        if bias.requires_grad:
            shape = grad_output.shape
            n_rows, n_cols = grad_output.shape
            grad_bias = torch.empty_like(bias)
            BLOCK_SIZE = triton.next_power_of_2(n_rows)
            _triton_bias_kernel[(n_cols, )](
                grad_bias, grad_output,
                n_rows, n_cols,
                grad_output.stride(0),
                BLOCK_SIZE
            )

            grad_output = grad_output.view(*shape)
        return grad_input, grad_weight, grad_bias


triton_linear = _triton_linear.apply
