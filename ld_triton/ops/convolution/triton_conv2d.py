

import torch
import triton
import triton.language as tl
import pytest


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_fwd_kernel(
    output_ptr, input_ptr, weight_ptr, bias_ptr, 
    N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (P * Q)
    npq_residual = gemm_i % (P * Q)
    p = npq_residual // Q
    q = npq_residual % Q
    k = gemm_j

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c = gemm_k // (R * S)
        crs_residual = gemm_k % (R * S)
        r = crs_residual // S
        s = crs_residual % S
        
        # triton broadcast rules is same as numpy
        # p: [BLOCK_SIZE_M], p[:, None]: [BLOCK_SIZE_M, 1]
        # r: [BLOCK_SIZE_K], r[None, :]: [1, BLOCK_SIZE_K]
        # h: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        h = p[:, None] * str_h + r[None, :] * dil_h - pad_h
        # q: [BLOCK_SIZE_M], q[:, None]: [BLOCK_SIZE_M, 1]
        # s: [BLOCK_SIZE_K], s[None, :]: [1, BLOCK_SIZE_K]
        # w: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        w = q[:, None] * str_w + s[None, :] * dil_w - pad_w

        mask_input = (h >= 0) & (h < H) & (w >= 0) & (w < W)
        mask_weight = (r[:, None] < R) & (s[:, None] < S) & (c[:, None] < C)

        # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
        # c: [BLOCK_SIZE_K], c[None, :]: [1, BLOCK_SIZE_K]
        # h: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # w: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # offs_input: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        offs_input = n[:, None] * C * H * W + c[None, :] * H * W + h * W + w
        # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
        # c: [BLOCK_SIZE_K], c[:, None]: [BLOCK_SIZE_K, 1]
        # r: [BLOCK_SIZE_K], r[:, None]: [BLOCK_SIZE_K, 1]
        # s: [BLOCK_SIZE_K], s[:, None]: [BLOCK_SIZE_K, 1]
        # offs_weight: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        offs_weight = k[None, :] * C * R * S + c[:, None] * R * S + r[:, None] * S + s[:, None]

        input_ptrs = input_ptr + offs_input
        weight_ptrs = weight_ptr + offs_weight
        
        input_data = tl.load(input_ptrs, mask=mask_input, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc = tl.dot(input_data, weight_data, acc)

    if bias_ptr is not None:
        # k: [BLOCK_SIZE_N], k[None, :]: [1, BLOCK_SIZE_N]
        # offs_bias: [1, BLOCK_SIZE_N]
        offs_bias = k[None, :]
        bias_ptrs = bias_ptr + offs_bias
        bias_data = tl.load(bias_ptrs)
        acc = acc + bias_data

    acc = acc.to(tl.float16)
    # gemm_i: [BLOCK_SIZE_M], gemm_i[:, None]: [BLOCK_SIZE_M, 1]
    # gemm_j: [BLOCK_SIZE_N], gemm_j[None, :]: [1, BLOCK_SIZE_N]
    # offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_output = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_output = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    output_ptrs = output_ptr + offs_output
    tl.store(output_ptrs, acc, mask=mask_output)
        

def _implicit_gemm_conv2d_fwd(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride, padding, dilation):
    N, C, H, W = input.shape
    K, C, R, S = weight.shape
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    output = torch.empty((N, K, P, Q), dtype=input.dtype, device=input.device)
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    debug = False
    if debug:
        pgm = _implicit_gemm_conv2d_fwd_kernel[grid](
            output, input, weight, bias, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
        ttir = pgm.asm['ttir']
        ttgir = pgm.asm['ttgir']
        llir = pgm.asm['llir']
        ptx = pgm.asm['ptx']
        print(f'ttir: {ttir}')
        print(f'ttgir: {ttgir}')
        print(f'llir: {llir}')
        print(f'ptx: {ptx}')
    else:
        _implicit_gemm_conv2d_fwd_kernel[grid](
            output, input, weight, bias, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
    return output


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_input_bwd_kernel(
    dinput_ptr, doutput_ptr, weight_ptr, 
    N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    n = gemm_i // (H * W)
    nhw_residual = gemm_i % (H * W)
    h = nhw_residual // W
    w = nhw_residual % W
    c = gemm_j

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        k = gemm_k // (R * S)
        krs_residual = gemm_k % (R * S)
        r = krs_residual // S
        s = krs_residual % S

        # triton broadcast rules is same as numpy
        # h: [BLOCK_SIZE_M], h[:, None]: [BLOCK_SIZE_M, 1]
        # r: [BLOCK_SIZE_K], r[None, :]: [1, BLOCK_SIZE_K]
        # p: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        h_tmp = (h[:, None] + pad_h - r[None, :] * dil_h)
        p = h_tmp // str_h
        mask_p = (h_tmp % str_h == 0) & (p >= 0) & (p < P)
        # w: [BLOCK_SIZE_M], w[:, None]: [BLOCK_SIZE_M, 1]
        # s: [BLOCK_SIZE_K], s[None, :]: [1, BLOCK_SIZE_K]
        # q: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        w_tmp = (w[:, None] + pad_w - s[None, :] * dil_w)
        q = w_tmp // str_w
        mask_q = (w_tmp % str_w == 0) & (q >= 0) & (q < Q)
        # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
        # k: [BLOCK_SIZE_K], k[None, :]: [1, BLOCK_SIZE_K]
        # mask_doutput: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        mask_doutput = (n[:, None] < N) & (k[None, :] < K) & mask_p & mask_q
        # k: [BLOCK_SIZE_K], k[:, None]: [BLOCK_SIZE_K, 1]
        # c: [BLOCK_SIZE_N], c[None, :]: [1, BLOCK_SIZE_N]
        # r: [BLOCK_SIZE_K], r[:, None]: [BLOCK_SIZE_K, 1]
        # s: [BLOCK_SIZE_K], s[:, None]: [BLOCK_SIZE_K, 1]
        # mask_weight: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        mask_weight = (k[:, None] < K) & (c[None, :] < C) & (r[:, None] < R) & (s[:, None] < S)

        # n: [BLOCK_SIZE_M], n[:, None]: [BLOCK_SIZE_M, 1]
        # k: [BLOCK_SIZE_K], k[None, :]: [1, BLOCK_SIZE_K]
        # p: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # q: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        # doutput[n, k, p, q]: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        offs_doutput = n[:, None] * K * P * Q + k[None, :] * P * Q + p * Q + q
        # weight_ptr[k, c, r, s]
        # k: [BLOCK_SIZE_K], k[:, None]: [BLOCK_SIZE_K, 1]
        # c: [BLOCK_SIZE_N], c[None, :]: [1, BLOCK_SIZE_N]
        # r: [BLOCK_SIZE_K], r[:, None]: [BLOCK_SIZE_K, 1]
        # s: [BLOCK_SIZE_K], s[:, None]: [BLOCK_SIZE_K, 1]
        # weight[k, c, r, s]: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        offs_weight = k[:, None] * C * R * S + c[None, :] * R * S + r[:, None] * S + s[:, None]
        
        doutput_ptrs = doutput_ptr + offs_doutput
        weight_ptrs = weight_ptr + offs_weight

        doutput_data = tl.load(doutput_ptrs, mask=mask_doutput, other=0.0)
        weight_data = tl.load(weight_ptrs, mask=mask_weight, other=0.0)

        acc = tl.dot(doutput_data, weight_data, acc)

    acc = acc.to(tl.float16)
    # gemm_i: [BLOCK_SIZE_M], gemm_i[:, None]: [BLOCK_SIZE_M, 1]
    # gemm_j: [BLOCK_SIZE_N], gemm_j[None, :]: [1, BLOCK_SIZE_N]
    # offs_output: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    offs_output = gemm_i[:, None] * GEMM_N + gemm_j[None, :]
    mask_output = (gemm_i[:, None] < GEMM_M) & (gemm_j[None, :] < GEMM_N)
    dinput_ptrs = dinput_ptr + offs_output
    tl.store(dinput_ptrs, acc, mask=mask_output)


def _implicit_gemm_conv2d_input_bwd(doutput, weight, N, C, H, W, K, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * H * W
    GEMM_N = C
    GEMM_K = K * R * S

    dinput = torch.zeros((N, C, H, W), dtype=doutput.dtype, device=doutput.device)
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    debug = False
    if debug:
        pgm = _implicit_gemm_conv2d_input_bwd_kernel[grid](
            dinput, doutput, weight, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
        ttir = pgm.asm['ttir']
        ttgir = pgm.asm['ttgir']
        llir = pgm.asm['llir']
        ptx = pgm.asm['ptx']
        print(f'ttir: {ttir}')
        print(f'ttgir: {ttgir}')
        print(f'llir: {llir}')
        print(f'ptx: {ptx}')
    else:
        _implicit_gemm_conv2d_input_bwd_kernel[grid](
            dinput, doutput, weight, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
    return dinput


@triton.autotune(
    configs=get_autotune_config(),
    key=['N', 'C', 'H', 'W', 'K', 'P', 'Q', 'R', 'S', 'str_h', 'str_w', 'pad_h', 'pad_w', 'dil_h', 'dil_w']
)
@triton.jit
def _implicit_gemm_conv2d_weight_bwd_kernel(
    dweight_ptr, doutput_ptr, input_ptr, 
    N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr
):             
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    gemm_i = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % GEMM_M
    gemm_j = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % GEMM_N

    k = gemm_i
    c = gemm_j // (R * S)
    crs_residual = gemm_j % (R * S)
    r = crs_residual // S
    s = crs_residual % S

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for idx_k in range(0, tl.cdiv(GEMM_K, BLOCK_SIZE_K)):
        gemm_k = (idx_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        n = gemm_k // (P * Q)
        npq_residual = gemm_k % (P * Q)
        p = npq_residual // Q
        q = npq_residual % Q
        # triton broadcast rules is same as numpy

        # n: [BLOCK_SIZE_K], n[None, :]: [1, BLOCK_SIZE_K]
        # k: [BLOCK_SIZE_M], k[:, None]: [BLOCK_SIZE_M, 1]
        # p: [BLOCK_SIZE_K], p[None, :]: [1, BLOCK_SIZE_K]
        # q: [BLOCK_SIZE_K], q[None, :]: [1, BLOCK_SIZE_K]
        # offs_doutput: [BLOCK_SIZE_M, BLOCK_SIZE_N]
        # doutput[n, k, p, q]
        offs_doutput = n[None, :] * K * P * Q + k[:, None] * P * Q + p[None, :] * Q + q[None, :]

        # p: [BLOCK_SIZE_K], p[:, None]: [BLOCK_SIZE_K, 1]
        # r: [BLOCK_SIZE_N], r[None, :]: [1, BLOCK_SIZE_N]
        # h: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        h = p[:, None] * str_h + r[None, :] * dil_h - pad_h
        # q: [BLOCK_SIZE_K], q[:, None]: [BLOCK_SIZE_K, 1]
        # s: [BLOCK_SIZE_N], s[None, :]: [1, BLOCK_SIZE_N]
        # w: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        w = q[:, None] * str_w + s[None, :] * dil_w - pad_w
        if h < 0 or h >= H or w < 0 or w >= W:
            continue

        # input[n, c, h, w]
        # n: [BLOCK_SIZE_K], n[:, None]: [BLOCK_SIZE_K, 1]
        # c: [BLOCK_SIZE_N], c[None, :]: [1, BLOCK_SIZE_N]
        # h: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # w: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # offs_input: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        # input[n, c, h, w]
        offs_input = n[:, None] * C * H * W + c[None, :] * H * W + h * W + w

        mask_doutput = (n[None, :] < N) & (k[:, None] < K) & (p[None, :] < P) & (q[None, :] < Q)
        mask_input = (n[:, None] < N) & (c[None, :] < C) & (h < H) & (w < W) & (h >= 0) & (w >= 0)

        acc += input_ptr[n, c, h, w] * doutput[n, k, p, q]

def _implicit_gemm_conv2d_weight_bwd(doutput, input, N, C, H, W, K, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = K
    GEMM_N = C * R * S
    GEMM_K = N * P * Q

    dweight = torch.zeros((K, C, R, S), dtype=doutput.dtype, device=doutput.device)
    
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_SIZE_M']) * triton.cdiv(GEMM_N, META['BLOCK_SIZE_N']), )
    debug = False
    if debug:
        pgm = _implicit_gemm_conv2d_weight_bwd_kernel[grid](
            dweight, doutput, input, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
        ttir = pgm.asm['ttir']
        ttgir = pgm.asm['ttgir']
        llir = pgm.asm['llir']
        ptx = pgm.asm['ptx']
        print(f'ttir: {ttir}')
        print(f'ttgir: {ttgir}')
        print(f'llir: {llir}')
        print(f'ptx: {ptx}')
    else:
        _implicit_gemm_conv2d_weight_bwd_kernel[grid](
            dweight, doutput, input, 
            N, C, H, W, K, P, Q, R, S, str_h, str_w, pad_h, pad_w, dil_h, dil_w,
            GEMM_M, GEMM_N, GEMM_K,
        )
    return dweight


        # output = torch.zeros((K, C, R, S), dtype=input.dtype, device=input.device)

        # for gemm_i in range(GEMM_M):
        #     k = gemm_i
        #     for gemm_j in range(GEMM_N):
        #         c = gemm_j // (R * S)
        #         crs_residual = gemm_j % (R * S)
        #         r = crs_residual // S
        #         s = crs_residual % S
        #         acc = 0.0
        #         for gemm_k in range(GEMM_K):
        #             n = gemm_k // (P * Q)
        #             npq_residual = gemm_k % (P * Q)
        #             p = npq_residual // Q
        #             q = npq_residual % Q
        #             h = p * str_h + r * dil_h - pad_h
        #             w = q * str_w + s * dil_w - pad_w
        #             if h < 0 or h >= H or w < 0 or w >= W:
        #                 continue
        #             acc += input[n, c, h, w] * doutput[n, k, p, q]
        #         output[k, c, r, s] = acc
        # return output


class _triton_conv2d_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: int = (1, 1),
                padding: int = (0, 0),
                dilation: int = (1, 1)
    ):
        output = _implicit_gemm_conv2d_fwd(input, weight, bias, stride, padding, dilation)

        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias_requires_grad = bias.requires_grad

        return output
    
    @staticmethod
    def backward(ctx, doutput: torch.Tensor):
        input, weight = ctx.saved_tensors
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        dinput = None
        if input.requires_grad:
            dinput = _implicit_gemm_conv2d_input_bwd(doutput, weight, N, C, H, W, K, R, S, stride, padding, dilation)

        dweight = None
        if weight.requires_grad:
            dweight = _implicit_gemm_conv2d_weight_bwd(doutput, input, N, C, H, W, K, R, S, stride, padding, dilation)

        dbias = None
        bias_requires_grad = ctx.bias_requires_grad
        if bias_requires_grad:
            dbias = torch.sum(doutput, (0, 2, 3))
        return dinput, dweight, dbias, None, None, None
    

triton_conv2d = _triton_conv2d_func.apply


# python -m pytest -s ld_triton/ops/convolution/triton_conv2d.py 
@pytest.mark.parametrize("N, C, H, W, K, R, S, stride, padding, dilation", 
                         [
                          (1, 1, 3, 3, 1, 2, 2, 1, 0, 1),
                        #   (2, 7, 8, 8, 5, 3, 3, 2, 2, 2),
                          ])
def test_conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float16}
    input = torch.randn(N, C, H, W, requires_grad=True, **factory_kwargs)
    weight = torch.randn(K, C, R, S, requires_grad=True, **factory_kwargs)
    bias = torch.randn(K, requires_grad=True, **factory_kwargs)
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
    doutput = torch.randn_like(output)
    output.backward(doutput)
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    dbias, bias.grad = bias.grad.clone(), None

    triton_output = triton_conv2d(input, weight, bias, stride, padding, dilation)
    triton_output.backward(doutput)
    triton_dinput, input.grad = input.grad.clone(), None
    triton_dweight, weight.grad = weight.grad.clone(), None
    # ig_dbias, bias.grad = bias.grad.clone(), None

    print(f'dweight: {dweight}')
    print(f'triton_dweight: {triton_dweight}')

    assert torch.allclose(output, triton_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dinput, triton_dinput, atol=1e-3, rtol=1e-3)
    # assert torch.allclose(dweight, ig_dweight, atol=1e-1, rtol=1e-1)
    # assert torch.allclose(dbias, ig_dbias, atol=1e-1, rtol=1e-1)
