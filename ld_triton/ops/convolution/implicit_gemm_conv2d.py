

import torch
import pytest


def _implicit_gemm_conv2d_fwd(input, weight, bias, stride, padding, dilation):
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

        output = torch.zeros((N, K, P, Q), dtype=input.dtype, device=input.device)

        for gemm_i in range(GEMM_M):
            n = gemm_i // (P * Q)
            npq_residual = gemm_i % (P * Q)
            p = npq_residual // Q
            q = npq_residual % Q
            for gemm_j in range(GEMM_N):
                k = gemm_j
                acc = 0.0
                for gemm_k in range(GEMM_K):
                    c = gemm_k // (R * S)
                    crs_residual = gemm_k % (R * S)
                    r = crs_residual // S
                    s = crs_residual % S

                    h = p * str_h + r * dil_h - pad_h
                    w = q * str_w + s * dil_w - pad_w
                    if h < 0 or h >= H or w < 0 or w >= W:
                        continue
                    input_val = input[n, c, h, w]
                    weight_val = weight[k, c, r, s]

                    acc += input_val * weight_val
                if bias is not None:
                    output[n, k, p, q] = acc + bias[k]
                else:
                    output[n, k, p, q] = acc
        return output


def _implicit_gemm_conv2d_input_bwd(doutput, weight, N, C, H, W, K, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * H * W
    GEMM_N = C
    GEMM_K = K * R * S

    output = torch.zeros((N, C, H, W), dtype=doutput.dtype, device=doutput.device)
    
    for gemm_i in range(GEMM_M):
        n = gemm_i // (H * W)
        nhw_residual = gemm_i % (H * W)
        h = nhw_residual // W
        w = nhw_residual % W
        for gemm_j in range(GEMM_N):
            c = gemm_j
            acc = 0.0
            for gemm_k in range(GEMM_K):
                k = gemm_k // (R * S)
                krs_residual = gemm_k % (R * S)
                r = krs_residual // S
                s = krs_residual % S

                if (h + pad_h - r * dil_h) % str_h == 0 and (w + pad_w - s * dil_w) % str_w == 0:
                    p = (h + pad_h - r * dil_h) // str_h
                    q = (w + pad_w - s * dil_w) // str_w
                    if p < 0 or p >= P or q < 0 or q >= Q:
                        continue
                    acc += doutput[n, k, p, q] * weight[k, c, r, s]
            output[n, c, h, w] = acc

    return output


def _implicit_gemm_conv2d_weight_bwd(doutput, input, N, C, H, W, K, R, S, stride, padding, dilation):
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        GEMM_M = K
        GEMM_N = C * R * S
        GEMM_K = N * P * Q

        output = torch.zeros((K, C, R, S), dtype=input.dtype, device=input.device)

        for gemm_i in range(GEMM_M):
            k = gemm_i
            for gemm_j in range(GEMM_N):
                c = gemm_j // (R * S)
                crs_residual = gemm_j % (R * S)
                r = crs_residual // S
                s = crs_residual % S
                acc = 0.0
                for gemm_k in range(GEMM_K):
                    n = gemm_k // (P * Q)
                    npq_residual = gemm_k % (P * Q)
                    p = npq_residual // Q
                    q = npq_residual % Q
                    h = p * str_h + r * dil_h - pad_h
                    w = q * str_w + s * dil_w - pad_w
                    if h < 0 or h >= H or w < 0 or w >= W:
                        continue
                    acc += input[n, c, h, w] * doutput[n, k, p, q]
                output[k, c, r, s] = acc
        return output


class _implicit_gemm_conv2d_func(torch.autograd.Function):
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
    

implicit_gemm_conv2d = _implicit_gemm_conv2d_func.apply


# python -m pytest -s ld_triton/ops/convolution/implicit_gemm_conv2d.py
@pytest.mark.parametrize("N, C, H, W, K, R, S, stride, padding, dilation", 
                         [
                          (1, 1, 3, 3, 1, 2, 2, 1, 0, 1),
                          (2, 7, 8, 8, 5, 3, 3, 2, 2, 2),
                          ])
def test_conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    factory_kwargs = {'device': 'cpu', 'dtype': torch.float16}
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

    ig_output = implicit_gemm_conv2d(input, weight, bias, stride, padding, dilation)
    ig_output.backward(doutput)
    ig_dinput, input.grad = input.grad.clone(), None
    ig_dweight, weight.grad = weight.grad.clone(), None
    ig_dbias, bias.grad = bias.grad.clone(), None

    assert torch.allclose(output, ig_output, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dinput, ig_dinput, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dweight, ig_dweight, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dbias, ig_dbias, atol=1e-1, rtol=1e-1)