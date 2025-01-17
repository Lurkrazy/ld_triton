
import torch
import pytest

def im2col_fwd(input, N, C, H, W, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * P * Q
    GEMM_K = C * R * S

    im2col_input = torch.zeros((GEMM_M, GEMM_K), dtype=input.dtype, device=input.device)

    for gemm_i in range(GEMM_M):
        n = gemm_i // (P * Q)
        npq_residual = gemm_i % (P * Q)
        p = npq_residual // Q
        q = npq_residual % Q
        for gemm_k in range(GEMM_K):
            c = gemm_k // (R * S)
            crs_residual = gemm_k % (R * S)
            r = crs_residual // S
            s = crs_residual % S

            h = p * str_h + r * dil_h - pad_h
            w = q * str_w + s * dil_w - pad_w
            if h < 0 or h >= H or w < 0 or w >= W:
                continue
            im2col_input[gemm_i, gemm_k] = input[n, c, h, w]

    return im2col_input


def im2col_input_bwd(doutput, N, C, H, W, K, R, S, stride, padding, dilation):
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * H * W
    GEMM_K = K * R * S

    im2col_doutput = torch.zeros((GEMM_M, GEMM_K), dtype=doutput.dtype, device=doutput.device)
    
    for gemm_i in range(GEMM_M):
        n = gemm_i // (H * W)
        nhw_residual = gemm_i % (H * W)
        h = nhw_residual // W
        w = nhw_residual % W
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
                im2col_doutput[gemm_i, gemm_k] = doutput[n, k, p, q]

    return im2col_doutput


class _im2col_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1
    ):
        assert input.is_contiguous()
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        im2col_input = im2col_fwd(input, N, C, H, W, R, S, (str_h, str_w), (pad_h, pad_w), (dil_h, dil_w))
        im2col_weight = weight.view(K, -1).t()

        output = im2col_input @ im2col_weight
        output = output.view(N, P, Q, K).permute(0, 3, 1, 2).contiguous() + bias.view(1, -1, 1, 1)

        ctx.save_for_backward(input, weight)
        ctx.N = N
        ctx.C = C
        ctx.H = H
        ctx.W = W
        ctx.P = P
        ctx.Q = Q
        ctx.K = K
        ctx.R = R
        ctx.S = S
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.bias_requires_grad = bias.requires_grad

        return output
    
    @staticmethod
    def backward(ctx, doutput: torch.Tensor):
        input, weight = ctx.saved_tensors

        N, C, H, W = ctx.N, ctx.C, ctx.H, ctx.W
        P = ctx.P
        Q = ctx.Q
        K = ctx.K
        R, S = ctx.R, ctx.S
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        
        dinput = None
        if input.requires_grad:
            im2col_doutput = im2col_input_bwd(doutput, N, C, H, W, K, R, S, stride, padding, dilation) # [N, K, P, Q] -> [N*H*W, K*R*S]
            im2col_weight = weight.permute(1, 0, 2, 3).reshape(C, -1).contiguous() # [K, C, R, S] -> [C, K*R*S]

            im2col_dinput = im2col_doutput @ im2col_weight.t() # M=N*H*W, N=C, K=K*R*S

            dinput = im2col_dinput.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        dweight = None
        if weight.requires_grad:
            im2col_input = im2col_fwd(input, N, C, H, W, R, S, stride, padding, dilation) # [N, C, H, W] -> [N*P*Q, C*R*S]
            im2col_doutput = doutput.permute(0, 2, 3, 1).contiguous().view(-1, K) # [N, P, Q, K] -> [N*P*Q, K]
            im2col_dweight = im2col_doutput.t() @ im2col_input # M=K, N=C*R*S, K=N*P*Q
            dweight = im2col_dweight.view(K, C, R, S)

        dbias = None
        bias_requires_grad = ctx.bias_requires_grad
        if bias_requires_grad:
            dbias = torch.sum(doutput, (0, 2, 3))
        return dinput, dweight, dbias, None, None, None
    

im2col_conv2d = _im2col_conv2d.apply


# python -m pytest -s ld_triton/ops/convolution/im2col_conv2d.py
@pytest.mark.parametrize("N, C, H, W, K, R, S, stride, padding, dilation", 
                         [
                          (1, 1, 3, 3, 1, 2, 2, 1, 0, 1),
                          (2, 7, 8, 8, 5, 3, 3, 2, 2, 2),
                          ])
def test_conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    input = torch.randn(N, C, H, W, requires_grad=True, device='cpu', dtype=torch.float16)
    weight = torch.randn(K, C, R, S, requires_grad=True, device='cpu', dtype=torch.float16)
    bias = torch.randn(K, requires_grad=True, device='cpu', dtype=torch.float16)
    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
    doutput = torch.randn_like(output)
    output.backward(doutput)
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    dbias, bias.grad = bias.grad.clone(), None

    im2col_output = im2col_conv2d(input, weight, bias, stride, padding, dilation)
    im2col_output.backward(doutput)
    im2col_dinput, input.grad = input.grad.clone(), None
    im2col_dweight, weight.grad = weight.grad.clone(), None
    im2col_dbias, bias.grad = bias.grad.clone(), None

    assert torch.allclose(output, im2col_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dinput, im2col_dinput, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dweight, im2col_dweight, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dbias, im2col_dbias, atol=1e-1, rtol=1e-1)
