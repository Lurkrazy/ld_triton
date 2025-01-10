
import torch
import pytest
torch.nn.functional.conv2d


def _naive_conv2d_nchw(input, weight, bias, stride, padding, dilation):
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


class _naive_conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input: torch.Tensor,
                weight: torch.Tensor,
                bias: torch.Tensor = None,
                stride: int = (1, 1),
                padding: int = (0, 0),
                dilation: int = (1, 1)
    ):
        assert input.is_contiguous()
        output = _naive_conv2d_nchw(input, weight, bias, stride, padding, dilation)
        ctx.save_for_backward(input, weight)
        ctx.bias_requires_grad = bias.requires_grad
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        return output
    
    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        input, weight = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        print(f'stride: {stride}')
        N, K, P, Q = dout.shape
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        
        dinput = None
        if input.requires_grad:
            dinput = torch.zeros_like(input)

            for n in range(N):
                for k in range(K):
                    for p in range(P):
                        for q in range(Q):
                            for r in range(R):
                                for s in range(S):
                                    for c in range(C):
                                        h = p * str_h + r * dil_h - pad_h
                                        w = q * str_w + s * dil_w - pad_w
                                        if h < 0 or h >= H or w < 0 or w >= W:
                                            continue
                                        dinput[n, c, h, w] += weight[k, c, r, s] * dout[n, k, p, q]

        dweight = None
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)

            for k in range(K):
                for c in range(C):
                    for r in range(R):
                        for s in range(S):
                            for n in range(N):
                                for p in range(P):
                                    for q in range(Q):
                                        h = p * str_h + r * dil_h - pad_h
                                        w = q * str_w + s * dil_w - pad_w
                                        if h < 0 or h >= H or w < 0 or w >= W:
                                            continue
                                        dweight[k, c, r, s] += input[n, c, h, w] * dout[n, k, p, q]

        dbias = None
        bias_requires_grad = ctx.bias_requires_grad
        if bias_requires_grad:
            dbias = torch.sum(dout, (0, 2, 3))
        return dinput, dweight, dbias, None, None, None


naive_conv2d = _naive_conv2d.apply


@pytest.mark.parametrize("N, C, H, W, K, R, S", [(2, 7, 8, 8, 5, 3, 3)])
def test_conv2d(N, C, H, W, K, R, S):
    input = torch.randn(N, C, H, W, requires_grad=True, device='cpu', dtype=torch.float16)
    weight = torch.randn(K, C, R, S, requires_grad=True, device='cpu', dtype=torch.float16)
    bias = torch.randn(K, requires_grad=True, device='cpu', dtype=torch.float16)
    print(f'input: {input.requires_grad }')

    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    output = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation)
    doutput = torch.randn_like(output)
    output.backward(doutput)
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    dbias, bias.grad = bias.grad.clone(), None

    naive_output = naive_conv2d(input, weight, bias, stride, padding, dilation)
    naive_output.backward(doutput)
    naive_dinput, input.grad  = input.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None
    naive_dbias, bias.grad = bias.grad.clone(), None
    
    assert torch.allclose(output, naive_output, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dinput, naive_dinput, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dweight, naive_dweight, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dbias, naive_dbias, atol=1e-1, rtol=1e-1)
