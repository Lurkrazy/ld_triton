
import torch
import pytest
torch.nn.functional.conv2d


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
        # output = _naive_conv2d_nchw(input, weight, bias, stride, padding, dilation)
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        str_h, str_w = stride
        pad_h, pad_w = padding
        dil_h, dil_w = dilation

        P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
        Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

        output = torch.zeros((N, K, P, Q), dtype=input.dtype, device=input.device)

        for n in range(N):
            for k in range(K):
                for p in range(P):
                    for q in range(Q):
                        acc = 0.0
                        for c in range(C):
                            for r in range(R):
                                for s in range(S):
                                    h = p * str_h + r * dil_h - pad_h
                                    w = q * str_w + s * dil_w - pad_w
                                    if h < 0 or h >= H or w < 0 or w >= W:
                                        continue
                                    output[n, k, p, q] += weight[k, c, r, s] * input[n, c, h, w]
                        if bias is not None:
                            output[n, k, p, q] += bias[k]


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

        N, K, P, Q = dout.shape
        N, C, H, W = input.shape
        K, C, R, S = weight.shape
        
        dinput = None
        if input.requires_grad:
            dinput = torch.zeros_like(input)

            for n in range(N):
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            for k in range(K):
                                for r in range(R):
                                    for s in range(S):
                                        if (h + pad_h - r * dil_h) % str_h == 0 and (w + pad_w - s * dil_w) % str_w == 0:
                                            p = (h + pad_h - r * dil_h) // str_h
                                            q = (w + pad_w - s * dil_w) // str_w
                                            if p < 0 or p >= P or q < 0 or q >= Q:
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


# python -m pytest -s ld_triton/ops/convolution/naive_conv2d.py
@pytest.mark.parametrize("N, C, H, W, K, R, S, stride, padding, dilation", [(2, 7, 8, 8, 5, 3, 3, 1, 0, 1)])
def test_conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    input = torch.randn(N, C, H, W, requires_grad=True, device='cpu', dtype=torch.float16)
    weight = torch.randn(K, C, R, S, requires_grad=True, device='cpu', dtype=torch.float16)
    bias = torch.randn(K, requires_grad=True, device='cpu', dtype=torch.float16)
    print(f'input: {input.requires_grad }')

    stride = (stride, stride)
    padding = (padding, padding)
    dilation = (dilation, dilation)
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
