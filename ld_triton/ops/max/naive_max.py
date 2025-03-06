
import torch


class _naive_max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, dim, keepdim):
        assert dim == -1 or dim == input.shape[-1], 'Only last dim is supported'
        ret = input.max(dim=dim, keepdim=keepdim)
        ctx.save_for_backward(ret.indices)
        ctx.input_shape = input.shape
        ctx.dim = dim
        ctx.keepdim = keepdim
        return ret.values, ret.indices

    @staticmethod
    def backward(ctx, dp: torch.Tensor, *args):
        indices, = ctx.saved_tensors
        input_shape = ctx.input_shape
        dx = torch.zeros(input_shape, device=dp.device)
        dx = dx.view(-1, input_shape[-1])
        indices = indices.view(-1)
        dx[torch.arange(dx.shape[0]), indices] = dp.view(-1)
        dx = dx.view(*input_shape)
        return dx, None, None


naive_max = _naive_max.apply


if __name__ == "__main__":
    M, N = 2, 5
    x = torch.randn(M, N, 3, requires_grad=True, device='cuda')
    keepdim = True
    y = torch.max(x, dim=-1, keepdim=keepdim)
    dy = torch.randn_like(y.values)
    y.values.backward(dy)
    dx, x.grad = x.grad.clone(), None

    naive_y_val, naive_y_idx = naive_max(x, -1, keepdim)
    naive_y_val.backward(dy)
    naive_dx, x.grad = x.grad.clone(), None
    
    assert torch.allclose(y.values, naive_y_val)
    assert torch.allclose(dx, naive_dx)