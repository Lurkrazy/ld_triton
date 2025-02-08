
import torch


class _naive_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ret = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(x)
        return ret

    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        x, = ctx.saved_tensors
        dx = dp * (torch.exp(-x) / (1 + torch.exp(-x)) ** 2)
        return dx


naive_sigmoid = _naive_sigmoid.apply