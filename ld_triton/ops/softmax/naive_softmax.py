
import torch


class _naive_softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # read  MN elements ; write M  elements
        x_max = x.max(dim=-1, keepdim=True).values
        # read MN + M elements ; write MN elements
        x = x - x_max
        # read  MN elements ; write MN elements
        numerator = torch.exp(x)
        # read  MN elements ; write M  elements
        denominator = numerator.sum(dim=-1, keepdim=True)
        # read MN + M elements ; write MN elements
        ret = numerator / denominator
        # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        p, = ctx.saved_tensors
        ds = torch.zeros_like(p)
        # ds = p * (dp - (p * dp).sum(dim=-1, keepdim=True))
        # read  2MN elements ; write MN elements
        ds = p * dp
        # read  MN elements ; write M elements
        ds = torch.sum(ds, dim=-1, keepdim=True)
        # read  MN + M elements ; write MN elements
        ds = dp - ds
        # read  2MN elements ; write MN elements
        ds = p * ds
        # in total: read 6MN + M elements ; wrote 3MN + M elements
        return ds


naive_softmax = _naive_softmax.apply