
import torch


class _naive_mse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, target: torch.Tensor):
        loss = (input - target).pow(2).sum() / input.numel()
        ctx.save_for_backward(input, target)
        return loss
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, target = ctx.saved_tensors
        grad_input, grad_target = None, None
        if input.requires_grad:
            grad_input = 2 * (input - target) / input.numel() * grad_output
        if target.requires_grad:
            grad_target = 2 * (target - input) / input.numel() * grad_output
        return grad_input, grad_target


naive_mse = _naive_mse.apply