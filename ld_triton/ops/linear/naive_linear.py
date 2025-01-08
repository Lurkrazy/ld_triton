
import torch


class _naive_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        ctx.save_for_backward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias, = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        if input.requires_grad:
            grad_input = torch.matmul(grad_output, weight)
        if weight.requires_grad:
            grad_weight = torch.matmul(grad_output.t(), input)
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(0, keepdim=False)
        return grad_input, grad_weight, grad_bias
        

naive_linear = _naive_linear.apply