
import torch
from typing import Union, List, Optional


_shape_t = Union[int, List[int], torch.Size]
    

class _naive_rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,     
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        eps: float = 1e-6,
        recompute = False
    ) -> torch.Tensor:
        shape = x.shape
        # x = x.view(-1, shape[-1])
        rmsnorm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            rmsnorm = rmsnorm * weight
        # x = x.view(*shape)
        # rmsnorm = rmsnorm.view(*shape)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rmsnorm
    
    @staticmethod
    def backward(ctx, doutput: torch.Tensor):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        shape = x.shape
        weight_shape = weight.shape
        x = x.view(-1, shape[-1])
        weight = weight.view(-1, weight_shape[-1])
        doutput = doutput.view(-1, shape[-1])
        N = x.shape[-1]
        u = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        if weight is not None:
            dx = doutput * u * weight - (x / float(N)) * torch.sum(u.pow(3) * doutput * x * weight, dim=-1, keepdim=True)
        else:
            dx = doutput * u - (x / float(N)) * torch.sum(u.pow(3) * doutput * x, dim=-1, keepdim=True)

        dweight = None
        if weight is not None:
            dweight = torch.sum(doutput * u * x, dim=0)
        x = x.view(*shape)
        weight = weight.view(*weight_shape)
        dx = dx.view(*shape)
        doutput = doutput.view(*shape)
        return dx, dweight, None, None
    

naive_rms_norm = _naive_rms_norm.apply