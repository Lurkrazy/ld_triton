import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _triton_rms_norm_fwd_kernel(
    rmsnorm_ptr, rrms_ptr, x_ptr, weight_ptr,
    n_rows, n_cols, eps,
    rmsnorm_row_stride, x_row_stride, weight_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    rmsnorm_ptrs = rmsnorm_ptr + row_idx * rmsnorm_row_stride + col_offsets
    x_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
    weight_ptrs = weight_ptr + col_offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    weight = tl.load(weight_ptrs, mask=mask, other=1.0)
    u = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    rmsnorm = x * u * weight
    
    tl.store(rmsnorm_ptrs, rmsnorm, mask=mask)
    tl.store(rrms_ptr + row_idx, u)

        
@triton.jit
def _triton_rms_norm_x_bwd_kernel(
    dx_ptr, x_ptr, dweight_ptr, weight_ptr, dout_ptr,
    n_rows, n_cols, eps,
    dx_row_stride, x_row_stride, dout_row_stride, weight_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dx_ptrs = dx_ptr + row_idx * dx_row_stride + col_offsets
    x_ptrs = x_ptr + row_idx * x_row_stride + col_offsets
    dout_ptrs = dout_ptr + row_idx * dout_row_stride + col_offsets
    weight_ptrs = weight_ptr + col_offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0)
    weight = tl.load(weight_ptrs, mask=mask, other=1.0)
    u = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    dx = dout * u * weight - (x / n_cols) * tl.sum(u * u * u * dout * x * weight, axis=0)

    tl.store(dx_ptrs, dx, mask=mask)


@triton.jit
def _triton_rms_norm_weight_bwd_kernel(
    dx_ptr, x_ptr, dweight_ptr, weight_ptr, dout_ptr, rrms_ptr,
    n_rows, n_cols, eps,
    dx_row_stride, x_row_stride, dout_row_stride, weight_row_stride,
    BLOCK_SIZE: tl.constexpr
):
    col_idx = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_rows

    dweight_ptrs = dweight_ptr + col_idx
    x_ptrs = x_ptr + row_offsets * x_row_stride + col_idx 
    dout_ptrs = dout_ptr + row_offsets * dout_row_stride + col_idx
    rrms_ptrs = rrms_ptr + row_offsets

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0)
    
    u = tl.load(rrms_ptrs, mask=mask, other=0.0)
    dweight =  tl.sum(dout * u * x, axis=0)

    tl.store(dweight_ptrs, dweight)

class _triton_rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,     
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        eps: float = 1e-6,
        recompute = False
    ) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, shape[-1])
        weight_shape = weight.shape
        weight = weight.view(-1, weight_shape[-1])

        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        rmsnorm = torch.zeros_like(x)
        rrms = torch.zeros_like(x[:, 0])
        
        _triton_rms_norm_fwd_kernel[(n_rows, )](
            rmsnorm, rrms, x, weight,
            n_rows, n_cols, eps,
            rmsnorm.stride(0), x.stride(0), weight.stride(0),
            BLOCK_SIZE,
        )
        # print(f'rrms: {rrms}')
        x = x.view(*shape)
        weight = weight.view(*weight_shape)
        rmsnorm = rmsnorm.view(*shape)
        ctx.save_for_backward(x, weight, rrms)
        ctx.eps = eps
        return rmsnorm
    
    @staticmethod
    def backward(ctx, dout: torch.Tensor, *args):
        x, weight, rrms = ctx.saved_tensors
        eps = ctx.eps
        shape = x.shape
        x = x.view(-1, shape[-1])
        dout = dout.view(-1, shape[-1])
        # weight = weight.view(-1, shape[-1])
        n_rows, n_cols = x.shape
        COL_BLOCK_SIZE = triton.next_power_of_2(n_cols)
        ROW_BLOCK_SIZE = triton.next_power_of_2(n_rows)

        dx = None
        dweight = None
        if x.requires_grad:
            dx = torch.zeros_like(x)
            _triton_rms_norm_x_bwd_kernel[(n_rows, )](
                dx, x, dweight, weight, dout, 
                n_rows, n_cols, eps,
                dx.stride(0), x.stride(0), dout.stride(0), weight.stride(0),
                COL_BLOCK_SIZE,
            )
        
        if weight.requires_grad:
            dweight = torch.zeros_like(weight)
            _triton_rms_norm_weight_bwd_kernel[(n_cols, )](
                dx, x, dweight, weight, dout, rrms,
                n_rows, n_cols, eps,
                dx.stride(0), x.stride(0), dout.stride(0), weight.stride(0),
                ROW_BLOCK_SIZE,
            )

        x = x.view(*shape)
        dx = dx.view(*shape)
        dout = dout.view(-1, shape[-1])
        return dx, dweight, None, None
    

triton_rms_norm = _triton_rms_norm.apply