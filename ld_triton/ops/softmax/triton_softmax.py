
# Adapted from https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py
import torch
import triton
import triton.language as tl


class _naive_softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        x_max = x.max(dim=-1, keepdim=True).values
        x = x - x_max
        numerator = torch.exp(x)
        denominator = numerator.sum(dim=-1, keepdim=True)
        ret = numerator / denominator
        ctx.save_for_backward(ret)
        return ret

    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        p, = ctx.saved_tensors
        ds = torch.zeros_like(p)
        for i in range(p.shape[0]):
            ds[i] =  p[i]*(dp[i] - (p[i] * dp[i]).sum(dim=-1, keepdim=True))
        return ds


naive_softmax = _naive_softmax.apply


@triton.jit
def _triton_softmax_fwd_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def _triton_softmax_bwd_kernel(ds_ptr, p_ptr, dp_ptr, ds_row_stride, p_row_stride, dp_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    p_start_ptr = p_ptr + row_idx * p_row_stride
    dp_start_ptr = dp_ptr + row_idx * dp_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    p_ptrs = p_start_ptr + col_offsets
    dp_ptrs = dp_start_ptr + col_offsets
    mask = col_offsets < n_cols
    p_row = tl.load(p_ptrs, mask=mask, other=0)
    dp_row = tl.load(dp_ptrs, mask=mask, other=0)
    ds_row = p_row * (dp_row - tl.sum(p_row * dp_row, axis=0))

    ds_start_ptr = ds_ptr + row_idx * ds_row_stride
    ds_ptrs = ds_start_ptr + col_offsets
    tl.store(ds_ptrs, ds_row, mask=mask)


class _triton_softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        p = torch.empty_like(x)
        _triton_softmax_fwd_kernel[(n_rows,)](
            p,
            x,
            x.stride(0),
            p.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
        )
        p = p.view(*shape)
        x = x.view(*shape)
        ctx.save_for_backward(p)
        return p
    
    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        p, = ctx.saved_tensors
        shape = p.shape
        p = p.view(-1, p.shape[-1])
        dp = dp.view(-1, dp.shape[-1])
        n_rows, n_cols = p.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        ds = torch.empty_like(p)
        _triton_softmax_bwd_kernel[(n_rows,)](
            ds,
            p,
            dp,
            ds.stride(0),
            p.stride(0),
            dp.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
        )
        p = p.view(*shape)
        dp = dp.view(*shape)
        return ds.view(*shape)
        

triton_softmax = _triton_softmax.apply