import torch
import triton
import triton.language as tl


@triton.jit
def _triton_sigmoid_fwd_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    sigmoid_output = 1.0 / (1.0 + tl.exp(0.0 - row))

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, sigmoid_output, mask=mask)


@triton.jit
def _triton_sigmoid_bwd_kernel(dx_ptr, x_ptr, dp_ptr, dx_row_stride, x_row_stride, dp_row_stride, n_rows, n_cols,
                           BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    x_start_ptr = x_ptr + row_idx * x_row_stride
    dp_start_ptr = dp_ptr + row_idx * dp_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_start_ptr + col_offsets
    dp_ptrs = dp_start_ptr + col_offsets
    mask = col_offsets < n_cols
    x_row = tl.load(x_ptrs, mask=mask, other=0)
    dp_row = tl.load(dp_ptrs, mask=mask, other=0)
    dx_row = dp_row * (tl.exp(0.0 - x_row) / (1.0 + tl.exp(0.0 - x_row) * tl.exp(0.0 - x_row)))

    dx_start_ptr = dx_ptr + row_idx * dx_row_stride
    dx_ptrs = dx_start_ptr + col_offsets
    tl.store(dx_ptrs, dx_row, mask=mask)


class _triton_sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, x.shape[-1])
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        p = torch.empty_like(x)
        _triton_sigmoid_fwd_kernel[(n_rows,)](
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
        ctx.save_for_backward(x)
        return p
    
    @staticmethod
    def backward(ctx, dp: torch.Tensor):
        x, = ctx.saved_tensors
        shape = x.shape
        x = x.view(-1, x.shape[-1])
        dp = dp.view(-1, dp.shape[-1])
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        dx = torch.empty_like(x)
        _triton_sigmoid_bwd_kernel[(n_rows,)](
            dx,
            x,
            dp,
            dx.stride(0),
            x.stride(0),
            dp.stride(0),
            n_rows,
            n_cols,
            BLOCK_SIZE,
        )
        x = x.view(*shape)
        dp = dp.view(*shape)
        return dx.view(*shape)
        

triton_sigmoid = _triton_sigmoid.apply