

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_mse_fwd_kernel(
    loss_ptr, input_ptr, target_ptr,
    loss_row_stride, input_row_stride, target_row_stride, 
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + pid * input_row_stride  + col_offsets
    target_ptrs = target_ptr + pid * target_row_stride + col_offsets
    
    input = tl.load(input_ptrs, mask=mask, other=0.0)
    target = tl.load(target_ptrs, mask=mask, other=0.0)
    loss = tl.sum((input - target) * (input - target)) / n_cols

    loss_ptrs = loss_ptr + pid

    tl.store(loss_ptrs, loss, mask=pid < n_rows)


@triton.jit
def _triton_mse_bwd_kernel(
    grad_ptr, input_ptr, target_ptr, grad_output,
    grad_row_stride, input_row_stride, target_row_stride, 
    n_rows, n_cols, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    grad_ptrs = grad_ptr + pid * grad_row_stride + col_offsets
    input_ptrs = input_ptr + pid * input_row_stride  + col_offsets
    target_ptrs = target_ptr + pid * target_row_stride + col_offsets
    
    input = tl.load(input_ptrs, mask=mask, other=0.0)
    target = tl.load(target_ptrs, mask=mask, other=0.0)
    grad_ = (input - target) * 2 * grad_output / (n_rows * n_cols)

    tl.store(grad_ptrs, grad_, mask=mask)


class _triton_mse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, target: torch.Tensor):
        shape = input.shape
        device = input.device
        input = input.view(-1, shape[-1])
        target = target.view(-1, shape[-1])
        n_rows, n_cols = input.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        loss = torch.zeros(n_rows, device=device)

        _triton_mse_fwd_kernel[(n_rows, )](
            loss, input, target, 
            loss.stride()[0], input.stride()[0], target.stride()[0], 
            n_rows, n_cols, 
            BLOCK_SIZE
        )

        loss = loss.sum() / loss.numel()
        input = input.view(*shape)
        target = target.view(*shape)
        ctx.save_for_backward(input, target)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # input, target = ctx.saved_tensors
        # grad_input = 2 * (input - target) / input.numel()
        input, target, = ctx.saved_tensors
        shape = input.shape
        device = input.device
        input = input.view(-1, shape[-1])
        target = target.view(-1, shape[-1])
        n_rows, n_cols = input.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        grad_input, grad_target = None, None
        grad_output = grad_output.item()
        if input.requires_grad:
            grad_input = torch.empty(n_rows, n_cols, device=device)
            _triton_mse_bwd_kernel[(n_rows, )](
                grad_input, input, target, grad_output,
                grad_input.stride()[0], input.stride()[0], target.stride()[0],
                n_rows, n_cols,
                BLOCK_SIZE
            )
            grad_input = grad_input.view(*shape)

        if target.requires_grad:
            grad_target = torch.empty(n_rows, n_cols, device=device)
            _triton_mse_bwd_kernel[(n_rows, )](
                grad_target, target, input, grad_output,
                grad_target.stride()[0], target.stride()[0], input.stride()[0],
                n_rows, n_cols,
                BLOCK_SIZE
            )
            grad_target = grad_target.view(*shape)
        
        return grad_input, grad_target


triton_mse = _triton_mse.apply
