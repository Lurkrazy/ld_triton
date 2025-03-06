
import torch
import triton
import triton.language as tl


@triton.jit
def triton_max_fwd(
    input_ptr, 
    output_val_ptr, 
    output_idx_ptr, 
    n_rows, 
    n_cols, 
    empty_val, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    input_offsets = pid * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    input = tl.load(input_ptr + input_offsets, mask=mask, other=empty_val)
    output_val, output_idx = tl.max(input, axis=0, return_indices=True)
    tl.store(output_val_ptr + pid, output_val)
    tl.store(output_idx_ptr + pid, output_idx)


class _triton_max(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        input: torch.Tensor, 
        dim, 
        keepdim
    ):
        assert dim == -1 or dim == input.shape[-1], 'Only last dim is supported'
        empty_val = torch.finfo(input.dtype).min
        shape = input.shape
        input = input.view(-1, shape[-1])
        n_rows, n_cols = input.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output_val = torch.empty(n_rows, device=input.device, dtype=input.dtype)
        output_idx = torch.empty(n_rows, device=input.device, dtype=torch.int64)
        grid = (n_rows, )
        triton_max_fwd[grid](
            input, 
            output_val, 
            output_idx, 
            n_rows, 
            n_cols, 
            empty_val, 
            BLOCK_SIZE
        )
        ctx.save_for_backward(output_idx)
        ctx.input_shape = shape
        ctx.dim = dim
        ctx.keepdim = keepdim

        if keepdim:
            return output_val.view((*shape[:-1], 1)), output_idx.view((*shape[:-1], 1))
        else:
            return output_val.view(*shape[:-1]), output_idx.view(*shape[:-1])
    
    @staticmethod
    def backward(ctx, dp, *args):
        indices, = ctx.saved_tensors
        input_shape = ctx.input_shape
        dx = torch.zeros(input_shape, device=dp.device)
        dx = dx.view(-1, input_shape[-1])
        indices = indices.view(-1)
        dx[torch.arange(dx.shape[0]), indices] = dp.view(-1)
        dx = dx.view(*input_shape)
        return dx, None, None

triton_max = _triton_max.apply


if __name__ == "__main__":
    M, N, K = 2, 5, 7
    keepdim = False
    keepdim = True

    input = torch.randn(M, N, K, device='cuda', requires_grad=True)
    val, idx = torch.max(input, dim=-1, keepdim=keepdim)
    dy = torch.randn_like(val)
    val.backward(dy)
    dinput, input.grad = input.grad.clone(), None

    triton_val, triton_idx = triton_max(input, -1, keepdim)
    triton_val.backward(dy)
    triton_dinput, input.grad = input.grad.clone(), None

    assert torch.allclose(val, triton_val)
    assert torch.allclose(idx, triton_idx)
    assert torch.allclose(dinput, triton_dinput)


