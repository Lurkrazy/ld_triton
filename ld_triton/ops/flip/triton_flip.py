
import torch
import triton
import triton.language as tl


@triton.jit
def triton_flip_kernel(input_ptr, output_ptr, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    m = tl.program_id(0)
    load_offsets = m * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols
    input = tl.load(input_ptr + load_offsets, mask=mask)

    store_offsets = m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output = tl.flip(input, dim=0)
    tl.store(output_ptr + store_offsets, output)
    

class _triton_flip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
        input: torch.Tensor, 
        dim: int
    ):
        # only support dim = -1
        shape = input.shape
        assert dim == -1 or dim == len(shape) - 1
        input = input.view(-1, shape[-1])
        n_rows, n_cols = input.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        output = torch.empty((n_rows, BLOCK_SIZE), device=input.device, dtype=input.dtype)
        grid = (n_rows, BLOCK_SIZE, )
        triton_flip_kernel[grid](
            input, 
            output, 
            n_rows, 
            n_cols,
            BLOCK_SIZE
        )
        return output[:, (BLOCK_SIZE-n_cols):].view(*shape)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        shape = grad_output.shape
        grad_output = grad_output.view(-1, shape[-1])
        n_rows, n_cols = grad_output.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grad_input = torch.empty((n_rows, BLOCK_SIZE), device=grad_output.device, dtype=grad_output.dtype)
        grid = (n_rows, BLOCK_SIZE, )
        triton_flip_kernel[grid](
            grad_output, 
            grad_input, 
            n_rows, 
            n_cols,
            BLOCK_SIZE
        )
        return grad_input[:, (BLOCK_SIZE-n_cols):].view(*shape), None


triton_flip = _triton_flip.apply


if __name__ == "__main__":
    x = torch.randn(10, 5, 3, requires_grad=True, device='cuda')
    y = x.flip(-1)
    dy = torch.randn_like(y)
    y.backward(dy)
    dx, x.grad = x.grad.clone(), None
    
    triton_y = triton_flip(x, 2)
    triton_y.backward(dy)
    triton_dx, x.grad = x.grad.clone(), None
    
    assert torch.allclose(triton_y, y)
    assert torch.allclose(dx, triton_dx)