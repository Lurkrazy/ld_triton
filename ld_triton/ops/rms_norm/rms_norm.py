import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numbers
import triton
import triton.language as tl
from typing import Union, List, Optional, Tuple


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


class _ld_rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,     
        x: torch.Tensor, 
        weight: Optional[torch.Tensor] = None, 
        eps: float = 1e-6,
        recompute = False
    ) -> torch.Tensor:
        pass
    
    @staticmethod
    def backward(ctx, doutput: torch.Tensor):
        pass

# Adapted from https://github.com/meta-llama/llama/blob/main/llama/model.py
# only replace forward
class NaiveRMSNorm(torch.nn.Module):
    def __init__(self, dim, weight: torch.tensor=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return naive_rms_norm(x, self.weight, self.eps, False)

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}'.format(**self.__dict__)


@triton.jit
def _ld_rms_norm_fwd_kernel(
    rmsnorm_ptr, x_ptr, weight_ptr,
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

        
@triton.jit
def _ld_rms_norm_x_bwd_kernel(
    dx_ptr, x_ptr, weight_ptr, dweight_ptr, dout_ptr,
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
    dx = dout * u - (x / n_cols) * tl.sum(u * u * u * dout * x * weight, axis=0)

    tl.store(dx_ptrs, dx, mask=mask)


@triton.jit
def _ld_rms_norm_weight_bwd_kernel(
    dx_ptr, x_ptr, dweight_ptr, weight_ptr, dout_ptr,
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

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=mask, other=0.0)
    
    u = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    dweight =  tl.sum(dout * u * x, axis=0)

    tl.store(dweight_ptrs, dweight)

class _ld_rms_norm(torch.autograd.Function):
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
        _ld_rms_norm_fwd_kernel[(n_rows, )](
            rmsnorm, x, weight,
            n_rows, n_cols, eps,
            rmsnorm.stride(0), x.stride(0), weight.stride(0),
            BLOCK_SIZE,
        )
        x = x.view(*shape)
        weight = weight.view(*weight_shape)
        rmsnorm = rmsnorm.view(*shape)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return rmsnorm
    
    @staticmethod
    def backward(ctx, drmsnorm: torch.Tensor, *args):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        shape = x.shape
        x = x.view(-1, shape[-1])
        drmsnorm = drmsnorm.view(-1, shape[-1])
        # weight = weight.view(-1, shape[-1])
        n_rows, n_cols = x.shape
        COL_BLOCK_SIZE = triton.next_power_of_2(n_cols)
        ROW_BLOCK_SIZE = triton.next_power_of_2(n_rows)

        dx = torch.zeros_like(x)
        dweight = torch.zeros_like(weight)

        _ld_rms_norm_x_bwd_kernel[(n_rows, )](
            dx, x, dweight, weight, drmsnorm, 
            n_rows, n_cols, eps,
            dx.stride(0), x.stride(0), drmsnorm.stride(0), weight.stride(0),
            COL_BLOCK_SIZE,
        )

        _ld_rms_norm_weight_bwd_kernel[(n_cols, )](
            dx, x, dweight, weight, drmsnorm, 
            n_rows, n_cols, eps,
            dx.stride(0), x.stride(0), drmsnorm.stride(0), weight.stride(0),
            ROW_BLOCK_SIZE,
        )

        x = x.view(*shape)
        dx = dx.view(*shape)
        drmsnorm = drmsnorm.view(-1, shape[-1])
        return dx, dweight, None, None
    

ld_rms_norm = _ld_rms_norm.apply


class LDRMSNorm(torch.nn.Module):
    def __init__(self, dim, weight: torch.tensor=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim).cuda())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ld_rms_norm(x, self.weight, self.eps, False)

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}'.format(**self.__dict__)
    

class LLAMARMSNorm(torch.nn.Module):
    def __init__(self, dim, weight: torch.tensor=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim).cuda())

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        return output


# python -m pytest -s rmsnorm.py -k test_2d_RMSNorm
@pytest.mark.parametrize('M, N', [(512, 513)])
def test_2d_RMSNorm(M, N):
    eps = 1e-6
    llama_m = LLAMARMSNorm(N, eps=eps)
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)

    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None
    llama_dweight = llama_m.weight.grad

    normalized_shape = [N]
    m = torch.nn.RMSNorm(normalized_shape, elementwise_affine=True, eps=eps).cuda()
    y = m(x)
    loss = loss_fn(y, target)
    loss.backward()
    dx, x.grad = x.grad.clone(), None
    dweight = m.weight.grad.clone()

    naive_m = NaiveRMSNorm(N, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None
    naive_dweight = naive_m.weight.grad.clone()

    ld_m = LDRMSNorm(N, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None
    ld_dweight = ld_m.weight.grad.clone()

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_y, ld_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_dx, ld_dx, atol=1e-3, rtol=1e-3)
 
    assert torch.allclose(llama_dweight, dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_dweight, dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dweight, dweight, atol=1e-3, rtol=1e-3)
    print(f'llama_dweight: {dweight.shape}')
    print(f'naive_dweight: {ld_dweight.shape}')
# python -m pytest -s rmsnorm.py -k test_3d_RMSNorm
@pytest.mark.parametrize('B, N, D', [(3, 781, 129)])
def test_3d_RMSNorm(B, N, D):
    eps = 1e-6
    llama_m = LLAMARMSNorm(D, eps=eps)
    x = torch.randn(B, N, D, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)
    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None

    naive_m = NaiveRMSNorm(D, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None

    ld_m = LDRMSNorm(D, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_y, naive_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dx, naive_dx, atol=1e-3, rtol=1e-3)


# python -m pytest -s rmsnorm.py -k test_4d_RMSNorm
@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_RMSNorm(B, N, H, D):
    eps = 1e-6
    llama_m = LLAMARMSNorm(D, eps=eps)
    x = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)
    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None

    naive_m = NaiveRMSNorm(D, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None

    ld_m = LDRMSNorm(D, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_y, naive_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dx, naive_dx, atol=1e-3, rtol=1e-3)


# python -m pytest -s rmsnorm.py -k test_rms_norm
# torch.rms_norm is not correct
@pytest.mark.parametrize('M, N', [(1, 2)])
def test_rms_norm(M, N, device='cuda', dtype=torch.float):
    input = torch.ones(M, N, device=device, dtype=dtype, requires_grad=True)
    input.grad = None
    weight = torch.ones(N, device=device, dtype=dtype, requires_grad=True)
    output = F.rms_norm(input, [N], weight=weight)
    doutput = torch.randn_like(output)
    output.backward(output)
    dinput, input.grad = input.grad.clone(),None
    print(f'dinput: {dinput.dtype}')
    print(dinput)

    naive_output = naive_rms_norm(input, weight)
    naive_output.backward(doutput)
    naive_dinput, input.grad = input.grad.clone(),None
    print(f'naive_dinput: {naive_dinput.dtype}')
    print(naive_dinput)
    
    assert torch.allclose(output, naive_output, atol=1e-3, rtol=1e-3)


perf_configs = []


@triton.testing.perf_report(perf_configs)
def benchmark(M, N, mode, provider, device='cuda'):
    eps = 1e-6
    dim = N
    weight = torch.nn.Parameter(torch.ones(dim)).cuda()
    x = torch.randn(M, N, device=device, requires_grad=True)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'llama':
        llama_m = torch.nn.RMSNorm([dim], eps=eps).cuda()
        fn = lambda: llama_m(x)
        if mode == 'bwd':
            # o =fn()
            # do = torch.randn_like(o)
            # fn = lambda: o.backward()
            pass
        ms = triton.testing.do_bench(fn)
    if provider == 'triton':
        fn = lambda: ld_rms_norm(x, weight, eps, False)
        if mode == 'bwd':
            o =fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == 'naive':
        fn = lambda: naive_rms_norm(x, weight, eps, False)
        if mode == 'bwd':
            o =fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if mode == 'fwd':
        gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e3)
    if mode == 'bwd':
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e3)
    return gbps(ms)


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true')
    args.add_argument('--bench', action='store_true')
    args.add_argument('--plot', action='store_true')
    args.add_argument('--mode', choices=('fwd', 'bwd'), default='fwd')
    args = args.parse_args()
    test = args.test
    bench = args.bench
    plot = args.plot
    print_data = bench
    mode = args.mode
    # # for mode in ['fwd', 'bwd']:
    if mode == 'fwd':
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[128 * i for i in range(2, 10)],
                line_arg='provider',
                line_vals=['triton', 'llama', 'naive'],
                line_names=['Triton', 'Llama', 'naive'],
                styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
                ylabel='GB/s',
                plot_name="softmax-performance",
                args={
                    'M': 4096,
                    'mode': 'fwd',
                },
            )
        )
    if mode == 'bwd':
        perf_configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[128 * i for i in range(2, 10)],
                line_arg='provider',
                line_vals=['triton', 'llama', 'naive'],
                line_names=['Triton', 'Llama', 'naive'],
                styles=[('blue', '-'), ('green', '-'), ('red', '-')],
                ylabel='GB/s',
                plot_name="softmax-performance",
                args={
                    'M': 4096,
                    'mode': 'bwd',
                },
            )
        )

    if bench:
        benchmark.run(show_plots=plot, print_data=print_data)

