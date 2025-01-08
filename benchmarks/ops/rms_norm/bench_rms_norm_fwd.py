
import torch
import triton

from ld_triton.ops.rms_norm.naive_rms_norm import naive_rms_norm
from ld_triton.ops.rms_norm.triton_rms_norm import triton_rms_norm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 32)],
        line_arg='provider',
        line_vals=['cublas', "naive", "triton"], 
        line_names=['cuBLAS', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel='GB/s',
        plot_name='rms-norm-fwd',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'fwd'},
    ))
def bench_rms_norm(M, N, dtype, provider, mode='backward', eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    
    def y_fwd():
        if provider == "cublas":
            return torch.nn.functional.rms_norm(x, w_shape, weight=weight, eps=eps)

        if provider == "naive":
            return naive_rms_norm(x, weight, eps)

        if provider == "triton":
            return triton_rms_norm(x, weight, eps)
        
    gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_rms_norm.run(show_plots=True, print_data=True)