
import torch
import triton

from ld_triton.ops.rms_norm.naive_rms_norm import naive_rms_norm
from ld_triton.ops.rms_norm.triton_rms_norm import triton_rms_norm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 8)],
        line_arg='provider',
        line_vals=['cublas', "naive", "triton"], 
        line_names=['cuBLAS', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel='GB/s',
        plot_name='rms-norm-backward',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'bwd'},
    ))
def bench_rms_norm(M, N, dtype, provider, mode='bwd', eps=1e-5, device='cuda'):
    # create data
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]

    loss_fn = torch.nn.MSELoss()
    target = torch.randn_like(x)

    def y_fwd():
        if provider == "cublas":
            return torch.nn.functional.rms_norm(x, w_shape, weight=weight, eps=eps)

        if provider == "naive":
            return naive_rms_norm(x, weight, eps)

        if provider == "triton":
            return triton_rms_norm(x, weight, eps)

    y = y_fwd()
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6  # noqa: F811, E704
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles,
                                                    grad_to_none=[x], rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


bench_rms_norm.run(show_plots=True, print_data=True)