import torch
import triton

from ld_triton.ops.mse.naive_mse import naive_mse
from ld_triton.ops.mse.triton_mse import triton_mse


configs = []


configs.append(
    triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="provider", 
        line_vals=['cublas', "naive", "triton"], 
        line_names=['cuBLAS', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={},
    ))


@triton.testing.perf_report(configs)
def benchmark(M, N, provider):
    input = torch.randn(M, N, requires_grad=True, device='cuda')
    target = torch.randn(M, N, requires_grad=True, device='cuda')
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.mse_loss(input, target), quantiles=quantiles)
    if provider == 'naive':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_mse(input, target), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_mse(input, target), quantiles=quantiles)
    perf = lambda ms: (M * N + M * N  + M * N + 1) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)