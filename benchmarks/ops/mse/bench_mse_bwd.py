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
    input = torch.randn(M, N, device='cuda', requires_grad=True)
    target = torch.randn(M, N, device='cuda', requires_grad=True)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        output = torch.nn.functional.mse_loss(input, target)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(retain_graph=True), quantiles=quantiles)
    if provider == 'naive':
        output = naive_mse(input, target)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(retain_graph=True), quantiles=quantiles)
    if provider == 'triton':
        output = triton_mse(input, target)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(retain_graph=True), quantiles=quantiles)
    perf = lambda ms: (2 * M * N + 2 * M * N + 2) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)