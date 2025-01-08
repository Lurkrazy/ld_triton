import torch
import triton

from ld_triton.ops.linear.naive_linear import naive_linear
from ld_triton.ops.linear.triton_linear import triton_linear


configs = []


configs.append(
    triton.testing.Benchmark(
        x_names=["M", "in_features", "out_features"],
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
def benchmark(M, in_features, out_features, provider):
    input = torch.randn((M, in_features), device='cuda', dtype=torch.float16, requires_grad=True)
    weight = torch.randn((out_features, in_features), device='cuda', dtype=torch.float16, requires_grad=False)
    bias = torch.randn(out_features, device='cuda', dtype=torch.float16, requires_grad=True)
    doutput = torch.randn((M, out_features), device='cuda', dtype=torch.float16, requires_grad=True)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        output = torch.functional.F.linear(input, weight, bias)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(doutput, retain_graph=True), quantiles=quantiles)
    if provider == 'naive':
        output = naive_linear(input, weight, bias)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(doutput, retain_graph=True), quantiles=quantiles)
    if provider == 'triton':
        output = triton_linear(input, weight, bias)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: output.backward(doutput, retain_graph=True), quantiles=quantiles)
    perf = lambda ms: (2 * M * in_features * out_features + 2 * M * in_features * out_features) * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)