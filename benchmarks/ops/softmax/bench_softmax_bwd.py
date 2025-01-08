
import torch
import triton

from ld_triton.ops.softmax.naive_softmax import naive_softmax
from ld_triton.ops.softmax.triton_softmax import triton_softmax


perf_configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 10)],
        line_arg='provider',
        line_vals=['cublas', "naive", "triton"], 
        line_names=['cuBLAS', "Naive","Triton"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel='GB/s',
        plot_name="softmax-performance",
        args={
            'M': 4096,
            'mode': 'bwd',
        },
    )
]


@triton.testing.perf_report(perf_configs)
def benchmark(M, N, mode, provider, device='cuda'):
    x = torch.randn(M, N, device=device, requires_grad=True)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'cublas':
        # fn = lambda: torch.softmax(x, dim=-1)
        fn = lambda: torch.nn.functional.softmax(x, dim=-1)
        o =fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == 'naive':
        fn = lambda: naive_softmax(x)
        o =fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == 'triton':
        fn = lambda: triton_softmax(x)
        o =fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e3)

    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)