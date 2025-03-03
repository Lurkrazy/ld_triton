
import torch
import triton

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.benchmark.core import get_voxel_data_large
from ld_triton.ops.spconv.triton_submconv3d import get_indice_pairs as triton_get_indice_pairs


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        # x_vals=[i for i in range(32, 913903, 32)],
        # x_vals=[i for i in range(32, 32 * 15, 32)],
        x_vals=[i for i in range(913903, 913903 + 32, 32)],
        line_arg='provider',
        line_vals=['spconv', "triton"], 
        line_names=['Spconv', "Triton"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel='GB/s',
        plot_name='rms-norm-fwd',
        args={}
    ))
def bench_submconv3d_fwd(N, provider,  device='cuda'):
    R = 3
    # 913903
    (_, coors, spatial_shape) = get_voxel_data_large()
    coors_th = torch.from_numpy(coors).to(device).int()[:N]
    triton_coors_th = coors_th.clone()
    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
                        coors_th.to('cuda').int(),
                        1,
                        spatial_shape,
                        ConvAlgo.Native,
                        [R, R, R],
                        [1, 1, 1],
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0],
                        True,
                        False)
    gather_idx = indice_pairs[0, :, :]
    gather_idx, _ = torch.sort(gather_idx, descending=True)
    scatter_idx = indice_pairs[1, :, :]
    scatter_idx, _ = torch.sort(scatter_idx, descending=True)

    PQ_0 = HW_0 = spatial_shape[0]
    PQ_1 = HW_1 = spatial_shape[1]
    PQ_2 = HW_2 = spatial_shape[2]
    dil_hw_0, dil_hw_1, dil_hw_2 = 1, 1, 1
    str_hw_0, str_hw_1, str_hw_2 = 1, 1, 1
    pad_hw_0 = (R // 2) * dil_hw_0
    pad_hw_1 = (R // 2) * dil_hw_1
    pad_hw_2 = (R // 2) * dil_hw_2
    triton_gather_idx, triton_scatter_idx,  triton_indice_pair_num = triton_get_indice_pairs(
        triton_coors_th.to('cuda').int(),
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        R, R, R,
        str_hw_0, str_hw_1, str_hw_2,
        pad_hw_0, pad_hw_1, pad_hw_2,
        dil_hw_0, dil_hw_1, dil_hw_2,)
    
    
    triton_gather_idx, _ = torch.sort(triton_gather_idx, descending=True)
    triton_scatter_idx, _ = torch.sort(triton_scatter_idx, descending=True)

    assert torch.allclose(gather_idx, triton_gather_idx)
    assert torch.allclose(scatter_idx, triton_scatter_idx)
    assert torch.allclose(indice_pair_num, triton_indice_pair_num)
    print(f'outids: {outids}')
    def y_fwd():
        if provider == "spconv":
            # return model(x_sp)
            return spconv.ops.get_indice_pairs(
                        coors_th.to('cuda').int(),
                        1,
                        spatial_shape,
                        ConvAlgo.Native,
                        [R, R, R],
                        [1, 1, 1],
                        [0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0],
                        True,
                        False)
        if provider == "triton":
            return triton_get_indice_pairs(
                        coors_th.to('cuda').int(),
                        spatial_shape[0], spatial_shape[1], spatial_shape[2],
                        spatial_shape[0], spatial_shape[1], spatial_shape[2],
                        R, R, R,
                        1, 1, 1,
                        0, 0, 0,
                        1, 1, 1,)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=10)
    return ms, min_ms, max_ms


# PYTHONWARNINGS="ignore::FutureWarning" python benchmarks/modules/spconv/bench_submconv3d_get_indice_pairs.py
if __name__ == "__main__":
    bench_submconv3d_fwd.run(show_plots=False, print_data=True)