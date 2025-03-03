
import torch
import triton

from ld_triton.modules.spconv.naive_submconv3d import NaiveSubMConv3d
from ld_triton.modules.spconv.triton_submconv3d import TritonSubMConv3d
import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
from spconv.core import ConvAlgo
from spconv.benchmark.core import get_voxel_data, get_voxel_data_large
from ld_triton.modules.spconv.utils import SparseConvTensor


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['C', 'K', 'R'],
        x_vals=[(3, 64, 3), 
                (64, 64, 3), 
                (64, 96, 3), (96, 96, 3), (96, 128, 3), 
                (128, 128, 3), (128, 160, 3), (160, 160, 3), (160, 192, 3), (192, 192, 3),
                (192, 224, 3), (224, 224, 3), (224, 256, 3)
                ],
        line_arg='provider',
        line_vals=['spconv', "naive"], 
        line_names=['Spconv', "Naive"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel='GB/s',
        plot_name='rms-norm-fwd',
        args={'dtype': torch.float16, 'mode': 'fwd'},
    ))
def bench_submconv3d_fwd(C, K, R, dtype, provider, mode='fwd', eps=1e-5, device='cuda'):
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SubMConv3d(C, K, R, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
    # (913903, 3), (80, 1600, 1600)
    (_, coors, spatial_shape) = get_voxel_data_large()
    voxels_th = torch.randn((coors.shape[0], C), device=device, dtype=dtype)
    # print(voxels_th.shape, coors.shape, spatial_shape)
    weight = torch.randn((K, R, R, R, C), device=device, dtype=dtype)
    bias = torch.randn((K,), device=device, dtype=dtype)
    coors_th = torch.from_numpy(coors).to(device).int()
    voxels_th.requires_grad = True
    x_sp = spconv.SparseConvTensor(voxels_th, coors_th, spatial_shape, 1)

    model = Net()
    model.to(device)
    model.net.weight = torch.nn.Parameter(weight)
    model.net.bias = torch.nn.Parameter(bias)
    out = model(x_sp)

    
    class TritonNet(torch.nn.Module):
        def __init__(self):
            super(TritonNet, self).__init__()
            self.net = TritonSubMConv3d(C, K, R, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    triton_model = TritonNet().to(device)
    triton_model.net.weight = torch.nn.Parameter(weight.to(device))
    triton_model.net.bias = torch.nn.Parameter(bias.to(device))
    triton_x_sp = SparseConvTensor(voxels_th.to(device), coors_th.to('cuda'), spatial_shape, 1)
    triton_out = triton_model(triton_x_sp)

    rtol = 0.0
    atol = 1.0
    if not torch.allclose(out.features, triton_out.features, rtol=rtol, atol=atol):
        # torch.set_printoptions(profile="full")
        print(f'out: {out.features}')
        print(f'triton_out: {triton_out.features}')
        print(f'out - triton_out: {torch.abs(out.features - triton_out.features)}')
        print(f'isclose: {torch.isclose(out.features, triton_out.features, rtol=rtol, atol=atol)}')
    else:
        assert torch.allclose(out.features, triton_out.features, rtol=rtol, atol=atol)
        
    def y_fwd():
        if provider == "spconv":
            return model(x_sp)
        if provider == "naive":
            return triton_model(triton_x_sp)
        
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=3)
    return ms, min_ms, max_ms


# PYTHONWARNINGS="ignore::FutureWarning" python benchmarks/modules/spconv/bench_submconv3d_fwd.py
if __name__ == "__main__":
    bench_submconv3d_fwd.run(show_plots=False, print_data=True)