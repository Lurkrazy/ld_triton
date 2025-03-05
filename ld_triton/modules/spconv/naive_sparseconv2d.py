
import torch
from ld_triton.modules.spconv.utils import SparseConvTensor
from ld_triton.ops.spconv.naive_sparseconv2d import naive_sparseconv2d


# only support channel_last
class NaiveSparseConv2d(torch.nn.Module):
    def __init__(self,                  
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 device = 'cpu',
                 dtype = torch.float32
        ):
        super(NaiveSparseConv2d, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.C = in_channels
        self.K = out_channels
        if isinstance(kernel_size, int):
            self.R = kernel_size
            self.S = kernel_size
        else:
            self.R, self.S = kernel_size
        if isinstance(stride, int):
            self.str_h = stride
            self.str_w = stride
        else:
            self.str_h, self.str_w = stride
        if isinstance(padding, int):
            self.pad_h = padding
            self.pad_w = padding
        else:    
            self.pad_h, self.pad_w = padding
        if isinstance(dilation, int):
            self.dil_h = dilation
            self.dil_w = dilation
        else:
            self.dil_h, self.dil_w = dilation
        
        # KRSC
        self.weight_shape = [self.K, self.R, self.S, self.C]

        self.weight = torch.nn.Parameter(
            torch.zeros(self.weight_shape, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.K, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: SparseConvTensor):
        assert len(x.spatial_shape) == 2
        features = x.features
        indices = x.indices
        batch_size = x.batch_size
        H, W = x.spatial_shape
        naive_out_features, naive_out_indices, P, Q = naive_sparseconv2d(features, indices, H, W, batch_size, self.weight, self.bias, (self.str_h, self.str_w), (self.pad_h, self.pad_w), (self.dil_h, self.dil_w))
        output = SparseConvTensor(naive_out_features, naive_out_indices, [P, Q], batch_size)
        print(f'NaiveSparseConv2d: {H, W, P, Q}')
        return output
