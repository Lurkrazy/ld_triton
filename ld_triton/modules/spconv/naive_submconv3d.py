
import torch
from ld_triton.modules.spconv.utils import SparseConvTensor
from ld_triton.ops.spconv.naive_submconv3d import naive_submconv3d
from ld_triton.ops.spconv.triton_submconv3d import triton_submconv3d


# only support channel_last
class NaiveSubMConv3d(torch.nn.Module):
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
        super(NaiveSubMConv3d, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.C = in_channels
        self.K = out_channels
        if isinstance(kernel_size, int):
            self.RS_0 = kernel_size
            self.RS_1 = kernel_size
            self.RS_2 = kernel_size
        else:
            self.RS_0, self.RS_1, self.RS_2 = kernel_size
        if isinstance(stride, int):
            self.str_hw_0 = stride
            self.str_hw_1 = stride
            self.str_hw_2 = stride
        else:
            self.str_0, self.str_1, self.str_2 = stride
        if isinstance(padding, int):
            self.pad_hw_0 = padding
            self.pad_hw_1 = padding
            self.pad_hw_2 = padding
        else:    
            self.pad_hw_0, self.pad_hw_1, self.pad_hw_2 = padding
        if isinstance(dilation, int):
            self.dil_hw_0 = dilation
            self.dil_hw_1 = dilation
            self.dil_hw_2 = dilation
        else:
            self.dil_hw_0, self.dil_hw_1, self.dil_hw_2 = dilation
        
        # KRSC
        self.weight_shape = [self.K, self.RS_0, self.RS_1, self.RS_2, self.C]

        self.weight = torch.nn.Parameter(
            torch.zeros(self.weight_shape, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.K, **factory_kwargs))
        else:
            self.register_parameter('bias', None)


    def forward(self, x: SparseConvTensor):
        assert len(x.spatial_shape) == 3
        features = x.features
        indices = x.indices
        batch_size = x.batch_size
        HW_0, HW_1, HW_2 = x.spatial_shape
        naive_out_features, naive_out_indices, PQ_0, PQ_1, PQ_2= naive_submconv3d(features, indices, 
                                                                                     HW_0, HW_1, HW_2, batch_size, self.weight, self.bias, 
                                                                                     (self.str_hw_1, self.str_hw_1, self.str_hw_2), 
                                                                                     (self.pad_hw_0, self.pad_hw_1, self.pad_hw_2), 
                                                                                     (self.dil_hw_0, self.dil_hw_1, self.dil_hw_2))
        output = SparseConvTensor(naive_out_features, naive_out_indices, [PQ_0, PQ_1, PQ_2], batch_size)
        return output


# only support channel_last
class TritonSubMConv3d(torch.nn.Module):
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
        super(TritonSubMConv3d, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.C = in_channels
        self.K = out_channels
        if isinstance(kernel_size, int):
            self.RS_0 = kernel_size
            self.RS_1 = kernel_size
            self.RS_2 = kernel_size
        else:
            self.RS_0, self.RS_1, self.RS_2 = kernel_size
        if isinstance(stride, int):
            self.str_hw_0 = stride
            self.str_hw_1 = stride
            self.str_hw_2 = stride
        else:
            self.str_0, self.str_1, self.str_2 = stride
        if isinstance(padding, int):
            self.pad_hw_0 = padding
            self.pad_hw_1 = padding
            self.pad_hw_2 = padding
        else:    
            self.pad_hw_0, self.pad_hw_1, self.pad_hw_2 = padding
        if isinstance(dilation, int):
            self.dil_hw_0 = dilation
            self.dil_hw_1 = dilation
            self.dil_hw_2 = dilation
        else:
            self.dil_hw_0, self.dil_hw_1, self.dil_hw_2 = dilation
        
        # KRSC
        self.weight_shape = [self.K, self.RS_0, self.RS_1, self.RS_2, self.C]

        self.weight = torch.nn.Parameter(
            torch.zeros(self.weight_shape, **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.K, **factory_kwargs))
        else:
            self.register_parameter('bias', None)


    def forward(self, x: SparseConvTensor):
        assert len(x.spatial_shape) == 3
        features = x.features
        indices = x.indices
        batch_size = x.batch_size
        HW_0, HW_1, HW_2 = x.spatial_shape
        naive_out_features, naive_out_indices, PQ_0, PQ_1, PQ_2= triton_submconv3d(features, indices, 
                                                                                     HW_0, HW_1, HW_2, batch_size, self.weight, self.bias, 
                                                                                     (self.str_hw_1, self.str_hw_1, self.str_hw_2), 
                                                                                     (self.pad_hw_0, self.pad_hw_1, self.pad_hw_2), 
                                                                                     (self.dil_hw_0, self.dil_hw_1, self.dil_hw_2))
        output = SparseConvTensor(naive_out_features, naive_out_indices, [PQ_0, PQ_1, PQ_2], batch_size)
        return output

