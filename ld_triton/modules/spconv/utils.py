
import torch

# Adaped from https://github.com/traveller59/spconv/blob/master/spconv/pytorch/core.py
def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    this function don't contain except handle code. so use this carefully
    when indice repeats, don't support repeat add which is supported
    in tensorflow.
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1]:]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[slices] = updates.view(*output_shape)
    return ret


class SparseConvTensor(object):
    def __init__(self, features: torch.Tensor, indices: torch.Tensor, spatial_shape: list, batch_size: int):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size

    def dense(self, channels_first: bool = True):
        output_shape = [self.batch_size] + list(
            self.spatial_shape) + [self.features.shape[1]]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features,
            output_shape)
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()
    
    @classmethod
    def from_dense(cls, x: torch.Tensor):
        x_sp = x.to_sparse(x.ndim - 1)
        spatial_shape = x_sp.shape[1:-1]
        batch_size = x_sp.shape[0]
        indices_th = x_sp.indices().permute(1, 0).contiguous().int()
        features_th = x_sp.values()
        return cls(features_th, indices_th, spatial_shape, batch_size)
    
    def replace_feature(self, feature: torch.Tensor):
        new_spt = SparseConvTensor(feature, self.indices, self.spatial_shape, self.batch_size)
        return new_spt
    
    def __repr__(self):
        return f"SparseConvTensor[shape={self.batch_size, self.spatial_shape, self.features.shape[1]}], indices={self.indices}, features={self.features}, spatial_shape={self.spatial_shape}"