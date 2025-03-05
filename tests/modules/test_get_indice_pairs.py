import pytest
import torch

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from ld_triton.ops.spconv.triton_sparseconv3d import get_indice_pairs as triton_get_indice_pairs


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_get_indice_pairs.py -k test_get_indice_pairs
@pytest.mark.parametrize("batch_size, C, K, R, stride, padding, dilation", 
                         [(1, 6, 7, 3, 1, 0, 1),
                        #   (1, 4, 5, 2, 1, 0, 1),
                        #   (2, 7, 5, 3, 2, 2, 2)
                          ])
def test_get_indice_pairs(batch_size, C, K, R, stride, padding, dilation):
    print('test_get_indice_pairs')

    HW_0 = 23
    HW_1 = 23
    HW_2 = 23
    spatial_shape = [HW_0, HW_1, HW_2]
    indices = torch.tensor([(0, 6, 1, 1), (0, 7, 2, 2),  (0, 5, 0, 0),
                            (0, 18, 9, 1), (0, 19, 9, 1), (0, 17, 9, 1),
                            (0, 15, 7, 1), (0, 15, 8, 1), (0, 15, 6, 1),
                            (0, 7, 8, 1), (0, 7, 8, 2), (0, 7, 8, 0),
                            (0, 20, 7, 1), (0, 21, 8, 1), (0, 19, 6, 1),
                            (0, 3, 13, 1), (0, 3, 14, 2), (0, 3, 12, 0),
                            (0, 3, 7, 1), (0, 4, 7, 2), (0, 2, 7, 0), 
                            ], device='cuda', dtype=torch.int32)
    
    
    outids, indice_pairs, indice_pair_num = spconv.ops.get_indice_pairs(
                            indices.to('cuda').int(),
                            1,
                            spatial_shape,
                            ConvAlgo.Native,
                            [R, R, R],
                            [stride, stride, stride],
                            [padding, padding, padding],
                            [dilation, dilation, dilation],
                            [0, 0, 0],
                            False,
                            False)
    gather_idx = indice_pairs[0, :, :]
    scatter_idx = indice_pairs[1, :, :]

    PQ_0 = (HW_0 + 2 * padding - dilation * (R - 1) - 1) // stride + 1
    PQ_1 = (HW_1 + 2 * padding - dilation * (R - 1) - 1) // stride + 1
    PQ_2 = (HW_2 + 2 * padding - dilation * (R - 1) - 1) // stride + 1

    
    triton_outids, triton_gather_idx, triton_scatter_idx,  triton_indice_pair_num = triton_get_indice_pairs(
        indices.to('cuda').int(),
        HW_0, HW_1, HW_2,
        PQ_0, PQ_1, PQ_2,
        R, R, R,
        stride, stride, stride,
        padding, padding, padding,
        dilation, dilation, dilation,)
    
    gather_idx_sort, _ = torch.sort(gather_idx, dim=-1, descending=True)
    scatter_idx_sort, _ = torch.sort(scatter_idx, dim=-1, descending=True)
    triton_gather_idx_sort, _ = torch.sort(triton_gather_idx, dim=-1, descending=True)
    triton_scatter_idx_sort, _ = torch.sort(triton_scatter_idx, dim=-1, descending=True)

    assert torch.allclose(gather_idx_sort, triton_gather_idx_sort)
    assert torch.allclose(scatter_idx_sort, triton_scatter_idx_sort)
    assert torch.allclose(outids, triton_outids)
    for rs in range(gather_idx.shape[0]):
        assert torch.allclose(gather_idx[rs][gather_idx[rs] != -1], triton_gather_idx[rs][triton_gather_idx[rs] != -1])
        assert torch.allclose(scatter_idx[rs][scatter_idx[rs] != -1], triton_scatter_idx[rs][triton_scatter_idx[rs] != -1])
