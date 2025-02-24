
import torch
import pytest
from ld_triton.ops.spconv.triton_hash import LinearHashTableSplit


# python -m pytest tests/ops/spconv/test_hash.py
@pytest.mark.parametrize('N', [100, 1000, 10000])
def test_triton_hash(N):
    table = LinearHashTableSplit(N, rate = 2.0)
    key = torch.arange(0, N, dtype=torch.int32, device='cuda')
    val = torch.arange(0, N, dtype=torch.int32, device='cuda')
    key_idx = torch.randperm(N)
    key = key[key_idx]
    val_idx = torch.randperm(N)
    val = val[val_idx]
    table.insert(key.to(torch.uint32), val.to(torch.uint32))
    val = table.lookup_offset(key.to(torch.uint32))


    d_key = key.cpu().tolist()
    d_val = val.cpu().tolist()
    d = {}
    for i in range(N):
        d[d_key[i]] = d_val[i]
    d_val = [d[i] for i in d_key]
    
    assert torch.allclose(val, torch.tensor(d_val, dtype=torch.uint32, device='cuda'))