import torch
import pytest

from ld_triton.ops.sigmoid.naive_sigmoid import naive_sigmoid
from ld_triton.ops.sigmoid.triton_sigmoid import triton_sigmoid


# python -m pytest -s tests/ops/test_sigmoid.py
@pytest.mark.parametrize('M, N', [(1823, 781)])
def test_sigmoid(M, N):
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    y = torch.sigmoid(x)
    dp = torch.randn_like(y)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_sigmoid(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = triton_sigmoid(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    # assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)