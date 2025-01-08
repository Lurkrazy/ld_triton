
import torch
import pytest

from ld_triton.ops.softmax.naive_softmax import naive_softmax
from ld_triton.ops.softmax.triton_softmax import triton_softmax


# python -m pytest -s tests/ops/test_softmax.py
@pytest.mark.parametrize('M, N', [(1823, 781)])
def test_2d_softmax(M, N):
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = triton_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)
    

@pytest.mark.parametrize('B, N, D', [(3, 781, 129)])
def test_3d_softmax(B, N, D):
    x = torch.randn(B, N, D, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = triton_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_softmax(B, N, H, D):
    x = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    y = torch.softmax(x, dim=-1)
    dp = torch.randn_like(x)
    y.backward(dp)
    dx, x.grad = x.grad.clone(), None
    naive_y = naive_softmax(x)
    naive_y.backward(dp)
    naive_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(dx, naive_dx, rtol=1e-3, atol=1e-3)
    assert torch.allclose(y, naive_y, rtol=1e-3, atol=1e-3)
    tt_y = triton_softmax(x)
    tt_y.backward(dp)
    tt_dx, x.grad = x.grad.clone(), None
    assert torch.allclose(y, tt_y, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx, tt_dx, rtol=1e-3, atol=1e-3)
