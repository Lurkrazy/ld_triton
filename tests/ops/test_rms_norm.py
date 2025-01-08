
import torch
import pytest

from ld_triton.ops.rms_norm.naive_rms_norm import naive_rms_norm
from ld_triton.ops.rms_norm.triton_rms_norm import triton_rms_norm


# python -m pytest -s tests/ops/test_rms_norm.py -k test_2d_RMSNorm
@pytest.mark.parametrize('M, N', [(512, 513)])
def test_2d_RMSNorm(M, N):
    eps = 1e-6
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    weight = torch.rand(N, requires_grad=True, device='cuda')
    dy = torch.randn(M, N, requires_grad=True, device='cuda')
    normalized_shape = [N]

    y = torch.nn.functional.rms_norm(x, normalized_shape, weight=weight, eps=eps)
    y.backward(dy)
    dx, x.grad = x.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None

    naive_y = naive_rms_norm(x, weight, eps)
    naive_y.backward(dy)
    naive_dx, x.grad = x.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None

    ld_y = triton_rms_norm(x, weight, eps)
    ld_y.backward(dy)
    ld_dx, x.grad = x.grad.clone(), None
    ld_dweight, weight.grad = weight.grad.clone(), None

    assert torch.allclose(y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, ld_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dx, ld_dx, atol=1e-3, rtol=1e-3)
 
    assert torch.allclose(dweight, naive_dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweight, ld_dweight, atol=1e-3, rtol=1e-3)


# python -m pytest -s tests/ops/test_rms_norm.py -k test_3d_RMSNorm
@pytest.mark.parametrize('B, N, D', [(3, 781, 129)])
def test_3d_RMSNorm(B, N, D):
    eps = 1e-6
    x = torch.randn(B, N, D, requires_grad=True, device='cuda')
    weight = torch.rand(D, requires_grad=True, device='cuda')
    dy = torch.randn(B, N, D, requires_grad=True, device='cuda')
    normalized_shape = [D]

    y = torch.nn.functional.rms_norm(x, normalized_shape, weight=weight, eps=eps)
    y.backward(dy)
    dx, x.grad = x.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None

    naive_y = naive_rms_norm(x, weight, eps)
    naive_y.backward(dy)
    naive_dx, x.grad = x.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None

    ld_y = triton_rms_norm(x, weight, eps)
    ld_y.backward(dy)
    ld_dx, x.grad = x.grad.clone(), None
    ld_dweight, weight.grad = weight.grad.clone(), None

    assert torch.allclose(y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, ld_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dx, ld_dx, atol=1e-3, rtol=1e-3)
 
    assert torch.allclose(dweight, naive_dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweight, ld_dweight, atol=1e-3, rtol=1e-3)


# python -m pytest -s tests/ops/test_rms_norm.py -k test_4d_RMSNorm
@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_RMSNorm(B, N, H, D):
    eps = 1e-6
    x = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    weight = torch.rand(D, requires_grad=True, device='cuda')
    dy = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    normalized_shape = [D]

    y = torch.nn.functional.rms_norm(x, normalized_shape, weight=weight, eps=eps)
    y.backward(dy)
    dx, x.grad = x.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None

    naive_y = naive_rms_norm(x, weight, eps)
    naive_y.backward(dy)
    naive_dx, x.grad = x.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None

    ld_y = triton_rms_norm(x, weight, eps)
    ld_y.backward(dy)
    ld_dx, x.grad = x.grad.clone(), None
    ld_dweight, weight.grad = weight.grad.clone(), None

    assert torch.allclose(y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, ld_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dx, ld_dx, atol=1e-3, rtol=1e-3)
 
    assert torch.allclose(dweight, naive_dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweight, ld_dweight, atol=1e-3, rtol=1e-3)
