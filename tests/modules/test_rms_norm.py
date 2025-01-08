
import torch
import pytest

from ld_triton.modules.rms_norm.llama3_rms_norm import Llama3RMSNorm
from ld_triton.modules.rms_norm.naive_rms_norm import NaiveRMSNorm
from ld_triton.modules.rms_norm.triton_rms_norm import TritonRMSNorm


# python -m pytest -s tests/modules/test_rms_norm.py -k test_2d_RMSNorm
@pytest.mark.parametrize('M, N', [(512, 513)])
def test_2d_RMSNorm(M, N):
    eps = 1e-6
    llama_m = Llama3RMSNorm(N, eps=eps)
    x = torch.randn(M, N, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)
    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None
    llama_dweight = llama_m.weight.grad

    normalized_shape = [N]
    m = torch.nn.RMSNorm(normalized_shape, elementwise_affine=True, eps=eps).cuda()
    y = m(x)
    loss = loss_fn(y, target)
    loss.backward()
    dx, x.grad = x.grad.clone(), None
    dweight = m.weight.grad.clone()

    naive_m = NaiveRMSNorm(N, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None
    naive_dweight = naive_m.weight.grad.clone()

    ld_m = TritonRMSNorm(N, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None
    ld_dweight = ld_m.weight.grad.clone()

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_y, ld_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_dx, ld_dx, atol=1e-3, rtol=1e-3)
 
    assert torch.allclose(llama_dweight, dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(naive_dweight, dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dweight, dweight, atol=1e-3, rtol=1e-3)


# python -m pytest -s tests/modules/test_rms_norm.py -k test_3d_RMSNorm
@pytest.mark.parametrize('B, N, D', [(3, 781, 129)])
def test_3d_RMSNorm(B, N, D):
    eps = 1e-6
    llama_m = Llama3RMSNorm(D, eps=eps)
    x = torch.randn(B, N, D, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)
    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None

    naive_m = NaiveRMSNorm(D, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None

    ld_m = TritonRMSNorm(D, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_y, naive_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dx, naive_dx, atol=1e-3, rtol=1e-3)


# python -m pytest -s tests/modules/test_rms_norm.py -k test_4d_RMSNorm
@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_RMSNorm(B, N, H, D):
    eps = 1e-6
    llama_m = Llama3RMSNorm(D, eps=eps)
    x = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    llama_y = llama_m(x)
    target = torch.randn_like(llama_y)
    loss_fn = torch.nn.MSELoss()
    llama_loss = loss_fn(llama_y, target)
    llama_loss.backward()
    llama_dx, x.grad = x.grad.clone(), None

    naive_m = NaiveRMSNorm(D, eps=eps)
    naive_y = naive_m(x)
    naive_loss = loss_fn(naive_y, target)
    naive_loss.backward()
    naive_dx, x.grad = x.grad.clone(), None

    ld_m = TritonRMSNorm(D, eps=eps)
    ld_y = ld_m(x)
    ld_loss = loss_fn(ld_y, target)
    ld_loss.backward()
    ld_dx, x.grad = x.grad.clone(), None

    assert torch.allclose(llama_y, naive_y, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_y, naive_y, atol=1e-3, rtol=1e-3)

    assert torch.allclose(llama_dx, naive_dx, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ld_dx, naive_dx, atol=1e-3, rtol=1e-3)
