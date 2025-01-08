
import torch
import pytest

from ld_triton.ops.mse.naive_mse import naive_mse
from ld_triton.ops.mse.triton_mse import triton_mse

# python -m pytest -s tests/ops/test_mse.py -k test_1d_mse
@pytest.mark.parametrize('N', [(512, 513)])
def test_1d_mse(N):
    input = torch.randn(N, requires_grad=True, device='cuda')
    target = torch.randn(N, requires_grad=True, device='cuda')
    
    output = torch.nn.functional.mse_loss(input, target)
    output.backward()
    dinput, input.grad = input.grad.clone(), None
    dtarget, target.grad = target.grad.clone(), None

    naive_output = naive_mse(input, target)
    naive_output.backward()
    naive_dinput, input.grad = input.grad.clone(), None
    naive_dtarget, target.grad = target.grad.clone(), None

    ld_output = triton_mse(input, target)
    ld_output.backward()
    ld_dinput, input.grad = input.grad.clone(), None
    ld_dtarget, target.grad = target.grad.clone(), None

    assert torch.allclose(output, naive_output, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output, ld_output, rtol=1e-3, atol=1e-3)

    assert torch.allclose(dinput, naive_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dinput, ld_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, naive_dtarget, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, ld_dtarget, rtol=1e-3, atol=1e-3)


# python -m pytest -s tests/ops/test_mse.py -k test_2d_mse
@pytest.mark.parametrize('M, N', [(512, 513)])
def test_2d_mse(M, N):
    input = torch.randn(M, N, requires_grad=True, device='cuda')
    target = torch.randn(M, N, requires_grad=True, device='cuda')
    
    output = torch.nn.functional.mse_loss(input, target)
    output.backward()
    dinput, input.grad = input.grad.clone(), None
    dtarget, target.grad = target.grad.clone(), None

    naive_output = naive_mse(input, target)
    naive_output.backward()
    naive_dinput, input.grad = input.grad.clone(), None
    naive_dtarget, target.grad = target.grad.clone(), None

    ld_output = triton_mse(input, target)
    ld_output.backward()
    ld_dinput, input.grad = input.grad.clone(), None
    ld_dtarget, target.grad = target.grad.clone(), None

    assert torch.allclose(output, naive_output, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output, ld_output, rtol=1e-3, atol=1e-3)

    assert torch.allclose(dinput, naive_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dinput, ld_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, naive_dtarget, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, ld_dtarget, rtol=1e-3, atol=1e-3)


# python -m pytest -s tests/ops/test_mse.py -k test_4d_mse
@pytest.mark.parametrize('B, N, H, D', [(3, 781, 8, 129)])
def test_4d_mse(B, N, H, D):
    input = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    target = torch.randn(B, N, H, D, requires_grad=True, device='cuda')
    
    output = torch.nn.functional.mse_loss(input, target)
    output.backward()
    dinput, input.grad = input.grad.clone(), None
    dtarget, target.grad = target.grad.clone(), None

    naive_output = naive_mse(input, target)
    naive_output.backward()
    naive_dinput, input.grad = input.grad.clone(), None
    naive_dtarget, target.grad = target.grad.clone(), None

    ld_output = triton_mse(input, target)
    ld_output.backward()
    ld_dinput, input.grad = input.grad.clone(), None
    ld_dtarget, target.grad = target.grad.clone(), None

    assert torch.allclose(output, naive_output, rtol=1e-3, atol=1e-3)
    assert torch.allclose(output, ld_output, rtol=1e-3, atol=1e-3)

    assert torch.allclose(dinput, naive_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dinput, ld_dinput, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, naive_dtarget, rtol=1e-3, atol=1e-3)
    assert torch.allclose(dtarget, ld_dtarget, rtol=1e-3, atol=1e-3)
