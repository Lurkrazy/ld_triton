
import pytest
import torch
from ld_triton.ops.matmul.naive_matmul import naive_matmul
from ld_triton.ops.matmul.triton_matmul import triton_matmul

@pytest.mark.parametrize('M, N, K', [(512, 512 + 1, 512 + 2)])
def test_matmul(M, N, K, dtype=torch.float16, device='cuda'):
    A = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
    B = torch.randn(K, N, device=device, dtype=dtype, requires_grad=True)
    dC = torch.randn(M, N, device=device, dtype=dtype, requires_grad=True)
    C = torch.matmul(A, B)
    C.backward(dC)
    dA, A.grad = A.grad.clone(), None
    dB, B.grad = B.grad.clone(), None

    naive_C = naive_matmul(A, B)
    naive_C.backward(dC)
    naive_dA, A.grad = A.grad.clone(), None
    naive_dB, B.grad = B.grad.clone(), None

    tt_C = triton_matmul(A, B)
    tt_C.backward(dC)
    tt_dA, A.grad = A.grad.clone(), None
    tt_dB, B.grad = B.grad.clone(), None
    
    assert torch.allclose(C, naive_C, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dA, naive_dA, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dB, naive_dB, atol=1e-3, rtol=1e-3)

    assert torch.allclose(C, tt_C, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dA, tt_dA, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dB, tt_dB, atol=1e-3, rtol=1e-3)

