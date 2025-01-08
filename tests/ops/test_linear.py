
import torch
import pytest

from ld_triton.ops.linear.naive_linear import naive_linear
from ld_triton.ops.linear.triton_linear import triton_linear

# python -m pytest -v -rsx -s -W ignore::UserWarning linear.py -k test_linear
@pytest.mark.parametrize('M, in_features, out_features', [(128, 256, 512)])
def test_linear(M, in_features, out_features):
    factory_kwargs = {'device': 'cuda', 'dtype': torch.float}
    input = torch.randn(M, in_features, requires_grad=True, **factory_kwargs)
    weight = torch.randn(out_features, in_features, requires_grad=True, **factory_kwargs)
    bias = torch.randn(out_features, requires_grad=True, **factory_kwargs)

    output = torch.functional.F.linear(input, weight, bias)
    doutput = torch.rand_like(output)
    output.backward(doutput, retain_graph=True)
    dinput, input.grad = input.grad.clone(), None
    dweight, weight.grad = weight.grad.clone(), None
    dbias, bias.grad = bias.grad.clone(), None

    naive_output = naive_linear(input, weight, bias)
    naive_output.backward(doutput, retain_graph=True)
    naive_dinput, input.grad = input.grad.clone(), None
    naive_dweight, weight.grad = weight.grad.clone(), None
    naive_dbias, bias.grad = bias.grad.clone(), None

    ld_output = triton_linear(input, weight, bias)
    ld_output.backward(doutput)
    ld_dinput, input.grad = input.grad.clone(), None
    ld_dweight, weight.grad = weight.grad.clone(), None
    ld_dbias, bias.grad = bias.grad.clone(), None

    assert torch.allclose(output, naive_output, atol=1e-3, rtol=1e-3)
    assert torch.allclose(output, ld_output, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dinput, naive_dinput, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dinput, ld_dinput, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dweight, naive_dweight, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dweight, ld_dweight, atol=1e-1, rtol=1e-1)
    assert torch.allclose(dbias, naive_dbias, atol=1e-3, rtol=1e-3)
    assert torch.allclose(dbias, ld_dbias, atol=1e-3, rtol=1e-3)

    