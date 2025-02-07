
import torch
import pytest
from ld_triton.modules.spconv.utils import SparseConvTensor
from ld_triton.modules.spconv.naive_spconv2d import NaiveSparseConv2d
from ld_triton.modules.spconv.naive_spconv3d import NaiveSparseConv3d
from ld_triton.modules.spconv.naive_submconv2d import NaiveSubMConv2d


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_spconv.py -k test_sparse_conv2d
@pytest.mark.parametrize("batch_size, C, H, W, K, R, S, stride, padding, dilation", 
                         [(2, 6, 3, 3, 7, 2, 2, 1, 0, 1), 
                          (1, 4, 3, 3, 5, 2, 2, 1, 0, 1),
                          (2, 7, 8, 8, 5, 3, 3, 2, 2, 2)])
def test_sparse_conv2d(batch_size, C, H, W, K, R, S, stride, padding, dilation):
    print('test_sparse_conv2d')
    import spconv.pytorch as spconv
    from spconv.pytorch import functional as Fsp

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SparseConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
        
    weight = torch.randn((K, R, S, C), device='cpu', dtype=torch.float32)
    bias = torch.randn((K,), device='cpu', dtype=torch.float32)
    x = torch.randn((batch_size, H, W, C), device='cuda', dtype=torch.float32)
    x_sp = spconv.SparseConvTensor.from_dense(x)

    model = Net()
    model.to('cuda')
    model.net.weight = torch.nn.Parameter(weight.to('cuda'))
    model.net.bias = torch.nn.Parameter(bias.to('cuda'))

    x_sp._features = x_sp.features.clone().detach().requires_grad_(True)
    out = model(x_sp)

    y_sp_features = torch.zeros_like(out.features) # the all elements must same, because the indices order is not same
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y_sp_features, out.features)
    loss.backward()
    dweight, model.net.weight.grad = model.net.weight.grad.clone(), None
    dbias, model.net.bias.grad = model.net.bias.grad.clone(), None
    dfeatures,  x_sp.features.grad = x_sp.features.grad.clone(), None
    dx_sp = x_sp.replace_feature(dfeatures)

    class NaiveNet(torch.nn.Module):
        def __init__(self):
            super(NaiveNet, self).__init__()
            self.net = NaiveSparseConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    naive_model = NaiveNet().to('cpu')
    x = x.to('cpu')
    x_sp = SparseConvTensor.from_dense(x)
    x_sp.features = x_sp.features.clone().detach().requires_grad_(True)
    naive_model.net.weight = torch.nn.Parameter(weight.to('cpu'))
    naive_model.net.bias = torch.nn.Parameter(bias.to('cpu'))

    naive_out = naive_model(x_sp)

    loss_fn = torch.nn.MSELoss()
    y_sp_features = y_sp_features.to('cpu')
    loss = loss_fn(y_sp_features, naive_out.features)
    loss.backward()
    naive_dweight, naive_model.net.weight.grad = naive_model.net.weight.grad.clone(), None
    naive_dbias, naive_model.net.bias.grad = naive_model.net.bias.grad.clone(), None
    naive_dfeatures, x_sp.features.grad = x_sp.features.grad.clone(), None
    naive_dx_sp = x_sp.replace_feature(naive_dfeatures)

    assert torch.allclose(model.net.weight.to('cpu'), naive_model.net.weight.to('cpu'))
    assert torch.allclose(model.net.bias.to('cpu'), naive_model.net.bias.to('cpu'))
    assert torch.allclose(out.dense().to('cpu'), naive_out.dense().to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dweight.to('cpu'), naive_dweight.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dbias.to('cpu'), naive_dbias.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx_sp.dense().to('cpu'), naive_dx_sp.dense().to('cpu'), rtol=1e-3, atol=1e-3)


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_spconv.py -k test_sparse_conv2d_1
@pytest.mark.parametrize("batch_size, C, K, R, S, stride, padding, dilation", 
                         [(2, 6, 7, 3, 3, 1, 0, 1),
                          (1, 4, 5, 2, 2, 1, 0, 1),
                          (2, 7, 5, 3, 3, 2, 2, 2)])
def test_sparse_conv2d_1(batch_size, C, K, R, S, stride, padding, dilation):
    print('test_sparse_conv2d_1')
    import spconv.pytorch as spconv
    from spconv.pytorch import functional as Fsp

    C = 5
    K = 7
    H = 23
    W = 23

    weight = torch.randn((K, R, S, C), device='cpu', dtype=torch.float32)
    bias = torch.randn((K,), device='cpu', dtype=torch.float32)

    indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13)], device='cuda', dtype=torch.int32)
    if batch_size == 2:
        indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13),
                                (1, 3 + 1, 7), (1, 6 + 1, 1), (1, 18 + 1, 9), (1, 15 + 1, 7), (1, 7 + 1, 8), (1, 20 + 1, 7), (1, 3 + 1, 13),], device='cuda', dtype=torch.int32)
    if batch_size == 3:
        indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13),
                                (1, 3 + 1, 7), (1, 6 + 1, 1), (1, 18 + 1, 9), (1, 15 + 1, 7), (1, 7 + 1, 8), (1, 20 + 1, 7), (1, 3 + 1, 13),
                                (2, 3, 7 + 1), (2, 6, 1 + 1), (2, 18, 9 + 1), (2, 15, 7 + 1), (2, 7, 8 + 1), (2, 20, 7 + 1), (0, 3, 13 + 1),], device='cuda', dtype=torch.int32)

    N = len(indices)
    features = torch.randn((N, C), device='cuda', dtype=torch.float32)
    x_sp = spconv.SparseConvTensor(features, indices, (H, W), batch_size)

    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SparseConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
        


    model = Net()
    model.to('cuda')
    model.net.weight = torch.nn.Parameter(weight.to('cuda'))
    model.net.bias = torch.nn.Parameter(bias.to('cuda'))

    x_sp._features = x_sp.features.clone().detach().requires_grad_(True)
    out = model(x_sp)

    y_sp_features = torch.zeros_like(out.features) # the all elements must same, because the indices order is not same
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y_sp_features, out.features)
    loss.backward()
    dweight, model.net.weight.grad = model.net.weight.grad.clone(), None
    dbias, model.net.bias.grad = model.net.bias.grad.clone(), None
    dfeatures,  x_sp.features.grad = x_sp.features.grad.clone(), None
    dx_sp = x_sp.replace_feature(dfeatures)

    class NaiveNet(torch.nn.Module):
        def __init__(self):
            super(NaiveNet, self).__init__()
            self.net = NaiveSparseConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    naive_model = NaiveNet().to('cpu')

    x_sp = SparseConvTensor(features.to('cpu'), indices.to('cpu'), (H, W), batch_size)
    x_sp.features = x_sp.features.clone().detach().requires_grad_(True)
    naive_model.net.weight = torch.nn.Parameter(weight.to('cpu'))
    naive_model.net.bias = torch.nn.Parameter(bias.to('cpu'))

    naive_out = naive_model(x_sp)

    loss_fn = torch.nn.MSELoss()
    y_sp_features = y_sp_features.to('cpu')
    loss = loss_fn(y_sp_features, naive_out.features)
    loss.backward()
    naive_dweight, naive_model.net.weight.grad = naive_model.net.weight.grad.clone(), None
    naive_dbias, naive_model.net.bias.grad = naive_model.net.bias.grad.clone(), None
    naive_dfeatures, x_sp.features.grad = x_sp.features.grad.clone(), None
    naive_dx_sp = x_sp.replace_feature(naive_dfeatures)

    assert torch.allclose(model.net.weight.to('cpu'), naive_model.net.weight.to('cpu'))
    assert torch.allclose(model.net.bias.to('cpu'), naive_model.net.bias.to('cpu'))
    assert torch.allclose(out.dense().to('cpu'), naive_out.dense().to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dweight.to('cpu'), naive_dweight.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dbias.to('cpu'), naive_dbias.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx_sp.dense().to('cpu'), naive_dx_sp.dense().to('cpu'), rtol=1e-3, atol=1e-3)


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_spconv.py -k test_sparse_conv3d
@pytest.mark.parametrize("batch_size, C, HW_0, HW_1, HW_2, K, RS_0, RS_1, RS_2, stride, padding, dilation", 
                         [(2, 6, 3, 3, 3, 7, 2, 2, 2, 1, 0, 1), 
                          (1, 4, 3, 3, 3, 5, 2, 2, 2, 1, 0, 1),
                          (2, 7, 8, 8, 8, 5, 3, 3, 3, 2, 2, 2)])
def test_sparse_conv3d(batch_size, C, HW_0, HW_1, HW_2, K, RS_0, RS_1, RS_2, stride, padding, dilation):
    print('test_sparse_conv3d')
    import spconv.pytorch as spconv
    from spconv.pytorch import functional as Fsp

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SparseConv3d(C, K, RS_0, stride, padding, dilation, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
        
    weight = torch.randn((K, RS_0, RS_1, RS_2, C), device='cpu', dtype=torch.float32)
    bias = torch.randn((K,), device='cpu', dtype=torch.float32)
    x = torch.randn((batch_size, HW_0, HW_1, HW_2, C), device='cuda', dtype=torch.float32)
    x_sp = spconv.SparseConvTensor.from_dense(x)

    model = Net()
    model.to('cuda')
    model.net.weight = torch.nn.Parameter(weight.to('cuda'))
    model.net.bias = torch.nn.Parameter(bias.to('cuda'))

    x_sp._features = x_sp.features.clone().detach().requires_grad_(True)
    out = model(x_sp)

    y_sp_features = torch.zeros_like(out.features) # the all elements must same, because the indices order is not same
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y_sp_features, out.features)
    loss.backward()
    dweight, model.net.weight.grad = model.net.weight.grad.clone(), None
    dbias, model.net.bias.grad = model.net.bias.grad.clone(), None
    dfeatures,  x_sp.features.grad = x_sp.features.grad.clone(), None
    dx_sp = x_sp.replace_feature(dfeatures)

    class NaiveNet(torch.nn.Module):
        def __init__(self):
            super(NaiveNet, self).__init__()
            self.net = NaiveSparseConv3d(C, K, RS_0, stride, padding, dilation, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    naive_model = NaiveNet().to('cpu')
    x = x.to('cpu')
    x_sp = SparseConvTensor.from_dense(x)
    x_sp.features = x_sp.features.clone().detach().requires_grad_(True)
    naive_model.net.weight = torch.nn.Parameter(weight.to('cpu'))
    naive_model.net.bias = torch.nn.Parameter(bias.to('cpu'))

    naive_out = naive_model(x_sp)

    loss_fn = torch.nn.MSELoss()
    y_sp_features = y_sp_features.to('cpu')
    loss = loss_fn(y_sp_features, naive_out.features)
    loss.backward()
    naive_dweight, naive_model.net.weight.grad = naive_model.net.weight.grad.clone(), None
    naive_dbias, naive_model.net.bias.grad = naive_model.net.bias.grad.clone(), None
    naive_dfeatures, x_sp.features.grad = x_sp.features.grad.clone(), None
    naive_dx_sp = x_sp.replace_feature(naive_dfeatures)

    assert torch.allclose(model.net.weight.to('cpu'), naive_model.net.weight.to('cpu'))
    assert torch.allclose(model.net.bias.to('cpu'), naive_model.net.bias.to('cpu'))
    assert torch.allclose(out.dense().to('cpu'), naive_out.dense().to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dweight.to('cpu'), naive_dweight.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dbias.to('cpu'), naive_dbias.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx_sp.dense().to('cpu'), naive_dx_sp.dense().to('cpu'), rtol=1e-3, atol=1e-3)


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_spconv.py -k test_submconv2d_0
@pytest.mark.parametrize("batch_size, C, H, W, K, R, S, stride, padding, dilation", 
                         [(2, 6, 3, 3, 7, 3, 3, 1, 0, 1), 
                          (1, 4, 3, 3, 5, 3, 3, 1, 0, 1),
                          (2, 7, 8, 8, 5, 3, 3, 2, 2, 2)])
def test_submconv2d_0(batch_size, C, H, W, K, R, S, stride, padding, dilation):
    print('test_submconv2d_0')
    import spconv.pytorch as spconv
    from spconv.pytorch import functional as Fsp

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SubMConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
        
    weight = torch.randn((K, R, S, C), device='cpu', dtype=torch.float32)
    bias = torch.randn((K,), device='cpu', dtype=torch.float32)
    x = torch.randn((batch_size, H, W, C), device='cuda', dtype=torch.float32)
    x_sp = spconv.SparseConvTensor.from_dense(x)

    model = Net()
    model.to('cuda')
    model.net.weight = torch.nn.Parameter(weight.to('cuda'))
    model.net.bias = torch.nn.Parameter(bias.to('cuda'))

    x_sp._features = x_sp.features.clone().detach().requires_grad_(True)
    out = model(x_sp)

    y_sp_features = torch.zeros_like(out.features) # the all elements must same, because the indices order is not same
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y_sp_features, out.features)
    loss.backward()
    dweight, model.net.weight.grad = model.net.weight.grad.clone(), None
    dbias, model.net.bias.grad = model.net.bias.grad.clone(), None
    dfeatures,  x_sp.features.grad = x_sp.features.grad.clone(), None
    dx_sp = x_sp.replace_feature(dfeatures)

    class NaiveNet(torch.nn.Module):
        def __init__(self):
            super(NaiveNet, self).__init__()
            self.net = NaiveSubMConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    naive_model = NaiveNet().to('cpu')
    x = x.to('cpu')
    x_sp = SparseConvTensor.from_dense(x)
    x_sp.features = x_sp.features.clone().detach().requires_grad_(True)
    naive_model.net.weight = torch.nn.Parameter(weight.to('cpu'))
    naive_model.net.bias = torch.nn.Parameter(bias.to('cpu'))

    naive_out = naive_model(x_sp)

    loss_fn = torch.nn.MSELoss()
    y_sp_features = y_sp_features.to('cpu')
    loss = loss_fn(y_sp_features, naive_out.features)
    loss.backward()
    naive_dweight, naive_model.net.weight.grad = naive_model.net.weight.grad.clone(), None
    naive_dbias, naive_model.net.bias.grad = naive_model.net.bias.grad.clone(), None
    naive_dfeatures, x_sp.features.grad = x_sp.features.grad.clone(), None
    naive_dx_sp = x_sp.replace_feature(naive_dfeatures)

    assert torch.allclose(model.net.weight.to('cpu'), naive_model.net.weight.to('cpu'))
    assert torch.allclose(model.net.bias.to('cpu'), naive_model.net.bias.to('cpu'))
    assert torch.allclose(out.dense().to('cpu'), naive_out.dense().to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dweight.to('cpu'), naive_dweight.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dbias.to('cpu'), naive_dbias.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx_sp.dense().to('cpu'), naive_dx_sp.dense().to('cpu'), rtol=1e-3, atol=1e-3)


# python -m pytest -W ignore::DeprecationWarning -W ignore::FutureWarning -s tests/modules/test_spconv.py -k test_submconv2d_1
@pytest.mark.parametrize("batch_size, C, K, R, S, stride, padding, dilation", 
                         [(1, 6, 7, 3, 3, 2, 2, 1),
                          (1, 4, 5, 3, 3, 1, 0, 1),
                          (2, 7, 5, 3, 3, 2, 2, 2)
                          ])
def test_submconv2d_1(batch_size, C, K, R, S, stride, padding, dilation):
    print('test_subm_conv2d_1')
    import spconv.pytorch as spconv
    from spconv.pytorch import functional as Fsp

    C = 5
    K = 7
    H = 23
    W = 23

    weight = torch.randn((K, R, S, C), device='cpu', dtype=torch.float32)
    bias = torch.randn((K,), device='cpu', dtype=torch.float32)

    indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13), (0, 4, 14),], device='cuda', dtype=torch.int32)
    if batch_size == 2:
        indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13), (0, 4, 14),
                                (1, 3 + 1, 7), (1, 6 + 1, 1), (1, 18 + 1, 9), (1, 15 + 1, 7), (1, 7 + 1, 8), (1, 20 + 1, 7), (1, 3 + 1, 13), (1, 4 + 1, 14),], device='cuda', dtype=torch.int32)
    if batch_size == 3:
        indices = torch.tensor([(0, 3, 7), (0, 6, 1), (0, 18, 9), (0, 15, 7), (0, 7, 8), (0, 20, 7), (0, 3, 13), (0, 2, 12),
                                (1, 3 + 1, 7), (1, 6 + 1, 1), (1, 18 + 1, 9), (1, 15 + 1, 7), (1, 7 + 1, 8), (1, 20 + 1, 7), (1, 3 + 1, 13), (0, 2 + 1, 12),
                                (2, 3, 7 + 1), (2, 6, 1 + 1), (2, 18, 9 + 1), (2, 15, 7 + 1), (2, 7, 8 + 1), (2, 20, 7 + 1), (0, 3, 13 + 1), (0, 2, 12 + 1),], device='cuda', dtype=torch.int32)

    N = len(indices)
    features = torch.randn((N, C), device='cuda', dtype=torch.float32)
    x_sp = spconv.SparseConvTensor(features, indices, (H, W), batch_size)

    
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = spconv.SubMConv2d(C, K, R, stride, padding, dilation, bias=True)
            # self.net = spconv.SparseConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: spconv.SparseConvTensor):
            x = self.net(x)
            return x
        


    model = Net()
    model.to('cuda')
    model.net.weight = torch.nn.Parameter(weight.to('cuda'))
    model.net.bias = torch.nn.Parameter(bias.to('cuda'))

    x_sp._features = x_sp.features.clone().detach().requires_grad_(True)
    out = model(x_sp)

    y_sp_features = torch.zeros_like(out.features) # the all elements must same, because the indices order is not same
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(y_sp_features, out.features)
    loss.backward()
    dweight, model.net.weight.grad = model.net.weight.grad.clone(), None
    dbias, model.net.bias.grad = model.net.bias.grad.clone(), None
    dfeatures,  x_sp.features.grad = x_sp.features.grad.clone(), None
    dx_sp = x_sp.replace_feature(dfeatures)

    class NaiveNet(torch.nn.Module):
        def __init__(self):
            super(NaiveNet, self).__init__()
            self.net = NaiveSubMConv2d(C, K, R, stride, padding, dilation, bias=True)

        def forward(self, x: SparseConvTensor):
            x = self.net(x)
            return x
        
    naive_model = NaiveNet().to('cpu')

    x_sp = SparseConvTensor(features.to('cpu'), indices.to('cpu'), (H, W), batch_size)
    x_sp.features = x_sp.features.clone().detach().requires_grad_(True)
    naive_model.net.weight = torch.nn.Parameter(weight.to('cpu'))
    naive_model.net.bias = torch.nn.Parameter(bias.to('cpu'))

    naive_out = naive_model(x_sp)

    loss_fn = torch.nn.MSELoss()
    y_sp_features = y_sp_features.to('cpu')
    loss = loss_fn(y_sp_features, naive_out.features)
    loss.backward()
    naive_dweight, naive_model.net.weight.grad = naive_model.net.weight.grad.clone(), None
    naive_dbias, naive_model.net.bias.grad = naive_model.net.bias.grad.clone(), None
    naive_dfeatures, x_sp.features.grad = x_sp.features.grad.clone(), None
    naive_dx_sp = x_sp.replace_feature(naive_dfeatures)

    assert torch.allclose(model.net.weight.to('cpu'), naive_model.net.weight.to('cpu'))
    assert torch.allclose(model.net.bias.to('cpu'), naive_model.net.bias.to('cpu'))
    torch.set_printoptions(threshold=torch.inf)
    
    # print(f'naive_out: {naive_out.dense().to("cpu")}')
    assert torch.allclose(out.dense().to('cpu'), naive_out.dense().to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dweight.to('cpu'), naive_dweight.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dbias.to('cpu'), naive_dbias.to('cpu'), rtol=1e-3, atol=1e-3)
    assert torch.allclose(dx_sp.dense().to('cpu'), naive_dx_sp.dense().to('cpu'), rtol=1e-3, atol=1e-3)
