import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer
from torch.nn.parameter import Parameter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class NaiveRingAllReduce(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super(NaiveRingAllReduce, self).__init__()
        self.module = module
        self._group = group
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()      
        params = list(self.module.parameters())
        if self._rank == 0:
            for p in params:
                dist.broadcast(p, 0)
        else:
            for p in params:
                dist.broadcast(p, 0)
        named_modules = list(self.module.named_modules())
        for name, m in named_modules:
            m.register_full_backward_hook(self._reduce_backward_hook)
            m.register_full_backward_pre_hook(self._async_backward_hook)
        
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def _async_backward_hook(self, m: torch.nn.Module, grad_output):
        pass
    
    def _reduce_backward_hook(self, m: torch.nn.Module, grad_input, grad_output):
        params = list(m.parameters())
        for p in params:
            # bug https://github.com/pytorch/pytorch/issues/149938
            if p.grad is None:
                continue
            worker = dist.all_reduce(p.grad.div_(self._world_size), op=dist.ReduceOp.SUM, group=self._group, async_op=True)
            worker.wait()


# torchrun --nnodes 1 --nproc_per_node 3 ld_triton/distributed/ring_allreduce.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group()
    
    batch = 1
    in_features = 4
    out_features = 8
    device = f'cpu'
    factory_kwargs = {'device': device, 'dtype': torch.float32}

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc = torch.nn.Linear(in_features, out_features, bias=False, **factory_kwargs)
            self.relu = torch.nn.ReLU()
            self.fc1 = torch.nn.Linear(out_features, out_features, bias=False, **factory_kwargs)

        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.fc1(x)
            return x
        
    lr = 0.1
    momentum = 0.1
    dampening = 0.1
    weight_decay = 0.1
    maximize = False
    optim_type = torch.float32


    weight = torch.randn(out_features, in_features,  **factory_kwargs, requires_grad=True)
    bias = torch.randn(out_features, **factory_kwargs, requires_grad=True)
        
    weight_1 = torch.randn(out_features, out_features,  **factory_kwargs, requires_grad=True)
    bias_1 = torch.randn(out_features, **factory_kwargs, requires_grad=True)

    # data parallel
    x = torch.randn(batch, in_features, **factory_kwargs, requires_grad=True)
    target = torch.randn(batch, out_features, **factory_kwargs, requires_grad=True)

    model = Model()
    model.fc.weight = Parameter(weight.clone())
    model.fc.bias = Parameter(bias.clone())
    model.fc1.weight = Parameter(weight_1.clone())
    model.fc1.bias = Parameter(bias_1.clone())
    model = DDP(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    
    naive_model = Model()
    naive_model.fc.weight = Parameter(weight.clone())
    naive_model.fc.bias = Parameter(bias.clone())
    naive_model.fc1.weight = Parameter(weight_1.clone())
    naive_model.fc1.bias = Parameter(bias_1.clone())
    naive_model = NaiveRingAllReduce(naive_model, group=group)
    naive_loss_fn = torch.nn.CrossEntropyLoss()
    naive_optimizer = torch.optim.SGD(
        naive_model.parameters(), 
        lr=lr, 
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    naive_x = x.clone()
    naive_target = target.clone()
    
    for i in range(20):
        y = model(x)
        loss = loss_fn(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        naive_y = naive_model(naive_x)
        naive_loss = naive_loss_fn(naive_y, naive_target)
        naive_optimizer.zero_grad()
        naive_loss.backward()
        naive_optimizer.step()

        rtol = 1e-2
        atol = 1e-2

        assert torch.allclose(y, naive_y, rtol=rtol, atol=atol), \
                              f'i: {i}, y: {y}, naive_y: {naive_y}, {torch.isclose(y, naive_y)}' \
                              f'x: {x}, naive_x: {naive_x}'

    dist.destroy_process_group()
