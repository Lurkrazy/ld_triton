
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer
from torch.nn.parameter import Parameter
import torch.distributed as dist


# https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
class NaiveSGD(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        maximize: bool = False,
        optim_type = torch.float32,
    ):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        
        defaults = dict(
            lr=lr, 
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            maximize=maximize,
            optim_type=optim_type,
        )

        super(NaiveSGD, self).__init__(params, defaults)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
        print('state:', self.state)


    def irecv_object_list(object_list, src, device, group=None):
        current_device = device
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long, device=current_device)

        # Receive object sizes
        recv_worker = dist.irecv(object_sizes_tensor, src=src, group=group)
        
        recv_worker.wait()
        # Tensor to receive serialized objects into.
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device
        )

        rank_objects = recv(object_tensor, src=src, group=group)
        assert rank_sizes == rank_objects, "Mismatch in return ranks for object sizes and objects."
        # Deserialize objects using their stored sizes.
        offset = 0
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset : offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)
        return rank_objects


    @torch.no_grad()
    def step(self, closure=None):
        # world_size = dist.get_world_size()
        # rank = dist.get_rank()
        # next_rank = (rank + 1) % world_size
        # pre_rank = (rank - 1) % world_size
        # print(f'rank: {rank}, next_rank: {next_rank}, pre_rank: {pre_rank}')
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_bb = []

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError('NaiveSGD does not support sparse gradients')
                grads.append(p.grad)

                momentum = group['momentum']

                if momentum != 0:
                    state = self.state[p]
                    state_bb.append(state.get('momentum_buffer'))

            naive_sgd(
                params_with_grad,
                grads,
                state_bb,
                lr = group['lr'],
                momentum = group['momentum'],
                dampening = group['dampening'],
                weight_decay = group['weight_decay'],
                maximize = group['maximize'],
                optim_type = group['optim_type'],
            )

            if group['momentum'] != 0:
                for p, bb_t in zip(params_with_grad, state_bb):
                    state = self.state[p]
                    state['momentum_buffer'] = bb_t
        
        for _ in range(world_size - 1):
            dist.all_reduce(loss)
            dist.recv_object_list(grads, src=pre_rank)
            dist.send_object_list(grads, dst=next_rank)
            
        return loss


def naive_sgd(
    params,
    grads,
    state_momentums: List[Tensor],
    *,
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    maximize: bool,
    optim_type = torch.float32,
):
    func = _single_tensor_naive_sgd
    func(
        params,
        grads,
        state_momentums,
        lr = lr,
        momentum = momentum,
        dampening = dampening,
        weight_decay = weight_decay,
        maximize = maximize,
        optim_type = optim_type,
    )


def _single_tensor_naive_sgd(
    params: List[Tensor],
    grads: List[Tensor],
    state_bb: List[Tensor],
    lr: float,
    momentum: float,
    dampening: float,
    weight_decay: float,
    maximize: bool,
    optim_type = torch.float32,
):
    for i, param in enumerate(params):
        # grad = grads[i] if not maximize else -grads[i]
        grad = grads[i]
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        
        if momentum != 0:
            bb_t = state_bb[i]
            if bb_t is None:
                bb_t = torch.clone(grad).detach().to(optim_type)
                state_bb[i] = bb_t
            else:
                bb_t.mul_(momentum).add_(grad.to(optim_type), alpha=1 - dampening)
            grad = bb_t.to(param.dtype)
        
        if maximize:
            param.add_(grad, alpha=lr)
        else:
            param.add_(grad, alpha=-lr)
        
from torch.nn.parallel import DistributedDataParallel as DDP
# torchrun --nnodes 1 --nproc_per_node 3 naive_sgd_ringallreduce.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    batch = 2
    in_features = 8
    out_features = 16
    device = f'cpu'
    factory_kwargs = {'device': device, 'dtype': torch.float16}

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.fc = torch.nn.Linear(in_features, out_features, bias=True, **factory_kwargs)

        def forward(self, x):
            x = self.fc(x)
            return x
    lr = 0.1
    momentum = 0.1
    dampening = 0.1
    weight_decay = 0.1
    maximize = False
    optim_type = torch.float32
    if rank == 0:
        weight = torch.randn(out_features, in_features,  **factory_kwargs)
        bias = torch.randn(out_features, **factory_kwargs)

        dist.broadcast(weight, 0)
        dist.broadcast(bias, 0)

    else:
        weight = torch.empty(out_features, in_features, **factory_kwargs)
        bias = torch.empty(out_features, **factory_kwargs)
        dist.broadcast(weight, 0)
        dist.broadcast(bias, 0)
    # data parallel
    x = torch.randn(batch, in_features, **factory_kwargs)
    target = torch.randn(batch, out_features, **factory_kwargs)

    # model = Model().to(device)
    # model.fc.weight = Parameter(weight.clone())
    # model.fc.bias = Parameter(bias.clone())
    # loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=lr, 
    #     momentum=momentum,
    #     dampening=dampening,
    #     weight_decay=weight_decay,
    #     maximize=maximize,
    # )

    
    naive_model = Model().to(device)
    naive_model.fc.weight = Parameter(weight.clone())
    naive_model.fc.bias = Parameter(bias.clone())
    naive_loss_fn = torch.nn.CrossEntropyLoss()
    naive_optimizer = NaiveSGD(
        naive_model.parameters(), 
        lr=lr, 
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        maximize=maximize,
        optim_type=optim_type,
    )
    naive_x = x.clone()
    naive_target = target.clone()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    next_rank = (rank + 1) % world_size
    pre_rank = (rank - 1) % world_size
    print(f'rank: {rank}, next_rank: {next_rank}, pre_rank: {pre_rank}')
    b = torch.tensor([0, 0])
    dist.send(b, dst=next_rank)
    a = torch.tensor([0, 0])
    dist.recv(a, src=pre_rank)
    for i in range(20):
        # y = model(x)
        # loss = loss_fn(y, target)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
        

        naive_y = naive_model(naive_x)
        naive_loss = naive_loss_fn(naive_y, naive_target)
        naive_optimizer.zero_grad()
        naive_loss.backward()
        naive_optimizer.step()
        

        rtol = 1e-2
        atol = 1e-2

        print(f'rank: {rank}, i: {i}, naive_loss: {naive_model.fc.bias}')
        # assert torch.allclose(y, naive_y, rtol=rtol, atol=atol), f'i: {i}, y: {y}, naive_y: {naive_y}, {torch.isclose(y, naive_y)}'

    dist.destroy_process_group()