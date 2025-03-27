
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.distributed as dist


class _naive_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        ctx.save_for_backward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, bias, = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = None, None, None
        if input.requires_grad:
            grad_input = torch.matmul(grad_output, weight)
        if weight.requires_grad:
            grad_weight = torch.matmul(grad_output.t(), input)
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.sum(0, keepdim=False)
        return grad_input, grad_weight, grad_bias


class LDLinear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.ones((out_features, in_features), **factory_kwargs) * 0.1)
        if bias:
            self.bias = Parameter(torch.zeros(out_features, **factory_kwargs) * 0.1)
        else:
            self.register_parameter('bias', None)


    def forward(self, input):
        return _naive_linear.apply(input, self.weight, self.bias)


class NaivePipeMLP(nn.Module):
    def __init__(
        self, 
        sizes: list[list[int]],
        activations: list[str],
        rank: int,
        world_size: int,
        device=None,
        dtype=None,
        group=None,
    ):  
        assert len(sizes) == world_size, f'Number of layers must be equal to world_size'
        assert len(activations) == world_size, f'Number of activations must be equal to world_size'
        super().__init__()
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._dtype = dtype
        
        node_sizes = sizes[rank]
        node_activations = activations[rank]
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.layers = nn.Sequential(
            *[nn.Sequential(
                LDLinear(node_sizes[i], node_sizes[i+1], bias=False, **factory_kwargs),
                nn.LeakyReLU() if node_activations[i] == 'relu' else nn.LeakyReLU()
             ) 
            for i in range(len(node_sizes) - 1)]
        )

        self.layers.register_forward_hook(self._forward_hook)
        self.layers.register_forward_pre_hook(self._forward_pre_hook)
        self.layers.register_full_backward_pre_hook(self._backward_pre_hook)
        self.layers.register_full_backward_hook(self._backward_hook)

    def _forward_pre_hook(self, module, input):
        if self._rank != 0:
            for x in input:
                dist.recv(x, src=self._rank - 1, group=self._group)

    def _forward_hook(self, module, input, output):
        if self._rank != self._world_size - 1:
            dist.send(output, dst=self._rank + 1, group=self._group)

    def _backward_pre_hook(self, module, grad_output):
        if self._rank != self._world_size - 1:
            for x in grad_output:
                dist.recv(x, src=self._rank + 1, group=self._group)

    def _backward_hook(self, module, grad_input, grad_output):
        if self._rank != 0:
            for x in grad_input:
                dist.send(x, dst=self._rank - 1, group=self._group)

    def forward(self, x):
        return self.layers(x)


class SingleMLP(nn.Module):
    def __init__(
        self, 
        sizes: list[list[int]],
        activations: list[str],
        device=None,
        dtype=None,
    ):  
        assert len(activations) == len(sizes), f'Number of activations must be equal to world_size'
        super().__init__()
        self._device = device
        self._dtype = dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.layers = nn.Sequential(
                *[nn.Sequential(
                    *[nn.Sequential(
                        LDLinear(node_sizes[i], node_sizes[i+1], bias=False, **factory_kwargs),
                        nn.LeakyReLU() if node_activations[i] == 'relu' else nn.LeakyReLU())
                    for i in range(len(node_sizes) - 1)]) 
                for (node_sizes, node_activations) in zip(sizes, activations)])

    def forward(self, x):
        return self.layers(x)
    

# torchrun --nnodes 1 --nproc_per_node 3 ld_triton/distributed/pipe/test_mlp_naive_pipe.py
if __name__ == '__main__':
    import time
    dist.init_process_group(backend='gloo')
    batch_size = 2
    step = 10
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group()
    sizes = [[4, 8, 8], [8, 8, 16], [16, 16, 4]]
    activations = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'softmax']]
    model = NaivePipeMLP(
        sizes,
        activations,
        rank,
        world_size,
        device='cpu',
        dtype=torch.float32,
        group=group
    )

    if rank == world_size - 1:
        loss_fn = torch.nn.CrossEntropyLoss()

        single_mlp = SingleMLP(sizes, activations, device='cpu', dtype=torch.float32)
        single_loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(step):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        if rank == 0:
            x = torch.randn(batch_size, sizes[rank][0], device='cpu', dtype=torch.float32, requires_grad=True)
            single_x = x.clone()
            dist.broadcast(single_x, src=0, group=group)
        else:
            x = torch.empty(batch_size, sizes[rank][0], device='cpu', dtype=torch.float32, requires_grad=True)
            single_x = torch.empty(batch_size, sizes[0][0], device='cpu', dtype=torch.float32, requires_grad=True)
            dist.broadcast(single_x, src=0, group=group)
        model.zero_grad()
        if rank == world_size - 1:
            target = torch.randn(batch_size, sizes[rank][-1], device='cpu', dtype=torch.float32, requires_grad=True)
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            single_target = target.clone()
            # print(f'step: {i}, Rank: {rank}, single_x: {single_x}')
            single_output = single_mlp(single_x)
            single_loss = single_loss_fn(single_output, single_target)
            single_mlp.zero_grad()
            single_optimizer = torch.optim.SGD(single_mlp.parameters(), lr=0.1)
            single_loss.backward()
            single_optimizer.step()
            print(f'Rank: {rank}, output: {output}, single_output: {single_output}')
            assert torch.allclose(output, single_output, atol=1e-2, rtol=1e-2), f'Rank: {rank}, output: {output}, single_output: {single_output}'
        else:
            grad_output = torch.empty(batch_size, sizes[rank][-1], device='cpu', dtype=torch.float32, requires_grad=True)
            output = model(x)
            output.backward(grad_output)

        optimizer.step()
    dist.destroy_process_group()