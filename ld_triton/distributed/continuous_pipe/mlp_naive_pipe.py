
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
    

class NaivePipeMLPTrain():
    def __init__(
        self,
        sizes: list[list[int]],
        activations: list[str],
        batch_size,
        rank: int,
        world_size: int,
        device=None,
        dtype=None,
        group=None,
        debug=False,
    ):  
        super().__init__()
        self._debug = debug
        self._batch_size = batch_size
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._dtype = dtype
        self.pipe_buffers = {
            'inputs': [None],
            'grad_outputs': [None],
            'debug_outputs': [None],
        }
        self.init_pipe_buffers()
        self.num_stages = world_size
        self.model = NaivePipeMLP(
            sizes,
            activations,
            rank,
            world_size,
            device=self._device,
            dtype=self._dtype,
            group=group
        )

        if rank == world_size - 1:
            self.loss_fn = torch.nn.MSELoss()
            if self._debug:
                self.single_mlp = SingleMLP(sizes, activations, device=self._device, dtype=self._dtype)
                self.single_loss_fn = torch.nn.MSELoss()

    def init_pipe_buffers(self):
        self.pipe_buffers['inputs'][0] = torch.empty(
            self._batch_size, 
            sizes[self._rank][0], 
            device=self._device, 
            dtype=self._dtype, 
            requires_grad=True
        )

        self.pipe_buffers['grad_outputs'][0] = torch.empty(
            self._batch_size, 
            sizes[self._rank][-1], 
            device=self._device, 
            dtype=self._dtype, 
            requires_grad=True
        )
    
    def first_stage_forward(self, x: torch.Tensor):
        self.pipe_buffers['inputs'][0] = x.clone()
        # recompute
        with torch.no_grad():
            output = self.model(self.pipe_buffers['inputs'][0])
        if self._debug:
            self.pipe_buffers['debug_outputs'][0] = output

    def forward(self):
        input = self.pipe_buffers['inputs'][0]
        # recompute
        with torch.no_grad():
            output = self.model(input)
        if self._debug:
            self.pipe_buffers['debug_outputs'][0] = output

    def backward_and_update(self):
        input = self.pipe_buffers['inputs'][0].requires_grad_()
        # recompute
        output = self.model(input)
        grad_output = self.pipe_buffers['grad_outputs'][0]

        self.model.zero_grad()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        output.backward(grad_output)
        self.optimizer.step()
    
    def last_stage_backward_and_update(self, target: torch.Tensor):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        # recompute
        input = self.pipe_buffers['inputs'][0]
        output = self.model(input)
        loss = self.loss_fn(output, target)
        self.model.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self):
        num_global_step = 20
        for global_stable_id in range(num_global_step):
            self.model.zero_grad()
            # forward
            if self._rank == 0:
                x = torch.randn(self._batch_size, sizes[self._rank][0], device=self._device, dtype=self._dtype, requires_grad=True)
                if self._debug:
                    single_x = x.clone()
                    dist.broadcast(single_x, src=0, group=self._group)
                self.first_stage_forward(x)
            else:
                if self._debug:
                    single_x = torch.empty(self._batch_size, sizes[0][0], device=self._device, dtype=self._dtype, requires_grad=True)
                    dist.broadcast(single_x, src=0, group=self._group)
                self.forward()
            
            # backward
            if rank == world_size - 1:
                target = torch.ones(self._batch_size, sizes[rank][-1], device=self._device, dtype=self._dtype, requires_grad=True)
                self.last_stage_backward_and_update(target)

                if self._debug:
                    single_target = target.clone()
                    single_output = self.single_mlp(single_x)
                    single_loss = self.single_loss_fn(single_output, single_target)
                    self.single_mlp.zero_grad()
                    single_optimizer = torch.optim.SGD(self.single_mlp.parameters(), lr=0.1)
                    single_loss.backward()
                    single_optimizer.step()
                    output = torch.cat(self.pipe_buffers['debug_outputs'], dim=0)
                    print(f'single_mlp_out, Rank: {self._rank}, single_output: {single_output}')
                    print(f'pipe_mlp_out, Rank: {self._rank}, output: {output}')
                    assert torch.allclose(output, single_output, atol=1e-2, rtol=1e-2), f'Rank: {rank}, output: {output}, single_output: {single_output}'
            else:
                self.backward_and_update()


# torchrun --nnodes 1 --nproc_per_node 4 ld_triton/distributed/continuous_pipe/mlp_naive_pipe.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group()
    # sizes = [[4, 8, 8], [8, 8, 16], [16, 16, 4]]
    sizes = [[2, 2], [2, 2], [2, 2], [2, 2]]
    activations = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'softmax']]

    batch_size = 2
    _debug = True
    # _debug = False
    
    train = NaivePipeMLPTrain(
        sizes,
        activations,
        batch_size,
        rank,
        world_size,
        device='cpu',
        # dtype=torch.float64,
        dtype=torch.float32,
        group=group,
        debug=_debug,
    )
    train.train()

    dist.destroy_process_group()
