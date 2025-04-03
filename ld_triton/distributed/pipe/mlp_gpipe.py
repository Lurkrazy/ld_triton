
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
            if weight.grad is None:
                weight.grad = torch.zeros_like(weight)
            grad_weight = torch.matmul(grad_output.t(), input)
            weight.grad.add_(grad_weight)

        if bias is not None and bias.requires_grad:
            if bias.grad is None:
                bias.grad = torch.zeros_like(bias)
            grad_bias = grad_output.sum(0, keepdim=False)
            bias.grad.add_(grad_bias)
        return grad_input, None, None


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
        self.weight = Parameter(torch.ones((out_features, in_features), **factory_kwargs) * 0.5)
        if bias:
            self.bias = Parameter(torch.ones(out_features, **factory_kwargs) * 0.5)
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
             ) 
            for i in range(len(node_sizes) - 1)]
        )
        
        self.register_forward_hook(self._forward_hook)
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_full_backward_pre_hook(self._backward_pre_hook)
        self.register_full_backward_hook(self._backward_hook)

    def _forward_pre_hook(self, module, input):
        if self._rank != 0:
            recv_ops = []
            for x in input:
                recv_op = dist.P2POp(dist.irecv, x, self._rank - 1, group=self._group)
                recv_ops.append(recv_op)
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

    def _forward_hook(self, module, input, output):
        if self._rank != self._world_size - 1:
            send_op = dist.P2POp(dist.isend, output, self._rank + 1, group=self._group)
            reqs = dist.batch_isend_irecv([send_op])
            for req in reqs:
                req.wait()

    def _backward_pre_hook(self, module, grad_output):
        if self._rank != self._world_size - 1:
            recv_ops = []
            for x in grad_output:
                recv_op = dist.P2POp(dist.irecv, x, self._rank + 1, group=self._group)
                recv_ops.append(recv_op)
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

    def _backward_hook(self, module, grad_input, grad_output):
        if self._rank != 0:
            send_ops = []
            for x in grad_input:
                send_op = dist.P2POp(dist.isend, x, self._rank - 1, group=self._group)
                send_ops.append(send_op)
            reqs = dist.batch_isend_irecv(send_ops)
            for req in reqs:
                req.wait()

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
                    )
                    for i in range(len(node_sizes) - 1)]) 
                for (node_sizes, node_activations) in zip(sizes, activations)])

    def forward(self, x):
        return self.layers(x)
    

class GPipeMLPTrain():
    def __init__(
        self,
        sizes: list[list[int]],
        activations: list[str],
        micro_batch_size,
        num_micro_batches,
        rank: int,
        world_size: int,
        device=None,
        dtype=None,
        group=None,
        debug=False,
    ):  
        super().__init__()
        self._debug = debug
        self._micro_batch_size = micro_batch_size
        self._num_micro_batches = num_micro_batches
        self._mini_batch_size = micro_batch_size * num_micro_batches
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._device = device
        self._dtype = dtype
        self.pipe_buffers = {
            'inputs': [None for _ in range(self._num_micro_batches)],
            'grad_outputs': [None for _ in range(self._num_micro_batches)],
            'debug_outputs': [None for _ in range(self._num_micro_batches)],
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
        for micro_batch_id in range(self._num_micro_batches):
            self.pipe_buffers['inputs'][micro_batch_id] = torch.empty(
                self._micro_batch_size, 
                sizes[self._rank][0], 
                device=self._device, 
                dtype=self._dtype, 
                requires_grad=True
            )

            self.pipe_buffers['grad_outputs'][micro_batch_id] = torch.empty(
                self._micro_batch_size, 
                sizes[self._rank][-1], 
                device=self._device, 
                dtype=self._dtype, 
                requires_grad=True
            )
    
    def first_stage_forward(self, x: torch.Tensor):
        assert x.shape[0] % self._num_micro_batches == 0, f'Batch size {x.shape[0]} must be divisible by {self._num_micro_batches}'
        assert x.shape[0] == self._mini_batch_size, f'Batch size {x.shape[0]} must be equal to {self._mini_batch_size}'
        for micro_batch_id in range(self._num_micro_batches):
            micro_input = x[micro_batch_id * self._micro_batch_size:(micro_batch_id + 1) * self._micro_batch_size]
            self.pipe_buffers['inputs'][micro_batch_id] = micro_input.clone()
            # recompute
            with torch.no_grad():
                micro_output = self.model(self.pipe_buffers['inputs'][micro_batch_id])
            if self._debug:
                self.pipe_buffers['debug_outputs'][micro_batch_id] = micro_output

    def forward(self):
        for micro_batch_id in range(self._num_micro_batches):
            micro_input = self.pipe_buffers['inputs'][micro_batch_id]
            # recompute
            with torch.no_grad():
                micro_output = self.model(micro_input)
            if self._debug:
                self.pipe_buffers['debug_outputs'][micro_batch_id] = micro_output

    def backward_and_update(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model.zero_grad()
        for micro_batch_id in range(self._num_micro_batches, 0, -1):
            micro_batch_id -= 1
            micro_input = self.pipe_buffers['inputs'][micro_batch_id].requires_grad_()
            # recompute
            micro_output = self.model(micro_input)
            micro_grad_output = self.pipe_buffers['grad_outputs'][micro_batch_id]
            micro_output.backward(micro_grad_output)
        for name, p in self.model.named_parameters():
            p.grad.div_(self._num_micro_batches)
        self.optimizer.step()
    
    def last_stage_backward_and_update(self, target: torch.Tensor):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.model.zero_grad()
        for micro_batch_id in range(self._num_micro_batches-1, -1, -1):
            # recompute
            micro_input = self.pipe_buffers['inputs'][micro_batch_id]
            micro_output = self.model(micro_input)
            micro_target = target[micro_batch_id * self._micro_batch_size:(micro_batch_id + 1) * self._micro_batch_size].clone()
            micro_loss = self.loss_fn(micro_output, micro_target)
            micro_loss.backward()
        for name, p in self.model.named_parameters():
            p.grad.div_(self._num_micro_batches)
        optimizer.step()

    def train(self):
        step = 100
        for i in range(step):
            self.model.zero_grad()
            # forward
            if self._rank == 0:
                x = torch.randn(self._mini_batch_size, sizes[self._rank][0], device=self._device, dtype=self._dtype, requires_grad=True)
                if self._debug:
                    single_x = x.clone()
                    dist.broadcast(single_x, src=0, group=self._group)
                self.first_stage_forward(x)
            else:
                if self._debug:
                    single_x = torch.empty(self._mini_batch_size, sizes[0][0], device=self._device, dtype=self._dtype, requires_grad=True)
                    dist.broadcast(single_x, src=0, group=self._group)
                self.forward()
            
            # backward
            if rank == world_size - 1:
                target = torch.ones(self._mini_batch_size, sizes[rank][-1], device=self._device, dtype=self._dtype, requires_grad=True)
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

                    print(f'single_output, Rank: {self._rank}, single_output: {single_output}')
                    print(f'output, Rank: {self._rank}, output: {output}')
                    assert torch.allclose(output, single_output, atol=1e-2, rtol=1e-2), f'Rank: {rank}, output: {output}, single_output: {single_output}'
            else:
                self.backward_and_update()


# torchrun --nnodes 1 --nproc_per_node 4 ld_triton/distributed/pipe/mlp_gpipe.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    group = dist.new_group()
    # sizes = [[4, 8, 8], [8, 8, 16], [16, 16, 4]]
    sizes = [[2, 2], [2, 2], [2, 2], [2, 2]]
    activations = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'softmax']]

    _micro_batch_size = 2
    _num_micro_batches = 2
    _debug = True
    # _debug = False
    
    train = GPipeMLPTrain(
        sizes,
        activations,
        _micro_batch_size,
        _num_micro_batches,
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