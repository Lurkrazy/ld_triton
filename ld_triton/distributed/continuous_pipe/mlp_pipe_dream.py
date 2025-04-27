
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
        stage_id: int,
        num_stage: int,
        device=None,
        dtype=None,
        group=None,
    ):  
        assert len(sizes) == num_stage, f'Number of layers must be equal to num_stage'
        assert len(activations) == num_stage, f'Number of activations must be equal to num_stage'
        super().__init__()
        self._group = group
        self._stage_id = stage_id
        self._num_stage = num_stage
        self._device = device
        self._dtype = dtype
        
        node_sizes = sizes[self._stage_id]
        node_activations = activations[self._stage_id]
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
        if self._stage_id != 0:
            for x in input:
                dist.recv(x, src=self._stage_id - 1, group=self._group)

    def _forward_hook(self, module, input, output):
        if self._stage_id != self._num_stage - 1:
            dist.send(output, dst=self._stage_id + 1, group=self._group)

    def _backward_pre_hook(self, module, grad_output):
        if self._stage_id != self._num_stage - 1:
            for x in grad_output:
                dist.recv(x, src=self._stage_id + 1, group=self._group)

    def _backward_hook(self, module, grad_input, grad_output):
        if self._stage_id != 0:
            for x in grad_input:
                dist.send(x, dst=self._stage_id - 1, group=self._group)

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
        assert len(activations) == len(sizes), f'Number of activations must be equal to num_stage'
        super().__init__()
        self._device = device
        self._dtype = dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.layers = nn.Sequential(
                *[nn.Sequential(
                    *[nn.Sequential(
                        LDLinear(node_sizes[i], node_sizes[i+1], bias=False, **factory_kwargs),
                        nn.LeakyReLU() if node_activations[i] == 'relu' else nn.LeakyReLU()
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
        num_global_step: int,
        micro_batch_size,
        num_micro_batches,
        stage_id: int,
        num_stage: int,
        device=None,
        dtype=None,
        group=None,
        debug=False,
    ):  
        super().__init__()
        self._debug = debug
        self._micro_batch_size = micro_batch_size
        self._num_micro_step = num_micro_batches
        self._mini_batch_size = micro_batch_size * num_micro_batches
        self._group = group
        self._device = device
        self._dtype = dtype
        self._num_stage = num_stage
        self._stage_id = stage_id
        self._num_global_step = num_global_step
        self.pipe_buffers = {
            'inputs': [None for _ in range(self._num_stage)],
            'weights': [None for _ in range(self._num_stage - 1 - self._stage_id)],
            'grad_outputs': [None for _ in range(self._num_stage)],
            'debug_outputs': [None for _ in range(self._num_stage)],
        }
        self.init_pipe_buffers()
        self.model = NaivePipeMLP(
            sizes,
            activations,
            stage_id,
            num_stage,
            device=self._device,
            dtype=self._dtype,
            group=group
        )

        if self._stage_id == num_stage - 1:
            self.loss_fn = torch.nn.MSELoss()
            if self._debug:
                self.single_mlp = SingleMLP(sizes, activations, device=self._device, dtype=self._dtype)
                self.single_loss_fn = torch.nn.MSELoss()

    def init_pipe_buffers(self):
        factory_kwargs = {'device': self._device, 'dtype': self._dtype, 'requires_grad': True}
        for stage_id in range(self._num_stage):
            self.pipe_buffers['inputs'][stage_id] = torch.empty(
                self._micro_batch_size, 
                sizes[self._stage_id][0], 
                **factory_kwargs
            )

            self.pipe_buffers['grad_outputs'][stage_id] = torch.empty(
                self._micro_batch_size, 
                sizes[self._stage_id][-1], 
                **factory_kwargs
            )
            
        for i in range(self._num_stage - 1 - stage_id):
            self.pipe_buffers['weights'][i] = torch.empty(
                sizes[self._stage_id][0],
                sizes[self._stage_id][-1], 
                **factory_kwargs
            )
    
    def first_stage_forward(self, x: torch.Tensor):
        # first stage: split the input into micro batches
        assert x.shape[0] % self._num_micro_step == 0, f'Batch size {x.shape[0]} must be divisible by {self._num_micro_step}'
        assert x.shape[0] == self._mini_batch_size, f'Batch size {x.shape[0]} must be equal to {self._mini_batch_size}'
        for micro_step_id in range(self._num_micro_step):
            micro_input = x[micro_step_id * self._micro_batch_size:(micro_step_id + 1) * self._micro_batch_size]
            self.pipe_buffers['inputs'][micro_step_id] = micro_input.clone()
            # recompute
            with torch.no_grad():
                micro_output = self.model(self.pipe_buffers['inputs'][micro_step_id])
            if self._debug:
                self.pipe_buffers['debug_outputs'][micro_step_id] = micro_output

    def forward(self):
        for micro_step_id in range(self._num_micro_step):
            micro_input = self.pipe_buffers['inputs'][micro_step_id]
            # recompute
            with torch.no_grad():
                micro_output = self.model(micro_input)
            if self._debug:
                self.pipe_buffers['debug_outputs'][micro_step_id] = micro_output

    def backward_and_update(self):
        for micro_step_id in range(self._num_micro_step, 0, -1):
            micro_step_id -= 1
            micro_input = self.pipe_buffers['inputs'][micro_step_id].requires_grad_()
            # recompute
            micro_output = self.model(micro_input)
            micro_grad_output = self.pipe_buffers['grad_outputs'][micro_step_id]

            self.model.zero_grad()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            micro_output.backward(micro_grad_output)
            self.optimizer.step()
    
    def last_stage_backward_and_update(self, target: torch.Tensor):
        for micro_step_id in range(self._num_micro_step-1, -1, -1):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            # recompute
            micro_input = self.pipe_buffers['inputs'][micro_step_id]
            micro_output = self.model(micro_input)
            micro_target = target[micro_step_id * self._micro_batch_size:(micro_step_id + 1) * self._micro_batch_size].clone()
            micro_loss = self.loss_fn(micro_output, micro_target)
            self.model.zero_grad()
            micro_loss.backward()
            optimizer.step()

    def train(self):
        for i in range(self._num_global_step):
            self.model.zero_grad()
            # forward
            if self._stage_id == 0:
                x = torch.randn(self._mini_batch_size, sizes[self._stage_id][0], device=self._device, dtype=self._dtype, requires_grad=True)
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
            if self._stage_id == self._num_stage - 1:
                target = torch.ones(self._mini_batch_size, sizes[self._stage_id][-1], device=self._device, dtype=self._dtype, requires_grad=True)
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
                    print(f'single_mlp_out, stage_id: {self._stage_id}, single_output: {single_output}')
                    print(f'gpipe_mlp_out, stage_id: {self._stage_id}, output: {output}')
                    # assert torch.allclose(output, single_output, atol=1e-2, rtol=1e-2), f'stage_id: {stage_id}, output: {output}, single_output: {single_output}'
            else:
                self.backward_and_update()


# torchrun --nnodes 1 --nproc_per_node 4 ld_triton/distributed/continuous_pipe/mlp_pipe_dream.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    _world_size = dist.get_world_size()
    _rank = dist.get_rank()
    group = dist.new_group()
    # sizes = [[4, 8, 8], [8, 8, 16], [16, 16, 4]]
    sizes = [[2, 2], [2, 2], [2, 2], [2, 2]]
    activations = [['relu', 'relu'], ['relu', 'relu'], ['relu', 'relu'], ['relu', 'softmax']]

    _num_global_step = 2
    _num_stage = _world_size
    _stage_id = _rank
    _micro_batch_size = 2
    _num_micro_step= 4
    
    _debug = True
    # _debug = False
    
    train = GPipeMLPTrain(
        sizes,
        activations,
        _num_global_step,
        _micro_batch_size,
        _num_micro_step,
        _stage_id,
        _num_stage,
        device='cpu',
        # dtype=torch.float64,
        dtype=torch.float32,
        group=group,
        debug=_debug,
    )
    train.train()

    dist.destroy_process_group()