
import torch
import torch.nn as nn
import torch.distributed as dist


class GPipeHook(nn.Module):
    def __init__(self, module, stage_id: int, num_stage: int, group=None):
        super().__init__()
        self.module = module
        self._stage_id = stage_id
        self._num_stage = num_stage
        self._group = group
        self.register_forward_hook(self._forward_hook)
        self.register_forward_pre_hook(self._forward_pre_hook)
        self.register_full_backward_pre_hook(self._backward_pre_hook)
        self.register_full_backward_hook(self._backward_hook)

    def _forward_pre_hook(self, module, input):
        if self._stage_id != 0:
            recv_ops = []
            for x in input:
                recv_op = dist.P2POp(dist.irecv, x, self._stage_id - 1, group=self._group)
                recv_ops.append(recv_op)
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

    def _forward_hook(self, module, input, output):
        if self._stage_id != self._num_stage - 1:
            send_op = dist.P2POp(dist.isend, output, self._stage_id + 1, group=self._group)
            reqs = dist.batch_isend_irecv([send_op])
            for req in reqs:
                req.wait()

    def _backward_pre_hook(self, module, grad_output):
        if self._stage_id != self._num_stage - 1:
            recv_ops = []
            for x in grad_output:
                recv_op = dist.P2POp(dist.irecv, x, self._stage_id + 1, group=self._group)
                recv_ops.append(recv_op)
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

    def _backward_hook(self, module, grad_input, grad_output):
        if self._stage_id != 0:
            send_ops = []
            for x in grad_input:
                send_op = dist.P2POp(dist.isend, x, self._stage_id - 1, group=self._group)
                send_ops.append(send_op)
            reqs = dist.batch_isend_irecv(send_ops)
            for req in reqs:
                req.wait()

    def forward(self, x):
        return self.module(x)


class GPipe(nn.Module):
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
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
        self._num_micro_batch = num_micro_batches
        self._mini_batch_size = micro_batch_size * num_micro_batches
        self._group = group
        self._num_stage = num_stage
        self._stage_id = stage_id
        self._device = device
        self._dtype = dtype
        self.pipe_buffers = {
            'inputs': [None for _ in range(self._num_micro_batch)],
            'grad_outputs': [None],
            'debug_outputs': [None for _ in range(self._num_micro_batch)],
        }
        
        self._model_input_shape = model.get_input_shape()
        if self._stage_id == 0:
            assert self._model_input_shape[0] % self._micro_batch_size == 0, f"micro_batch_size {self._micro_batch_size} must be a divisor of model input shape {self._model_input_shape[0]}"
        self._model_output_shape = model.get_output_shape()
        self._model_dtype = model.get_dtype()
        self._device = model.get_device()
        self._model = GPipeHook(model, self._stage_id, self._num_stage, group=group)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self.init_pipe_buffers()

    def init_pipe_buffers(self):
        factory_kwargs = {'device': self._device, 'dtype': self._dtype, 'requires_grad': True}
        shape = (self._micro_batch_size, *self._model_input_shape[1:])
        for micro_batch_id in range(self._num_micro_batch):
            self.pipe_buffers['inputs'][micro_batch_id] = torch.empty(
                shape,
                **factory_kwargs
            )

        shape = (self._micro_batch_size, *self._model_output_shape[1:])

        self.pipe_buffers['grad_outputs'][0] = torch.empty(
            shape,
            **factory_kwargs
        )
    
    # torchgpipe: 
    def _clock_cycles(self, num_stage, num_micro_batch, stage_id, micro_step_id):
        num_micro_step = 2 * (num_micro_batch + num_stage - 1)
        if micro_step_id < num_micro_batch + num_stage - 1:
            micro_batch_id = micro_step_id - stage_id
            if micro_batch_id < 0 or micro_batch_id >= num_micro_batch:
                return micro_batch_id, 'bubble'
            else:
                return micro_batch_id, 'forward'
        else:
            micro_batch_id = num_micro_step - (micro_step_id + stage_id + 1)
            if micro_batch_id < 0 or micro_batch_id >= num_micro_batch:
                return micro_batch_id, 'bubble'
            else:
                return micro_batch_id, 'backward'
    
    def _print_clock_cycles(self, num_stage, num_micro_batch):
        total_steps = 2 * (num_micro_batch + num_stage - 1)
        print(f'| Micro Steps ', end='')
        for step_id in range(total_steps):
            print(f'| {step_id:3} ', end='')
        print('|')
        for stage_id in range(num_stage):
            print(f'|  Stage{stage_id:3}   ', end='')
            for step_id in range(total_steps):
                if step_id < num_micro_batch + num_stage - 1:
                    micro_batch_id = step_id - stage_id
                    if micro_batch_id < 0 or micro_batch_id >= num_micro_batch:
                        print(f'|  *  ', end='')
                    else:
                        print(f'| {micro_batch_id:2}F ', end='')
                else:
                    micro_batch_id = total_steps - (step_id + stage_id + 1)
                    if micro_batch_id < 0 or micro_batch_id >= num_micro_batch:
                        print(f'|  *  ', end='')
                    else:
                        print(f'| {micro_batch_id:2}B ', end='')
            print('|')
        total_compute = total_steps * num_stage
        valid_compute = 2 * num_micro_batch * num_stage
        print(f'total compute: {total_compute}')
        print(f'valid compute: {valid_compute}')
        print(f'bubble numbers: {total_compute - valid_compute}')
        print(f'bubble ratio: {(total_compute - valid_compute) / total_compute:.2%}')

    def global_step(self, input: torch.Tensor, target: torch.Tensor):
        if self._stage_id == self._num_stage - 1 and self._debug:
            self._print_clock_cycles(self._num_stage, self._num_micro_batch)
        num_micro_step = 2 * (self._num_micro_batch + self._num_stage - 1)

        self._model.zero_grad()
        optimizer = self._optimizer(self._model.parameters(), lr=0.1)
        # forward
        if self._stage_id == 0:
            assert input.shape == self._model_input_shape, f"input shape {input.shape} must be equal to model input shape {self._model_input_shape}"
            input = input.chunk(self._num_micro_batch, dim=0)

        if self._stage_id == self._num_stage - 1:
            assert target.shape == self._model_output_shape, f"target shape {target.shape} must be equal to model output shape {self._model_output_shape}"
            target = target.chunk(self._num_micro_batch, dim=0)

        for micro_step_id in range(num_micro_step):
            micro_batch_id, step_type = self._clock_cycles(self._num_stage, self._num_micro_batch, self._stage_id, micro_step_id)
            if step_type == 'forward':
                if self._stage_id == 0:
                    self.pipe_buffers['inputs'][micro_batch_id] = input[micro_batch_id].clone()
                    # recompute
                    with torch.no_grad():
                        micro_output = self._model(self.pipe_buffers['inputs'][micro_batch_id])
                    if self._debug:
                        self.pipe_buffers['debug_outputs'][micro_batch_id] = micro_output
                else:
                    micro_input = self.pipe_buffers['inputs'][micro_batch_id]
                    # recompute
                    with torch.no_grad():
                        micro_output = self._model(micro_input)
                    if self._debug:
                        self.pipe_buffers['debug_outputs'][micro_batch_id] = micro_output
            elif step_type == 'backward':
                if self._stage_id == self._num_stage - 1:
                    # recompute
                    micro_input = self.pipe_buffers['inputs'][micro_batch_id].requires_grad_()
                    micro_output = self._model(micro_input)
                    micro_target = target[micro_batch_id].clone()
                    micro_loss = self._loss_fn(micro_output, micro_target)
                    micro_loss.backward()
                else:
                    # recompute
                    micro_input = self.pipe_buffers['inputs'][micro_batch_id].requires_grad_()
                    micro_output = self._model(micro_input)
                    micro_grad_output = self.pipe_buffers['grad_outputs']
                    for grad_output in micro_grad_output:
                        micro_output.backward(grad_output)

            elif step_type == 'bubble':
                pass

        for name, p in self._model.named_parameters():
            p.grad.div_(self._num_micro_batch)
        optimizer.step()

# torchrun --nproc_per_node 4 --nnodes 1 ld_triton/distributed/pipe/gpipe.py
if __name__ == '__main__':
    dist.init_process_group(backend='gloo')
    num_stage = dist.get_world_size()
    stage_id = dist.get_rank()
    group = dist.new_group()

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
            self.weight = nn.Parameter(torch.ones((out_features, in_features), **factory_kwargs) * 0.5)
            if bias:
                self.bias = nn.Parameter(torch.ones(out_features, **factory_kwargs) * 0.5)
            else:
                self.register_parameter('bias', None)

        def forward(self, input):
            return _naive_linear.apply(input, self.weight, self.bias)
    
    class NaivePipeMLP(nn.Module):
        def __init__(
            self, 
            size: int,
            global_batch_size: int,
            num_layer: int,
            stage_id: int,
            num_stage: int,
            device=None,
            dtype=None,
            group=None,
        ):  
            super().__init__()
            self._group = group
            self._stage_id = stage_id
            self._num_stage = num_stage
            self._device = device
            self._dtype = dtype
            self._size = size
            self._global_batch_size = global_batch_size
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.layers = nn.Sequential(
                *[nn.Sequential(
                    LDLinear(size, size, bias=False, **factory_kwargs),
                ) 
                for _ in range(num_layer)]
            )
            
        def forward(self, x):
            return self.layers(x)

        def get_input_shape(self):
            return torch.Size((self._global_batch_size , self._size))
        
        def get_output_shape(self):
            return torch.Size((self._global_batch_size , self._size))
        
        def get_dtype(self):
            return self._dtype
        
        def get_device(self):
            return self._device
        
    class SingleMLP(nn.Module):
        def __init__(
            self, 
            size: int,
            num_layer: int,
            num_stage: int,
            device=None,
            dtype=None,
        ):  
            super().__init__()
            self._device = device
            self._dtype = dtype
            factory_kwargs = {'device': device, 'dtype': dtype}
            self.layers = nn.Sequential(
                    *[nn.Sequential(
                        *[nn.Sequential(
                            LDLinear(size, size, bias=False, **factory_kwargs),
                        )
                        for _ in range(num_layer)]) 
                    for _ in range(num_stage)])

        def forward(self, x):
            return self.layers(x)

    size = 2
    num_layer = 2
    GBS = 32
    MBS = 4
    _debug = True
    _debug = False

    model = NaivePipeMLP(
        size=size,
        global_batch_size=GBS,
        num_layer=num_layer,
        stage_id=0,
        num_stage=4,
        device='cpu',
        dtype=torch.float32,
    )

    model = GPipe(
        model=model,
        loss_fn=nn.MSELoss(),
        optimizer=torch.optim.SGD,
        micro_batch_size=MBS,
        num_micro_batches=GBS // MBS,
        stage_id=stage_id,
        num_stage=num_stage,
        device='cpu',
        dtype=torch.float32,
        group=group,
        debug=_debug,
    )
    
    if _debug:
        single_model = SingleMLP(
            size=size,
            num_layer=num_layer,
            num_stage=num_stage,
            device='cpu',
            dtype=torch.float32,
        )

    
    num_global_step = 5
    if _debug:
        single_loss_fn = nn.MSELoss()

    for i in range(num_global_step):

        if stage_id != 0:
            if _debug:
                single_x = torch.empty(GBS, size, device='cpu', dtype=torch.float32, requires_grad=True)
                dist.broadcast(single_x, src=0, group=group)

        if stage_id == 0:
            input = torch.randn(GBS, size, device='cpu', dtype=torch.float32, requires_grad=True)
            if _debug:
                single_x = input.clone()
                dist.broadcast(single_x, src=0, group=group)
            model.global_step(input, None)

        elif stage_id == num_stage - 1:
            target = torch.randn(GBS, size, device='cpu', dtype=torch.float32, requires_grad=True)
            if _debug:
                single_target = target.clone()
            model.global_step(None, target)
            if _debug:
                single_output = single_model(single_x)
                single_loss = single_loss_fn(single_output, single_target)
                single_model.zero_grad()
                single_optimizer = torch.optim.SGD(single_model.parameters(), lr=0.1)
                single_loss.backward()
                single_optimizer.step()
                
                output = torch.cat(model.pipe_buffers['debug_outputs'], dim=0)
                print(f'single_output: {single_output}')
                print(f'output: {output}')
                assert torch.allclose(output, single_output, atol=1e-2, rtol=1e-2), f'Rank: {stage_id}, output: {output}, single_output: {single_output}'

        else:
            model.global_step(None, None)

    dist.barrier()
    dist.destroy_process_group()
