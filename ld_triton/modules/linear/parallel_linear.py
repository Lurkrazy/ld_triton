
import torch

from ld_triton.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_all_reduce_launcher,
)


class _column_parallel_linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        return super().forward(*args, **kwargs)


class ColumnParallelLinear(torch.nn.Module):
    '''
        Y = XA + b. 
        A = [A_1, ..., A_p]

        Y = X(A.t())

            | A_1 |
            | .   |
        A = | .   | 
            | .   |
            | A_p |

    '''
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = 'cpu',
        gather_output=True,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()
        assert device in ['cpu', 'cuda'], f'device {device} not supported'
        if device == 'cuda':
            factory_kwargs = {'device': f'cuda:{torch.cuda.current_device()}', 'dtype': dtype}
        else:
            factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        world_size = get_tensor_model_parallel_world_size()
        assert out_features % world_size == 0, f'out_features {out_features} must be divisible by world_size {world_size}'
        self.output_features_per_partition = out_features // world_size


        self.weight = torch.nn.Parameter(
            torch.zeros((self.output_features_per_partition, self.in_features), **factory_kwargs)
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.output_features_per_partition, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_parallel = input
        output_parallel = torch.nn.functional.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            world_size = get_tensor_model_parallel_world_size()
            if world_size == 1:
                return output_parallel
            rank = get_tensor_model_parallel_rank()

            tensor_list = [torch.empty_like(output_parallel) for _ in range(world_size)]
            tensor_list[rank] = output_parallel
            torch.distributed.all_gather(tensor_list, output_parallel, group=get_tensor_model_parallel_group())
            output = torch.cat(tensor_list, dim=-1).contiguous()
        else:
            output = output_parallel
        return output
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    

# one node
# torchrun --nproc_per_node 4 --nnodes 1 ld_triton/modules/linear/parallel_linear.py
# two nodes
# torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master-addr 10.10.13.101 --master-port 29400 --max_restarts=0 ./ld_triton/distributed/pipe/gpipe.py
# torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master-addr 10.10.13.101 --master-port 29400  --max_restarts=0 ./ld_triton/distributed/pipe/gpipe.py
if __name__ == "__main__":
    import argparse
    import torch.distributed as dist
    from ld_triton.distributed.parallel_state import (
        initialize_model_parallel,
        initialize_all_reduce_launcher,
    )
    
    dist.init_process_group(
        backend='gloo',
    )

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tensor_parallel_size = 4
    pipeline_parallel_size = 1
    initialize_model_parallel(tensor_parallel_size,
                              pipeline_parallel_size)
    
    dist.all_reduce(torch.zeros(1).cpu())

    activation_group = dist.new_group(range(0, world_size))
    weight_group = dist.new_group(range(0, world_size))
    bias_group = dist.new_group(range(0, world_size))
    doutput_group = dist.new_group(range(0, world_size))

    batch_size = 16
    in_features = 1024
    out_features = 2048
    dtype = torch.float16
    device = 'cpu'
    factory_kwargs = {'device': device, 'dtype': dtype, 'requires_grad': True}
    if rank == 0:
        weight = torch.randn((out_features, in_features), **factory_kwargs)
        bias = torch.randn((out_features), **factory_kwargs)
        input = torch.randn((batch_size, in_features), **factory_kwargs)
        doutput = torch.randn((batch_size, out_features), **factory_kwargs)
        dist.broadcast(input, src=0, group=activation_group)
        dist.broadcast(weight, src=0, group=weight_group)
        dist.broadcast(bias, src=0, group=bias_group)
        dist.broadcast(doutput, src=0, group=doutput_group)
    else:
        input = torch.empty((batch_size, in_features), **factory_kwargs)
        weight = torch.empty((out_features, in_features), **factory_kwargs)
        bias = torch.empty((out_features), **factory_kwargs)
        doutput = torch.empty((batch_size, out_features), **factory_kwargs)
        dist.broadcast(input, src=0, group=activation_group)
        dist.broadcast(weight, src=0, group=weight_group)
        dist.broadcast(bias, src=0, group=bias_group)
        dist.broadcast(doutput, src=0, group=doutput_group)


    model = torch.nn.Linear(in_features, out_features, bias=False, device=factory_kwargs['device'], dtype=factory_kwargs['dtype'])

    model.weight = torch.nn.Parameter(weight)
    if bias is not None:
        model.bias = torch.nn.Parameter(bias)
    
    output = model(input)
    output.backward(doutput)

    dweight, model.weight.grad = model.weight.grad, None

    print(f'weight grad: {dweight}')

    column_model = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        dtype=dtype,
        device=device,
        gather_output=False
    )

    output_features_per_partition = out_features // world_size
    weight_shard = weight[rank * output_features_per_partition:(rank + 1) * output_features_per_partition, :]
    bias_shard = bias[rank * output_features_per_partition:(rank + 1) * output_features_per_partition]

    column_model.weight = torch.nn.Parameter(weight_shard)
    column_model.bias = torch.nn.Parameter(bias_shard)

    column_out = column_model(input)
    column_out.backward(doutput)

    column_dweight, column_model.weight.grad = column_model.weight.grad, None

    print(f'column weight grad: {column_dweight}')
    
    # assert torch.allclose(output, column_out)

    dist.barrier()
    dist.destroy_process_group()
    