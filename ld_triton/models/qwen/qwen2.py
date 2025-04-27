
import ray
import torch

from ld_triton.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_all_reduce_launcher,
)

class Qwen2MLP(torch.nn.Module):
    def __init__(self,   
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linnear_method
    ) -> None:
        super().__init__()


class Qwen2ForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Q

@ray.remote(num_gpus=1, num_cpus=0)
def parallel_add(x):
    return x + x

# ray start --head --port=6379
# 
if __name__ == "__main__":
    ray.init()
    ray.nodes()
    # result_ids = [parallel_add.remote(torch.tensor([i, i], device='cuda:0')) for i in range(10)]
    # results = ray.get(result_ids)
    print(ray.nodes())
