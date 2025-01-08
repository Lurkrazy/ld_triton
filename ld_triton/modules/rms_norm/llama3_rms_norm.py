
import torch


# adpated from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L35C1-L46C36
class Llama3RMSNorm(torch.nn.Module):
    def __init__(self, dim, weight: torch.tensor=None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if weight is not None:
            self.weight = torch.nn.Parameter(weight)
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim).cuda())

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        return output
    