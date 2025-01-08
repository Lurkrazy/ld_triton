import torch
import triton.language as tl


class _naive_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor):
        ctx.save_for_backward(A, B)
        C = torch.matmul(A, B)
        return C
    
    @staticmethod
    def backward(ctx, dC: torch.Tensor):
        # https://github.com/l1351868270/implicit_gemm.triton/blob/main/triton_kernel/matmul.md
        A, B = ctx.saved_tensors
        dA = torch.matmul(dC, B.t())
        dB = torch.matmul(A.t(), dC)
        return dA, dB
    

naive_matmul = _naive_matmul.apply
