
import torch


class _naive_mha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal, sm_scale):
        assert q.shape[0] == k.shape[0] == v.shape[0], f'q.shape[0]: {q.shape[0]}, k.shape[0]: {k.shape[0]}, v.shape[0]: {v.shape[0]}'
        assert q.shape[1] == k.shape[1] == v.shape[1], f'q.shape[1]: {q.shape[1]}, k.shape[1]: {k.shape[1]}, v.shape[1]: {v.shape[1]}'
        assert q.shape[2] == k.shape[2] == v.shape[2], f'q.shape[2]: {q.shape[2]}, k.shape[2]: {k.shape[2]}, v.shape[2]: {v.shape[2]}'
        assert q.shape[3] == k.shape[3] == v.shape[3], f'q.shape[3]: {q.shape[3]}, k.shape[3]: {k.shape[3]}, v.shape[3]: {v.shape[3]}'
        N_CTX = q.shape[2]
        p: torch.Tensor = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale # [Z, H, N_CTX, N_CTX]
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))     # [N_CTX, N_CTX]
            p[:, :, M == 0] = float('-inf')                                 # [Z, H, N_CTX, N_CTX]
        p = torch.softmax(p.float(), dim=-1).to(q.dtype) # [Z, H, N_CTX, HEAD_DIM]
        o = torch.matmul(p, v)                           # [Z, H, N_CTX, N_CTX]
        ctx.save_for_backward(q, k, v, o)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, o, = ctx.saved_tensors
        causal = ctx.causal
        sm_scale = ctx.sm_scale

 # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
        # # element-wise 0
        # Z, H, N_CTX, HEAD_DIM = q.shape
        # dq = torch.zeros_like(q)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        # for z in range(Z):
        #     for h in range(H):
        #         for i in range(N_CTX):
        #             for j in range(HEAD_DIM):
        #                 res = 0.0
        #                 if causal:
        #                     for w in range(i + 1):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += p[z, h, i, x] * dp[z, h, i, x]
        #                         res += p[z, h, i, w] * (dp[z, h, i, w] - d) * k[z, h, w, j] * sm_scale
        #                 else:
        #                     for w in range(N_CTX):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += p[z, h, i, x] * dp[z, h, i, x]
        #                         res += p[z, h, i, w] * (dp[z, h, i, w] - d) * k[z, h, w, j] * sm_scale
        #                 dq[z, h, i, j] = res

        # # row-wise 0
        # Z, H, N_CTX, HEAD_DIM = q.shape
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        
        # dq = torch.zeros_like(q)
        # for z in range(Z):
        #     for h in range(H):
        #         for i in range(N_CTX):
        #             for j in range(HEAD_DIM):
        #                 d = 0.0
        #                 for x in range(N_CTX):
        #                     d += p[z, h, i, x] * dp[z, h, i, x]
        #                 dq[z, h, i, j] = torch.sum(p[z, h, i, :] * (dp[z, h, i, :] - d) * k[z, h, :, j]) * sm_scale

        # # row-wise 1
        # Z, H, N_CTX, HEAD_DIM = q.shape
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        
        # dq = torch.zeros_like(q)
        # for z in range(Z):
        #     for h in range(H):
        #         for i in range(N_CTX):
        #             d = torch.zeros(HEAD_DIM, device=q.device, dtype=q.dtype)
        #             d[i] = torch.sum(p[z, h, i, :] * dp[z, h, i, :])
        #             dq[z, h, i, :] = torch.matmul(p[z, h, i, :] * (dp[z, h, i, :] - d[i]), k[z, h, :, :]) * sm_scale

        # # matrix-wise 0
        # Z, H, N_CTX, HEAD_DIM = q.shape
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        
        # dq = torch.zeros_like(q)
        # for z in range(Z):
        #     for h in range(H):
        #         d = torch.sum(p[z, h, :, :] * dp[z, h, :, :], dim=-1, keepdim=True)
        #         dq[z, h, :, :] = torch.matmul(p[z, h, :, :] * (dp[z, h, :, :] - d), k[z, h, :, :]) * sm_scale

        # matrix-wise 1
        N_CTX = q.shape[2]
        p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale           # [Z, H, N_CTX, N_CTX]
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device)) # [N_CTX, N_CTX]
            p[:, :, M == 0] = float('-inf')                             # [Z, H, N_CTX, N_CTX]
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)                # [Z, H, N_CTX, N_CTX]
        dp = torch.matmul(do, v.permute(0, 1, 3, 2))                    # [Z, H, N_CTX, N_CTX]
        
        d = torch.sum(o * do, dim=-1, keepdim=True)                     # [Z, H, N_CTX, 1]
        # or, because p * dp == o * do
        # d = torch.sum(p * dp, dim=-1, keepdim=True)                     # [Z, H, N_CTX, 1]
        dq = torch.matmul(p * (dp - d), k) * sm_scale                   # [Z, H, N_CTX, N_CTX]

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # # element-wise
        # Z, H, N_CTX, HEAD_DIM = k.shape
        # dk = torch.zeros_like(k)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(k.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        # for z in range(Z):
        #     for h in range(H):
        #         for i in range(N_CTX):
        #             for j in range(HEAD_DIM):
        #                 res = 0.0
        #                 if causal:
        #                     for a in range(i, N_CTX):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += p[z, h, a, x] * dp[z, h, a, x]
        #                         res += p[z, h, a, i] * (dp[z, h, a, i] - d) * q[z, h, a, j] * sm_scale
        #                 else:
        #                     for a in range(N_CTX):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += p[z, h, a, x] * dp[z, h, a, x]
        #                         res += p[z, h, a, i] * (dp[z, h, a, i] - d) * q[z, h, a, j] * sm_scale
        #                 dk[z, h, i, j] = res

        # # row-wise 0
        # Z, H, N_CTX, HEAD_DIM = k.shape
        # dk = torch.zeros_like(k)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(k.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        # for z in range(Z):
        #     for h in range(H):
        #         d  = torch.sum(p[z, h, :, :] * dp[z, h, :, :], dim=-1)
        #         for i in range(N_CTX):
        #             for j in range(HEAD_DIM):
        #                 res = torch.sum(p[z, h, :, i] * (dp[z, h, :, i] - d) * q[z, h, :, j]) * sm_scale
        #                 dk[z, h, i, j] = res

        # # row-wise 1
        # Z, H, N_CTX, HEAD_DIM = k.shape
        # dk = torch.zeros_like(k)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(k.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        # for z in range(Z):
        #     for h in range(H):
        #         d  = torch.sum(p[z, h, :, :] * dp[z, h, :, :], dim=-1)
        #         for i in range(N_CTX):
        #             res = torch.matmul((p[z, h, :, i] * (dp[z, h, :, i] - d)).t(), q[z, h, :, :]) * sm_scale
        #             dk[z, h, i, :] = res

        # # matrix-wise 0
        # Z, H, N_CTX, HEAD_DIM = k.shape
        # dk = torch.zeros_like(k)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(k.dtype)
        # dp = torch.matmul(do, v.permute(0, 1, 3, 2))
        # for z in range(Z):
        #     for h in range(H):
        #         d  = torch.sum(p[z, h, :, :] * dp[z, h, :, :], dim=-1, keepdim=True)
        #         res = torch.matmul((p[z, h, :, :] * (dp[z, h, :, :] - d)).t(), q[z, h, :, :]) * sm_scale
        #         dk[z, h, :, :] = res

        # matrix-wise 1
        Z, H, N_CTX, HEAD_DIM = k.shape
        p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale           # [Z, H, N_CTX, N_CTX]
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device)) # [Z, H, N_CTX, N_CTX]
            p[:, :, M == 0] = float('-inf')                             # [Z, H, N_CTX, N_CTX]
        p = torch.softmax(p.float(), dim=-1).to(k.dtype)                # [Z, H, N_CTX, N_CTX]
        dp = torch.matmul(do, v.permute(0, 1, 3, 2))                    # [Z, H, N_CTX, N_CTX]


        d =  torch.sum(o * do, dim=-1, keepdim=True)                    # [Z, H, N_CTX, 1]
        # or, because p * dp == o * do
        # d  = torch.sum(p * dp, dim=-1, keepdim=True)                    # [Z, H, N_CTX, 1]
        
        dk = torch.matmul((p * (dp - d)).permute(0, 1, 3, 2), q) * sm_scale # [Z, H, N_CTX, N_CTX]

 # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # # element-wise
        # Z, H, N_CTX, HEAD_DIM = v.shape
        # dv = torch.zeros_like(v)
        # p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale
        # if causal:
        #     M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
        #     p[:, :, M == 0] = float('-inf')
        # p = torch.softmax(p.float(), dim=-1).to(v.dtype)

        # for z in range(Z):
        #     for h in range(H):
        #         for i in range(N_CTX):
        #             for j in range(HEAD_DIM):
        #                 res = 0.0
        #                 if causal:
        #                     for a in range(i, N_CTX):
        #                         res += p[z, h, a, i] * do[z, h, a, j]
        #                 else:
        #                     for a in range(N_CTX):
        #                         res += p[z, h, a, i] * do[z, h, a, j]
        #                 dv[z, h, i, j] = res

        # matrix-wise
        Z, H, N_CTX, HEAD_DIM = v.shape
        p = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale           # [Z, H, N_CTX, N_CTX]
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device)) # [N_CTX, N_CTX]
            p[:, :, M == 0] = float('-inf')                             # [Z, H, N_CTX, N_CTX]
        p = torch.softmax(p.float(), dim=-1).to(v.dtype)                # [Z, H, N_CTX, N_CTX]
        dv = torch.matmul(p.permute(0, 1, 3, 2), do)                    # [Z, H, N_CTX, HEAD_DIM]

        return dq, dk, dv, None, None


naive_mha = _naive_mha.apply


if __name__ == '__main__':
    # Z = 1
    # H = 1
    # N_CTX = 4
    # HEAD_DIM = 8
    # BLOCK_M = 2
    # BLOCK_N = 2

    Z = 2
    H = 3
    N_CTX = 1024
    HEAD_DIM = 32
    BLOCK_M = 128
    BLOCK_N = 128

    dtype_ = [torch.float32, torch.float16]
    dtype_ = [torch.float32]
    for dtype in dtype_:
        q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        dout = torch.randn_like(q)
        causal_ = [False, True]
        causal_ = [False]
        sm_scale = 0.5
        for causal in causal_:
            torch_k = k.view(Z, 1, N_CTX, HEAD_DIM).expand(Z, H, N_CTX, HEAD_DIM)
            torch_v = v.view(Z, 1, N_CTX, HEAD_DIM).expand(Z, H, N_CTX, HEAD_DIM)
            torch_p: torch.Tensor = torch.matmul(q, torch_k.transpose(2, 3)) * sm_scale
            if causal:
                M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
                torch_p[:, :, M == 0] = float('-inf')
            torch_p = torch.softmax(torch_p.float(), dim=-1).to(dtype)
            torch_o_0 = torch.matmul(torch_p, torch_v)
            torch_o_0.backward(dout)
            torch_dq_0, q.grad = q.grad.clone(), None
            torch_dk_0, k.grad = k.grad.clone(), None
            torch_dv_0, v.grad = v.grad.clone(), None
            print(f'torch_dv_0: {torch_dk_0}')
            
            torch_o = torch.zeros_like(q)
            for h in range(H):
                p = torch.matmul(q[:, h, :, :], k.transpose(1, 2)) * sm_scale
                if causal:
                    M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
                    p[:, M == 0] = float('-inf')
                p = torch.softmax(p.float(), dim=-1).to(dtype)
                o = torch.matmul(p, v)
                torch_o[:, h, :, :] = o
            torch_o.backward(dout)
            dq, dk, dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
            print(f'dk: {dk}')

            naive_o = naive_mha(q, k, v, causal, sm_scale)
            # naive_o.backward(dout)
            # naive_dq, q.grad = q.grad.clone(), None
            # naive_dk, k.grad = k.grad.clone(), None
            # naive_dv, v.grad = v.grad.clone(), None
            # atol = 1e-2 if dtype == torch.float16 else 1e-3
            # rtol = 1e-2 if dtype == torch.float16 else 1e-3
            # assert torch.allclose(torch_o, naive_o, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(torch_dq, naive_dq, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(torch_dk, naive_dk, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(torch_dv, naive_dv, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            
            