
import torch


class _naive_mha_flash_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal, sm_scale, BLOCK_M=128, BLOCK_N=128):
        assert Q.shape[0] == K.shape[0] == V.shape[0], f'q.shape[0]: {Q.shape[0]}, k.shape[0]: {K.shape[0]}, v.shape[0]: {V.shape[0]}'
        assert Q.shape[1] == K.shape[1] == V.shape[1], f'q.shape[1]: {Q.shape[1]}, k.shape[1]: {K.shape[1]}, v.shape[1]: {V.shape[1]}'
        assert Q.shape[2] == K.shape[2] == V.shape[2], f'q.shape[2]: {Q.shape[2]}, k.shape[2]: {K.shape[2]}, v.shape[2]: {V.shape[2]}'
        assert Q.shape[3] == K.shape[3] == V.shape[3], f'q.shape[3]: {Q.shape[3]}, k.shape[3]: {K.shape[3]}, v.shape[3]: {V.shape[3]}'

        Z, H, N_CTX, HEAD_DIM = Q.shape
        factory_kwargs = {'device': Q.device, 'dtype': Q.dtype}
        O = torch.zeros((Z, H, N_CTX, HEAD_DIM), **factory_kwargs)                      # [Z, H, N_CTX, HEAD_DIM]
        M = torch.full((Z, H, N_CTX, 1), float('-inf'), device=Q.device, dtype=Q.dtype) # [Z, H, N_CTX, 1]
        L = torch.zeros((Z, H, N_CTX, 1), device=Q.device, dtype=Q.dtype)               # [Z, H, N_CTX, 1]
        
        for z in range(Z):
            for h in range(H):
                for j in range(0, N_CTX, BLOCK_N):
                    k = K[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    v = V[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    for i in range(0, N_CTX, BLOCK_M):
                        q = Q[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        o = O[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        l = L[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]
                        m = M[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]

                        s = torch.matmul(q, k.t()) * sm_scale                        # [BLOCK_M, BLOCK_N]
                        if causal:
                            offs_m = torch.arange(i, i + BLOCK_M, device=Q.device)[:, None]
                            offs_n = torch.arange(j, j + BLOCK_N, device=Q.device)[None, :]
                            s[offs_m < offs_n] = torch.finfo(s.dtype).min
                        m_j, _ = torch.max(s, dim=-1, keepdim=True)                  # [BLOCK_M, 1]
                        p = torch.exp(s.float() - m_j).to(q.dtype)                   # [BLOCK_M, BLOCK_N]
                        l_j = torch.sum(p, dim=-1, keepdim=True)                     # [BLOCK_M, 1]

                        m_new = torch.max(m, m_j)                                    # [BLOCK_M, 1]
                        l_new = torch.exp(m - m_new) * l + torch.exp(m_j - m_new) * l_j # [BLOCK_M, 1]
                        o = (1.0 / l_new) * (l * torch.exp(m - m_new) * o +  torch.exp(m_j - m_new) * torch.matmul( p, v)) # [BLOCK_M, HEAD_DIM]

                        O[z, h, i: i+BLOCK_M] = o        # [BLOCK_M, HEAD_DIM]
                        L[z, h, i: i+BLOCK_M, :] = l_new # [BLOCK_M, 1]
                        M[z, h, i: i+BLOCK_M, :] = m_new # [BLOCK_M, 1]

        # for j in range(0, N_CTX, BLOCK_N):
        #     k = K[:, :, j: j+BLOCK_N, :]                                     # [Z, H, BLOCK_N, HEAD_DIM]
        #     v = V[:, :, j: j+BLOCK_N, :]                                     # [Z, H, BLOCK_N, HEAD_DIM]
        #     for i in range(0, N_CTX, BLOCK_M):
        #         q = Q[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         o = O[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         l = L[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, 1]
        #         m = M[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, 1]
                
        #         s = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale        # [Z, H, BLOCK_M, BLOCK_N]
        #         if causal:
        #             offs_m = torch.arange(i, i + BLOCK_M, device=Q.device)[:, None]
        #             offs_n = torch.arange(j, j + BLOCK_N, device=Q.device)[None, :]
        #             s[:, :, offs_m < offs_n] = torch.finfo(s.dtype).min
        #         m_j, _ = torch.max(s, dim=-1, keepdim=True)                  # [Z, H, BLOCK_M, 1]
        #         p = torch.exp(s.float() - m_j).to(q.dtype)                   # [Z, H, BLOCK_M, BLOCK_N]
        #         l_j = torch.sum(p, dim=-1, keepdim=True)                     # [Z, H, BLOCK_M, 1]

        #         m_new = torch.max(m, m_j)                                    # [Z, H, BLOCK_M, 1]
        #         l_new = torch.exp(m - m_new) * l + torch.exp(m_j - m_new) * l_j # [Z, H, BLOCK_M, 1]
        #         o = (1.0 / l_new) * (l * torch.exp(m - m_new) * o +  torch.exp(m_j - m_new) * torch.matmul( p, v)) # [Z, H, BLOCK_M, HEAD_DIM]

        #         O[:, :, i: i+BLOCK_M] = o        # [Z, H, BLOCK_M, HEAD_DIM]
        #         L[:, :, i: i+BLOCK_M, :] = l_new # [Z, H, BLOCK_M, 1]
        #         M[:, :, i: i+BLOCK_M, :] = m_new # [Z, H, BLOCK_M, 1]
        
        ctx.save_for_backward(Q, K, V, O, M, L)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M, L = ctx.saved_tensors
        Z, H, N_CTX, HEAD_DIM = Q.shape
        causal = ctx.causal
        sm_scale = ctx.sm_scale
        BLOCK_M = ctx.BLOCK_M
        BLOCK_N = ctx.BLOCK_N
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        for z in range(Z):
            for h in range(H):
                for j in range(0, N_CTX, BLOCK_N):
                    k = K[z, h, j: j+BLOCK_N, :]                                       # [BLOCK_N, HEAD_DIM]
                    v = V[z, h, j: j+BLOCK_N, :]                                       # [BLOCK_N, HEAD_DIM]
                    dk = dK[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    dv = dV[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    for i in range(0, N_CTX, BLOCK_M):
                        q  =  Q[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        o  =  O[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        do = dO[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        dq = dQ[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        l  =  L[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]
                        m  =  M[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]

                        s = torch.matmul(q, k.t()) * sm_scale                          # [BLOCK_M, BLOCK_N]
                        if causal:
                            offs_m = torch.arange(i, i + BLOCK_M, device=Q.device)[:, None]
                            offs_n = torch.arange(j, j + BLOCK_N, device=Q.device)[None, :]
                            s[offs_m < offs_n] = torch.finfo(s.dtype).min
                        p = (1.0 / l) * torch.exp(s.float() - m).to(q.dtype)           # [BLOCK_M, BLOCK_N]
                        dp = torch.matmul(do, v.t())                                   # [BLOCK_M, BLOCK_N]
                        d = torch.sum(do * o, dim=-1, keepdim=True)                    # [BLOCK_M, 1]
                        ds = p * (dp - d) * sm_scale                                   # [BLOCK_M, BLOCK_N]
                        dq = dq + torch.matmul(ds, k)                                  # [BLOCK_M, HEAD_DIM]
                        dk = dk + torch.matmul(ds.t(), q)                              # [BLOCK_N, HEAD_DIM]
                        dv = dv + torch.matmul(p.t(), do)                              # [BLOCK_N, HEAD_DIM]

                        dQ[z, h, i: i+BLOCK_M, :] = dq                                 # [BLOCK_M, HEAD_DIM]
                        dK[z, h, j: j+BLOCK_N, :] = dk                                 # [BLOCK_N, HEAD_DIM]
                        dV[z, h, j: j+BLOCK_N, :] = dv                                 # [BLOCK_N, HEAD_DIM]

        # for j in range(0, N_CTX, BLOCK_N):
        #     k = K[:, :, j: j+BLOCK_N, :]                                       # [Z, H, BLOCK_N, HEAD_DIM]
        #     v = V[:, :, j: j+BLOCK_N, :]                                       # [Z, H, BLOCK_N, HEAD_DIM]
        #     dk = dK[:, :, j: j+BLOCK_N, :]                                     # [Z, H, BLOCK_N, HEAD_DIM]
        #     dv = dV[:, :, j: j+BLOCK_N, :]                                     # [Z, H, BLOCK_N, HEAD_DIM]
        #     for i in range(0, N_CTX, BLOCK_M):
        #         q  =  Q[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         o  =  O[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         do = dO[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         dq = dQ[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         l  =  L[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, 1]
        #         m  =  M[:, :, i: i+BLOCK_M, :]                                 # [Z, H, BLOCK_M, 1]

        #         s = torch.matmul(q, k.permute(0, 1, 3, 2)) * sm_scale          # [Z, H, BLOCK_M, BLOCK_N]
        #         if causal:
        #             offs_m = torch.arange(i, i + BLOCK_M, device=Q.device)[:, None]
        #             offs_n = torch.arange(j, j + BLOCK_N, device=Q.device)[None, :]
        #             s[:, :, offs_m < offs_n] = torch.finfo(s.dtype).min
        #         p = (1.0 / l) * torch.exp(s.float() - m).to(q.dtype)           # [Z, H, BLOCK_M, BLOCK_N]
        #         dp = torch.matmul(do, v.permute(0, 1, 3, 2))                   # [Z, H, BLOCK_M, BLOCK_N]
        #         d = torch.sum(do * o, dim=-1, keepdim=True)                    # [Z, H, BLOCK_M, 1]
        #         ds = p * (dp - d) * sm_scale                                   # [Z, H, BLOCK_M, BLOCK_N]
        #         dq = dq + torch.matmul(ds, k)                                  # [Z, H, BLOCK_M, HEAD_DIM]
        #         dk = dk + torch.matmul(ds.permute(0, 1, 3, 2), q)              # [Z, H, BLOCK_N, HEAD_DIM]
        #         dv = dv + torch.matmul(p.permute(0, 1, 3, 2), do)              # [Z, H, BLOCK_N, HEAD_DIM]

        #         dQ[:, :, i: i+BLOCK_M, :] = dq                                 # [Z, H, BLOCK_M, HEAD_DIM]
        #         dK[:, :, j: j+BLOCK_N, :] = dk                                 # [Z, H, BLOCK_N, HEAD_DIM]
        #         dV[:, :, j: j+BLOCK_N, :] = dv                                 # [Z, H, BLOCK_N, HEAD_DIM]

        return dQ, dK, dV, None, None, None, None


naive_mha_flash_v1 = _naive_mha_flash_v1.apply


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
    for dtype in dtype_:
        q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        dout = torch.randn_like(q)
        causal_ = [False, True]
        sm_scale = 0.5
        for causal in causal_:
            p: torch.Tensor = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if causal:
                M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
                p[:, :, M == 0] = float('-inf')
            p = torch.softmax(p.float(), dim=-1).to(dtype)
            o = torch.matmul(p, v)
            o.backward(dout)
            dq, q.grad = q.grad.clone(), None
            dk, k.grad = k.grad.clone(), None
            dv, v.grad = v.grad.clone(), None

            naive_o = naive_mha_flash_v1(q, k, v, causal, sm_scale, BLOCK_M, BLOCK_N)
            naive_o.backward(dout)
            naive_dq, q.grad = q.grad.clone(), None
            naive_dk, k.grad = k.grad.clone(), None
            naive_dv, v.grad = v.grad.clone(), None
            
            atol = 1e-2
            rtol = 1e-2
            assert torch.allclose(o, naive_o, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            assert torch.allclose(dq, naive_dq, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            assert torch.allclose(dk, naive_dk, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            assert torch.allclose(dv, naive_dv, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            
            
