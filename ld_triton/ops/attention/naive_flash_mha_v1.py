
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
                for j in range(0, N_CTX, BLOCK_M):
                    k = K[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    v = V[z, h, j: j+BLOCK_N, :]                                     # [BLOCK_N, HEAD_DIM]
                    for i in range(0, N_CTX, BLOCK_N):
                        q = Q[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        o = O[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, HEAD_DIM]
                        l = L[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]
                        m = M[z, h, i: i+BLOCK_M, :]                                 # [BLOCK_M, 1]

                        s = torch.matmul(q, k.t()) * sm_scale                        # [BLOCK_M, BLOCK_N]
                        m_i, _ = torch.max(s, dim=-1, keepdim=True)                  # [BLOCK_M, 1]
                        p = torch.exp(s.float() - m_i).to(q.dtype)                   # [BLOCK_M, BLOCK_N]
                        l_i = torch.sum(p, dim=-1, keepdim=True)                     # [BLOCK_M, 1]

                        m_new = torch.max(m, m_i)                                    # [BLOCK_M, 1]
                        l_new = torch.exp(m - m_new) * l + torch.exp(m_i - m_new) * l_i # [BLOCK_M, 1]
                        o = (1.0 / l_new) * (l * torch.exp(m - m_new) * o +  torch.exp(m_i - m_new) * torch.matmul( p, v)) # [BLOCK_M, HEAD_DIM]

                        O[z, h, i: i+BLOCK_M] = o        # [BLOCK_M, HEAD_DIM]
                        L[z, h, i: i+BLOCK_M, :] = l_new # [BLOCK_M, 1]
                        M[z, h, i: i+BLOCK_M, :] = m_new # [BLOCK_M, 1]
                
        ctx.save_for_backward(q, k, v, o)
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        return O


naive_mha_flash_v1 = _naive_mha_flash_v1.apply


if __name__ == '__main__':
    Z = 2
    H = 3
    N_CTX = 4
    HEAD_DIM = 8

    dtype_ = [torch.float32, torch.float16]
    # dtype_ = [torch.float32]
    for dtype in dtype_:
        q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        dout = torch.randn_like(q)
        # causal_ = [False, True]
        causal_ = [False]
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
            BLOCK_M = N_CTX // 2
            BLOCK_N = N_CTX // 2
            naive_o = naive_mha_flash_v1(q, k, v, causal, sm_scale, BLOCK_M, BLOCK_N)
            # naive_o.backward(dout)
            # naive_dq, q.grad = q.grad.clone(), None
            # naive_dk, k.grad = k.grad.clone(), None
            # naive_dv, v.grad = v.grad.clone(), None
            print(f'o: {o}')
            print(f'naive_o: {naive_o}')
            atol = 1e-3 if dtype == torch.float16 else 1e-5
            rtol = 1e-3 if dtype == torch.float16 else 1e-5
            assert torch.allclose(o, naive_o, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(dq, naive_dq, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(dk, naive_dk, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            # assert torch.allclose(dv, naive_dv, atol=atol, rtol=rtol), f'Z: {Z}, H: {H}, N_CTX: {N_CTX}, HEAD_DIM: {HEAD_DIM}, causal: {causal}, dtype: {dtype}'
            
            
