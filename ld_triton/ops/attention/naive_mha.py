
import torch


class _naive_mha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal, sm_scale):
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[1] == k.shape[1] == v.shape[1]
        assert q.shape[2] == k.shape[2] == v.shape[2]
        assert q.shape[3] == k.shape[3] == v.shape[3]
        N_CTX = q.shape[2]
        p: torch.Tensor = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
            p[:, :, M == 0] = float('-inf')
        p = torch.softmax(p.float(), dim=-1)
        o = torch.matmul(p, v)
        ctx.save_for_backward(q, k, v)
        ctx.sm_scale = sm_scale
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        N_CTX = q.shape[2]
        dq = None

        # # element-wise
        # if q.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = q.shape
        #     dq = torch.zeros_like(q)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
        #     for z in range(Z):
        #         for h in range(H):
        #             for i in range(N_CTX):
        #                 for j in range(HEAD_DIM):
        #                     res = 0.0
        #                     for w in range(N_CTX):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += sqk[z, h, i, x] * dov[z, h, i, x]
        #                         res += sqk[z, h, i, w] * (dov[z, h, i, w] - d) * k[z, h, w, j] * sm_scale
        #                     dq[z, h, i, j] = res

        # # row-wise 0
        # if q.requires_grad:
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
            
        #     dq = torch.zeros_like(q)
        #     for z in range(Z):
        #         for h in range(H):
        #             for i in range(N_CTX):
        #                 for j in range(HEAD_DIM):
        #                     d = 0.0
        #                     for x in range(N_CTX):
        #                         d += sqk[z, h, i, x] * dov[z, h, i, x]
        #                     dq[z, h, i, j] = torch.sum(sqk[z, h, i, :] * (dov[z, h, i, :] - d) * k[z, h, :, j]) * sm_scale

        # # row-wise 1
        # if q.requires_grad:
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
            
        #     dq = torch.zeros_like(q)
        #     for z in range(Z):
        #         for h in range(H):
        #             for i in range(N_CTX):
        #                 diag = torch.zeros(HEAD_DIM, device=q.device, dtype=q.dtype)
        #                 diag[i] = torch.sum(sqk[z, h, i, :] * dov[z, h, i, :])
        #                 dq[z, h, i, :] = torch.matmul(sqk[z, h, i, :] * (dov[z, h, i, :] - diag[i]), k[z, h, :, :]) * sm_scale

        # # matrix-wise 0
        # if q.requires_grad:
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
            
        #     dq = torch.zeros_like(q)
        #     for z in range(Z):
        #         for h in range(H):
        #             diag = torch.sum(sqk[z, h, :, :] * dov[z, h, :, :], dim=-1, keepdim=True)
        #             dq[z, h, :, :] = torch.matmul(sqk[z, h, :, :] * (dov[z, h, :, :] - diag), k[z, h, :, :]) * sm_scale

        # matrix-wise 1
        if q.requires_grad:
            k_t = k.transpose(2, 3)
            v_t = v.transpose(2, 3)
            qk = torch.matmul(q, k_t) * sm_scale
            sqk = torch.softmax(qk.float(), dim=-1)
            dov = torch.matmul(do, v_t)
            
            diag = torch.sum(sqk * dov, dim=-1, keepdim=True)
            dq = torch.matmul(sqk * (dov - diag), k) * sm_scale

        dk = None
        # # element-wise
        # if k.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = k.shape
        #     dk = torch.zeros_like(k)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
        #     for z in range(Z):
        #         for h in range(H):
        #             for i in range(N_CTX):
        #                 for j in range(HEAD_DIM):
        #                     res = 0.0
        #                     for a in range(N_CTX):
        #                         d = 0.0
        #                         for x in range(N_CTX):
        #                             d += sqk[z, h, a, x] * dov[z, h, a, x]
        #                         res += sqk[z, h, a, i] * (dov[z, h, a, i] - d) * q[z, h, a, j] * sm_scale
        #                     dk[z, h, i, j] = res

        # # row-wise 0
        # if k.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = k.shape
        #     dk = torch.zeros_like(k)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
        #     for z in range(Z):
        #         for h in range(H):
        #             d  = torch.sum(sqk[z, h, :, :] * dov[z, h, :, :], dim=-1)
        #             for i in range(N_CTX):
        #                 for j in range(HEAD_DIM):
        #                     res = torch.sum(sqk[z, h, :, i] * (dov[z, h, :, i] - d) * q[z, h, :, j]) * sm_scale
        #                     dk[z, h, i, j] = res

        # # row-wise 1
        # if k.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = k.shape
        #     dk = torch.zeros_like(k)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
        #     for z in range(Z):
        #         for h in range(H):
        #             d  = torch.sum(sqk[z, h, :, :] * dov[z, h, :, :], dim=-1)
        #             for i in range(N_CTX):
        #                 res = torch.matmul((sqk[z, h, :, i] * (dov[z, h, :, i] - d)).t(), q[z, h, :, :]) * sm_scale
        #                 dk[z, h, i, :] = res

        # # matrix-wise 0
        # if k.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = k.shape
        #     dk = torch.zeros_like(k)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     dov = torch.matmul(do, v_t)
        #     for z in range(Z):
        #         for h in range(H):
        #             d  = torch.sum(sqk[z, h, :, :] * dov[z, h, :, :], dim=-1, keepdim=True)
        #             res = torch.matmul((sqk[z, h, :, :] * (dov[z, h, :, :] - d)).t(), q[z, h, :, :]) * sm_scale
        #             dk[z, h, :, :] = res

        # matrix-wise 1
        if k.requires_grad:
            Z, H, N_CTX, HEAD_DIM = k.shape
            dk = torch.zeros_like(k)
            k_t = k.transpose(2, 3)
            v_t = v.transpose(2, 3)
            qk = torch.matmul(q, k_t) * sm_scale
            sqk = torch.softmax(qk.float(), dim=-1)
            dov = torch.matmul(do, v_t)

            d  = torch.sum(sqk * dov, dim=-1, keepdim=True)
            res = torch.matmul((sqk * (dov - d)).permute(0, 1, 3, 2), q) * sm_scale
            dk = res

        dv = None
        # # element-wise
        # if v.requires_grad:
        #     Z, H, N_CTX, HEAD_DIM = v.shape
        #     dv = torch.zeros_like(v)
        #     k_t = k.transpose(2, 3)
        #     v_t = v.transpose(2, 3)
        #     qk = torch.matmul(q, k_t) * sm_scale
        #     sqk = torch.softmax(qk.float(), dim=-1)
        #     for z in range(Z):
        #         for h in range(H):
        #             for i in range(N_CTX):
        #                 for j in range(HEAD_DIM):
        #                     res = 0.0
        #                     for a in range(N_CTX):
        #                         res += sqk[z, h, a, i] * do[z, h, a, j]
        #                     dv[z, h, i, j] = res

        # matrix-wise
        if v.requires_grad:
            Z, H, N_CTX, HEAD_DIM = v.shape
            k_t = k.transpose(2, 3)
            v_t = v.transpose(2, 3)
            qk = torch.matmul(q, k_t) * sm_scale
            sqk = torch.softmax(qk.float(), dim=-1)
            dv = torch.matmul(sqk.permute(0, 1, 3, 2), do)
        return dq, dk, dv, None, None


naive_mha = _naive_mha.apply


if __name__ == '__main__':
    Z = 2
    H = 3
    N_CTX = 4
    HEAD_DIM = 8

    dtype_ = [torch.float32]
    for dtype in dtype_:
        q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        dout = torch.randn_like(q)
        causal_ = [False]
        sm_scale = 1.0
        for causal in causal_:
            p: torch.Tensor = torch.matmul(q, k.transpose(2, 3)) * sm_scale
            if causal:
                M = torch.tril(torch.ones((N_CTX, N_CTX), device=q.device))
                p[:, :, M == 0] = float('-inf')
            p = torch.softmax(p.float(), dim=-1)
            o = torch.matmul(p, v)
            o.backward(dout)
            dq, q.grad = q.grad.clone(), None
            dk, k.grad = k.grad.clone(), None
            dv, v.grad = v.grad.clone(), None
            
            naive_o = naive_mha(q, k, v, causal, sm_scale)
            naive_o.backward(dout)
            naive_dq, q.grad = q.grad.clone(), None
            naive_dk, k.grad = k.grad.clone(), None
            naive_dv, v.grad = v.grad.clone(), None

            assert torch.allclose(o, naive_o, atol=1e-5, rtol=1e-5)
            assert torch.allclose(dq, naive_dq, atol=1e-5, rtol=1e-5)
            assert torch.allclose(dk, naive_dk, atol=1e-5, rtol=1e-5)
            assert torch.allclose(dv, naive_dv, atol=1e-5, rtol=1e-5)
            
            
