
import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.tensor,
):
    start_qm = tl.program_id(0)
    off_hz = tl.program_id(1)

    offs_m = start_qm * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    for start_n in tl.range(0, start_qm + 1)


class _triton_flash_mha_v1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal, sm_scale):
        assert q.shape[0] == k.shape[0] == v.shape[0], f'q.shape[0]: {q.shape[0]}, k.shape[0]: {k.shape[0]}, v.shape[0]: {v.shape[0]}'
        assert q.shape[1] == k.shape[1] == v.shape[1], f'q.shape[1]: {q.shape[1]}, k.shape[1]: {k.shape[1]}, v.shape[1]: {v.shape[1]}'
        assert q.shape[2] == k.shape[2] == v.shape[2], f'q.shape[2]: {q.shape[2]}, k.shape[2]: {k.shape[2]}, v.shape[2]: {v.shape[2]}'
        assert q.shape[3] == k.shape[3] == v.shape[3], f'q.shape[3]: {q.shape[3]}, k.shape[3]: {k.shape[3]}, v.shape[3]: {v.shape[3]}'
        BLOCK_M = 128
        BLOCK_N = 128

        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1],)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX = q.shape[2],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM_K,
            STAGE=1
        )

triton_flash_mha_v1 = _triton_flash_mha_v1.apply


if __name__ == '__main__':
    Z = 2
    H = 3
    N_CTX = 4
    HEAD_DIM = 8

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
            
            naive_o = triton_flash_mha_v1(q, k, v, causal, sm_scale)
            # naive_o.backward(dout)
            # naive_dq, q.grad = q.grad.clone(), None
            # naive_dk, k.grad = k.grad.clone(), None
            # naive_dv, v.grad = v.grad.clone(), None

            # atol = 1e-3 if dtype == torch.float16 else 1e-5
            # rtol = 1e-3 if dtype == torch.float16 else 1e-5
            # assert torch.allclose(o, naive_o, atol=atol, rtol=rtol)
            # assert torch.allclose(dq, naive_dq, atol=atol, rtol=rtol)
            # assert torch.allclose(dk, naive_dk, atol=atol, rtol=rtol)
            # assert torch.allclose(dv, naive_dv, atol=atol, rtol=rtol)