
# forward
## 一般流程
https://github.com/meta-llama/llama3/blob/main/llama/model.py

## kvcache 可以使用理论分析
```

Transformer(
  (embed): ParallelEmbedding()
  (layers): ModuleList(
    (0): Block(
      (attn): MLA(
        (wq): ColumnParallelLinear()
        (wkv_a): Linear()
        (kv_norm): RMSNorm()
        (wkv_b): ColumnParallelLinear()
        (wo): RowParallelLinear()
      )
      (ffn): MLP(
        (w1): ColumnParallelLinear()
        (w2): RowParallelLinear()
        (w3): ColumnParallelLinear()
      )
      (attn_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
    (1-26): 26 x Block(
      (attn): MLA(
        (wq): ColumnParallelLinear()
        (wkv_a): Linear()
        (kv_norm): RMSNorm()
        (wkv_b): ColumnParallelLinear()
        (wo): RowParallelLinear()
      )
      (ffn): MoE(
        (gate): Gate()
        (experts): ModuleList(
          (0-1): 2 x Expert(
            (w1): Linear()
            (w2): Linear()
            (w3): Linear()
          )
        )
        (shared_experts): MLP(
          (w1): ColumnParallelLinear()
          (w2): RowParallelLinear()
          (w3): ColumnParallelLinear()
        )
      )
      (attn_norm): RMSNorm()
      (ffn_norm): RMSNorm()
    )
  )
  (norm): RMSNorm()
  (head): ColumnParallelLinear()
)


norm = RMSNorm(HEAD_DIM)
logits = ColumnParallelLinear(HEAD_DIM, vocab_size) [M, vocab_size]
logits.argmax(dim=-1) # M, 1]
```
## Matrix Representation

```
next_token = logits.argmax(dim=-1)
```

$Q, K, V \in R^{N \times d}$

$R = scale$

$O=softmax(QK^{T}*R)V$

$O_{i} = softmax(Q_{i}K^{T}*R_{i})V$
