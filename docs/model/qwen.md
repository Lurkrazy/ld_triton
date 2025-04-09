
# A Simple Model of Memory

## All data initially in slow memory
$m =$ number of memory elements (words) moved between fast and slow memory

$t_m =$ time per slow memory operation (inverse bandwidth in best case)

$f =$ number of arithmetic operations

$t_f =$ time per arithmetic operation << tm

$CI = \frac{f}{m}$  average number of flops per slow memory access

## Actual time

$f * t_f + m * t_m = f * t_f * (1 + \frac{t_m}{t_f}  * \frac{1}{CI})$

# Model FLOPs Utilization (MFU)
<p>
$SEQ\_LEN = 2048$
</p>

<p>
$GBS = global\_batch\_size$
</p>

## embed_tokens(Embedding)

<p>
$f\_forward = 0$
</p>

<p>
$f\_backward = o(0)$
</p>

<p>
$weight\_shape = (vocab\_size, hidden\_size)$
</p>


## rotary_emb(Qwen2RotaryEmbedding)

<p>
$f\_forward = GBS * SEQ\_LEN * 3$
</p>

# layers
[Qwen2DecoderLayer for _ in range(num_hidden_layers )]
## Qwen2DecoderLayer
### self_attn(Qwen2Attention)
#### q_proj(Linear)

<p>
$weight\_shape = (hidden\_size, num\_attention\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_attention\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, num\_attention\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * (num\_attention\_heads * head\_dim) + GBS * SEQ\_LEN * (num\_attention\_heads * head\_dim)$
</p>

#### k_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * SEQ\_LEN * (num\_key\_value\_heads * head\_dim)$
</p>

#### v_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * SEQ\_LEN * (num\_key\_value\_heads * head\_dim)$
</p>

#### o_proj(Linear)
<p>
$weight\_shape = (num\_attention\_heads * head\_dim, hidden\_size)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, num\_attention\_heads * head\_dim)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

#### attention_interface
<p>
$q\_shape = (GBS, num\_attention\_heads, SEQ\_LEN, head\_dim)$
</p>

<p>
$k\_shape = (GBS, num\_key\_value\_heads, SEQ\_LEN, head\_dim) -> (GBS, num\_attention\_heads, SEQ\_LEN, head\_dim)$
</p>

<p>
$v\_shape = (GBS, num\_key\_value\_heads, SEQ\_LEN, head\_dim) -> (GBS, num\_attention\_heads, SEQ\_LEN, head\_dim)$
</p>

<p>
$qk: 2 * GBS * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim$
</p>

<p>
$scaling: GBS * num\_attention\_heads * SEQ\_LEN^{2}$
</p>

<p>
$softmax: GBS * num\_attention\_heads * SEQ\_LEN^{2}$
</p>

<p>
$pv: 2 * GBS * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim$
</p>

<p>
$total: 4 * GBS * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim + 2 * GBS * num\_attention\_heads * SEQ\_LEN^{2}$
</p>

<p>
$Qwen2Attention total: 4 * GBS * SEQ\_LEN * hidden\_size * (num\_attention\_heads * head\_dim) + 4 * GBS * SEQ\_LEN * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * SEQ\_LEN * (num\_attention\_heads * head\_dim) + 2 * GBS * SEQ\_LEN * (num\_key\_value\_heads * head\_dim) + 4 * GBS * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim + 2 * GBS * num\_attention\_heads * SEQ\_LEN^{2}$
</p>

### rotary_emb
<p>
$f\_forward = GBS * SEQ\_LEN * 3$
</p>

### mlp(Qwen2MLP)
#### gate_proj(Linear)
<p>
$weight\_shape = (hidden\_size, intermediate\_size)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, intermediate\_size)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * intermediate\_size$
</p>

#### up_proj(Linear)

<p>
$weight\_shape = (hidden\_size, intermediate\_size)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, intermediate\_size)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * intermediate\_size$
</p>

#### act_fn(silu)

<p>
$silu(x) = \frac{x}{1 + e^{-x}}$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, intermediate\_size)$
</p>

<p>
$f\_forward = 3 * GBS * SEQ\_LEN * intermediate\_size$
</p>

#### down_proj(Linear)
<p>
$weight\_shape = (intermediate\_size, hidden\_size)$
</p>

<p>
$input\_shape = (GBS, SEQ\_LEN, intermediate\_size)$
</p>

<p>
$output\_shape = (GBS, SEQ\_LEN, hidden\_size)$
</p>

<p>
$f\_forward = 2 * GBS * SEQ\_LEN * hidden\_size * intermediate\_size$
</p>

<p>
$mlp total: 6 * GBS * SEQ\_LEN * hidden\_size * intermediate\_size + 3 * GBS * SEQ\_LEN * intermediate\_size$
</p>

### input_layernorm(Qwen2RMSNorm)
<p>
$y_{i} = \frac{x_{i}}{RMS(x)} * \gamma_{i}$
</p>

<p>
$RMS(x) = \sqrt {\epsilon + \frac{1}{n}\sum_{i=0}^{n-1}x_{i}^2}$
</p>

<p>
$f\_forward = 4 * GBS * SEQ\_LEN * hidden\_size$
</p>

## Total
只考虑attention, linear(不包含bias), matmul
### self_attn

<p>
$q\_proj: 2 * GBS * SEQ\_LEN * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$k\_proj: 2 * GBS * SEQ\_LEN * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$v\_proj: 2 * GBS * SEQ\_LEN * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$o\_proj: 2 * GBS * SEQ\_LEN * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$qk: 2 * GBS * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim$
</p>

<p>
$pv: 2 * GBS  * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim$
</p>

<p>
$self\_atten\_total: 4 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim) + 4 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim) + 4 * GBS  * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN^{2} * head\_dim$
</p>

### mlp
<p>
$gate\_proj: 2 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$up\_proj: 2 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$down\_proj: 2 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$mlp\_total: 6 * GBS * SEQ\_LEN  * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

# kv cache
# prefill 阶段同forward
# encoder 阶段
## self_attn(Qwen2Attention)
### q_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_attention\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_attention\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, 1, num\_attention\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * hidden\_size * (num\_attention\_heads * head\_dim) + GBS * (num\_attention\_heads * head\_dim)$
</p>

### k_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, 1, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * (num\_key\_value\_heads * head\_dim)$
</p>

### v_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, 1, num\_key\_value\_heads * head\_dim)$
</p>

<p>
$f\_forward = 2 * GBS * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * (num\_key\_value\_heads * head\_dim)$
</p>

### o_proj(Linear)
<p>
$weight\_shape = (num\_attention\_heads * head\_dim, hidden\_size)$
</p>

<p>
$input\_shape = (GBS, 1, num\_attention\_heads * head\_dim)$
</p>

<p>
$output\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$f\_forward = 2 * GBS * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

### attention_interface
<p>
$q\_shape = (GBS, num\_attention\_heads, 1, head\_dim)$
</p>

<p>
$k\_shape = (GBS, num\_key\_value\_heads, SEQ\_LEN, head\_dim) -> (GBS, num\_attention\_heads, SEQ\_LEN, head\_dim)$
</p>

<p>
$v\_shape = (GBS, num\_key\_value\_heads, SEQ\_LEN, head\_dim) -> (GBS, num\_attention\_heads, SEQ\_LEN, head\_dim)$
</p>

<p>
$qk: 2 * GBS * SEQ\_LEN * num\_attention\_heads * head\_dim$
</p>

<p>
$scaling: GBS * SEQ\_LEN * num\_attention\_heads$
</p>

<p>
$softmax: GBS * SEQ\_LEN * num\_attention\_heads$
</p>

<p>
$pv: 2 * GBS  * SEQ\_LEN * num\_attention\_heads * head\_dim$
</p>

<p>
$total: 4 * GBS * SEQ\_LEN * num\_attention\_heads * head\_dim + 2 * GBS * SEQ\_LEN * num\_attention\_heads$
</p>

<p>
$Qwen2Attention total: 4 * GBS * hidden\_size * (num\_attention\_heads * head\_dim) + 4 * GBS * hidden\_size * (num\_key\_value\_heads * head\_dim) + GBS * (num\_attention\_heads * head\_dim) + 2 * GBS * (num\_key\_value\_heads * head\_dim) + 4 * GBS * SEQ\_LEN * num\_attention\_heads * head\_dim + 2 * GBS * SEQ\_LEN * num\_attention\_heads$
</p>

## mlp(Qwen2MLP)
### gate_proj(Linear)
<p>
$weight\_shape = (hidden\_size, intermediate\_size)$
</p>

<p>
$input\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, 1, intermediate\_size)$
</p>

<p>
$f\_forward = 2 * GBS * 1 * hidden\_size * intermediate\_size$
</p>

### up_proj(Linear)

<p>
$weight\_shape = (hidden\_size, intermediate\_size)$
</p>

<p>
$input\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$output\_shape = (GBS, 1, intermediate\_size)$
</p>

<p>
$f\_forward = 2 * GBS * 1 * hidden\_size * intermediate\_size$
</p>

### down_proj(Linear)
<p>
$weight\_shape = (intermediate\_size, hidden\_size)$
</p>

<p>
$input\_shape = (GBS, 1, intermediate\_size)$
</p>

<p>
$output\_shape = (GBS, 1, hidden\_size)$
</p>

<p>
$f\_forward = 2 * GBS * 1 * hidden\_size * intermediate\_size$
</p>

<p>
$mlp total: 6 * GBS * hidden\_size * intermediate\_size + 3 * GBS * intermediate\_size$
</p>

## Total
只考虑attention, linear(不包含bias), matmul
### self_attn

<p>
$q\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$k\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$v\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$o\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$qk: 2 * GBS * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN * head\_dim$
</p>

<p>
$pv: 2 * GBS  * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN * head\_dim$
</p>

<p>
$self\_atten\_total: 4 * GBS * num\_hidden\_layers * hidden\_size * (num\_attention\_heads * head\_dim) + 4 * GBS * num\_hidden\_layers * hidden\_size * (num\_key\_value\_heads * head\_dim) + 4 * GBS  * num\_hidden\_layers * num\_attention\_heads * SEQ\_LEN * head\_dim$
</p>

### mlp
<p>
$gate\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$up\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$down\_proj: 2 * GBS * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>

<p>
$mlp\_total: 6 * GBS * num\_hidden\_layers * hidden\_size * intermediate\_size$
</p>


# References

[modeling_qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py)

[gpu-utilization-misleading](https://www.trainy.ai/blog/gpu-utilization-misleading)

[large-scale-training-hugging-face](https://pytorch.org/blog/large-scale-training-hugging-face/)

[PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/pdf/2204.02311)