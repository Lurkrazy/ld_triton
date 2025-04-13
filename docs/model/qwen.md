
# A Simple Model of Memory

## All data initially in slow memory
$m =$ number of memory elements (words) moved between fast and slow memory

$t_m =$ time per slow memory operation (inverse bandwidth in best case)

$f =$ number of arithmetic operations

$t_f =$ time per arithmetic operation << tm

$CI = \frac{f}{m}$  average number of flops per slow memory access

## Actual time

$f * t_f + m * t_m = f * t_f * (1 + \frac{t_m}{t_f}  * \frac{1}{CI})$

## matmul CI

$A \in R^{MK}$

$B \in R^{KN}$

$C \in R^{MN}$

$f = 2MNK$

$m_{C} = 2MN$    读一次，写一次

$m_{A} = MK* \frac{N}{BLOCK\_N}$

$m_{B} = KN* \frac{M}{BLOCK\_M}$

$m = 2MN + \frac{MNK}{BLOCK\_N} + \frac{MNK}{BLOCK\_M}$

$CI = \frac{f}{m}$

$= \frac{2MNK}{2MN + \frac{MNK}{BLOCK\_N} + \frac{MNK}{BLOCK\_M}}$

$= \frac{2}{\frac{2}{K}+ \frac{1}{BLOCK\_N} + \frac{1}{BLOCK\_M}}$

# Model FLOPs Utilization (MFU)
<p>
$batch\_size, seqlen\_q, seqlen\_kv$
</p>

## embed_tokens(Embedding)

<p>
$weight\_shape = (vocab\_size, hidden\_size)$
</p>

### Tensor Core
#### forward

<p>
$FLOPs = 0$
</p>

#### backword

<p>
$FLOPs = 0$
</p>

### Cuda Core

#### forward

<p>
$FLOPs = 0$
</p>

#### backword

<p>
$FLOPs = o(0)$
</p>

### SFU

#### forward

<p>
$FLOPs = 0$
</p>

#### backword

<p>
$FLOPs = 0$
</p>


## rotary_emb(Qwen2RotaryEmbedding)
### Tensor Core
#### forward

<p>
$FLOPs = 0$
</p>

#### backword

<p>
$FLOPs = 0$
</p>

### Cuda Core

#### forward

<p>
$FLOPs = hidden\_size * seqlen$
</p>

#### backword

<p>
$FLOPs = 0$
</p>

### SFU

#### forward

<p>
$FLOPs = hidden\_size * seqlen$
</p>

#### backword

<p>
$FLOPs = 0$
</p>

## layers(Qwen2DecoderLayer * num_hidden_layers )
## Qwen2DecoderLayer
### self_attn(Qwen2Attention)
#### q_proj(Linear)

<p>
$weight\_shape = (num\_attention\_heads * head\_dim, hidden\_size)$
</p>

<p>
$bias\_shape = (num\_attention\_heads * head\_dim)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, hidden\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_q, num\_attention\_heads * head\_dim)$
</p>

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

###### backword

##### Cuda Core

###### forward

<p>
$batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim)$
</p>

###### backword

##### SFU

###### forward

<p>
$FLOPs = 0$
</p>

###### backword

<p>
$FLOPs = 0$
</p>

#### k_proj(Linear)
<p>
$weight\_shape = ( num\_key\_value\_heads * head\_dim, hidden\_size)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_kv, hidden\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_kv, num\_key\_value\_heads * head\_dim)$
</p>

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

##### Cuda Core

###### forward

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

##### SFU

###### forward

<p>
$FLOPs = 0$
</p>

###### backword

<p>
$FLOPs = 0$
</p>


#### v_proj(Linear)
<p>
$weight\_shape = (num\_key\_value\_heads * head\_dim, hidden\_size)$
</p>

<p>
$bias\_shape = (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_kv , hidden\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_kv , num\_key\_value\_heads * head\_dim)$
</p>

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

##### Cuda Core

###### forward

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

##### SFU

###### forward

<p>
$FLOPs = 0$
</p>

###### backword

<p>
$FLOPs = 0$
</p>

#### o_proj(Linear)
<p>
$weight\_shape = (hidden\_size, num\_attention\_heads * head\_dim)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, num\_attention\_heads * head\_dim)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_q, hidden\_size)$
</p>

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q  * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

</p>

###### backword

##### Cuda Core

###### forward

<p>
$FLOPs = 0$
</p>

###### backword

##### SFU

###### forward

<p>
$FLOPs = 0$
</p>

###### backword

<p>
$FLOPs = 0$
</p>


#### attention_interface
<p>
$q\_shape = (batch\_size, num\_attention\_heads, seqlen\_q, head\_dim)$
</p>

<p>
$k\_shape = (batch\_size, num\_key\_value\_heads, seqlen\_kv, head\_dim) -> (batch\_size, num\_attention\_heads, seqlen\_kv, head\_dim)$
</p>

<p>
$v\_shape = (batch\_size, num\_key\_value\_heads, seqlen\_kv, head\_dim) -> (GBS, num\_attention\_heads, seqlen\_kv, head\_dim)$
</p>

$softmax(x_{ij}) = \frac{exp^{x_{ij}-M}}{\sum_{k=0}^{seqlen\_kv-1}exp^{x_{ik}-M}}$

##### Tensor Core
###### forward
<p>
$qk: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$pv: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$total: 4 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

###### backward

##### Cuda Core
###### forward

<p>
$scaling: batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

<p>
$softmax: 3 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

<p>
$total: 4 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

###### backward

##### SFU
<p>
$softmax: batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

###### backward


#### apply_rotary_pos_emb
##### Tensor Core
###### forward

$FLOPs=0$

###### backward

$FLOPs=0$


##### Cuda Core
###### forward

<p>
$q_embed: 3 * batch\_size * seqlen\_q * (num\_attention\_heads * head\_dim)$
</p>

<p>
$k_embed: 3 * batch\_size * seqlen\_kv * (num\_key\_value\_heads * head\_dim)$
</p>

###### backward

##### SFU

###### forward

$FLOPs=0$

###### backward

$FLOPs=0$


#### self_attn Total
##### Tensor Core
###### forward

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim) $
</p>

<p>
$+ 4 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim) $
</p>

<p>
$+ 4 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

###### backward

##### Cuda Core
###### forward
<p>
$FLOPs = batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim) $
</p>

<p>
$+ 2 * batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$+ 4 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv $
</p>

<p>
$+ 3 * batch\_size * seqlen\_q * (num\_attention\_heads * head\_dim) $
</p>

<p>
$+ 3 * batch\_size * seqlen\_kv * (num\_key\_value\_heads * head\_dim) $
</p>

###### backward

##### SFU
###### forward
<p>
$FLOPs = batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

###### backward

### mlp(Qwen2MLP)
#### gate_proj(Linear)
<p>
$weight\_shape = (intermediate\_size, hidden\_size)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, hidden\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_q, intermediate\_size)$
</p>


##### Tensor Core
###### forward
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

###### backward

<p>
$dweight: 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$dinput:  2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

#### up_proj(Linear)

<p>
$weight\_shape = (intermediate\_size, hidden\_size)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, hidden\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_q, intermediate\_size)$
</p>


##### Tensor Core
###### forward
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

###### backward
<p>
$dweight: 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$dinput:  2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>


#### act_fn(silu)

<p>
$silu(x) = \frac{x}{1 + e^{-x}}$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, intermediate\_size)$
</p>


##### Cuda Core
###### forward
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * intermediate\_size$
</p>

###### backward

<p>
$FLOPs = 6 * batch\_size * seqlen\_q * intermediate\_size$
</p>

##### SFU
###### forward
<p>
$FLOPs = batch\_size * seqlen\_q * intermediate\_size$
</p>

###### backward

<p>
$FLOPs = 0$
</p>

or

<p>
$FLOPs = batch\_size * seqlen\_q * intermediate\_size$
</p>

#### down_proj(Linear)
<p>
$weight\_shape = (intermediate\_size, hidden\_size)$
</p>

<p>
$input\_shape = (batch\_size, seqlen\_q, intermediate\_size)$
</p>

<p>
$output\_shape = (batch\_size, seqlen\_q, hidden\_size)$
</p>

##### Tensor Core
###### forward
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

###### backward
<p>
$dweight: 2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$dinput:  2 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

#### mlp Total
##### Tensor Core
###### forward
<p>
$FLOPs = 6 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

###### backward
<p>
$FLOPs = 12 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

##### Cuda Core
###### forward
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * intermediate\_size$
</p>

###### backward
<p>
$FLOPs = 6 * batch\_size * seqlen\_q * intermediate\_size$
</p>

##### SFU
###### forward
<p>
$FLOPs = batch\_size * seqlen\_q * intermediate\_size$
</p>

###### backward

<p>
$FLOPs = 0$
</p>

or

<p>
$FLOPs = batch\_size * seqlen\_q * intermediate\_size$
</p>

### input_layernorm(Qwen2RMSNorm)
#### forward
<p>
$y_{ij} = x_{ij} * rrms(x)_{i} * \gamma_{j}$
</p>

<p>
$rrms(x)_{i} = \frac{1}{\sqrt {\epsilon + \frac{1}{hidden\_size}\sum_{j=0}^{hidden\_size-1}x_{ij}^2}}$
</p>

save $rrms$ for backward

#### backward

<p>
$dinput_{ij} = doutput_{ij} * rrms(x)_{i} * \gamma_{j} -\frac{x_{ij}}{hidden\_size}*\sum_{j=0}^{hidden\_size-1}(doutput_{ij} * x_{ij} * rrms(x)_{i} ^ {3} * \gamma_{j})$
</p>

<p>
$d\gamma_{j}=\sum_{k=0}^{batch\_size * seqlen\_q} df_{kj} . rrms(x)_{k} .x_{kj}$
</p>

#### Cuda Core
##### forward

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

##### backward
<p>
$dinput = 8 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

<p>
$d\gamma = 3 * batch\_size * seqlen\_q * hidden\_size$
</p>

<p>
$FLOPs = 11 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

#### SFU
##### forward
<p>
$FLOPs = batch\_size * seqlen\_q$
</p>

##### backward

<p>
$FLOPs = 0$
</p>

### post_attention_layernorm(Qwen2RMSNorm)
#### Cuda Core
##### forward

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

##### backward
<p>
$dinput = 8 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

<p>
$d\gamma = 3 * batch\_size * seqlen\_q * hidden\_size$
</p>

<p>
$FLOPs = 11 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q$
</p>

#### SFU
##### forward
<p>
$FLOPs = batch\_size * seqlen\_q$
</p>

##### backward

<p>
$FLOPs = 0$
</p>

### residual
#### Cuda Core
##### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size$
</p>

##### backward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size$
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

[WASP: Exploiting GPU Pipeline Parallelism with Hardware-Accelerated Automatic Warp Specialization](https://www.nealcrago.com/wp-content/uploads/WASP_HPCA2024_preprint.pdf)

[ISPA: Exploiting Intra-SM Parallelism in GPUs via Fine-grained Resource Management](https://mivenhan.github.io/publication/2022ispa/2022ispa.pdf)

[Improving GPU Throughput through Parallel Execution Using Tensor Cores and CUDA Cores](https://par.nsf.gov/servlets/purl/10415343)

[Warp Scheduling and Divergenc](https://cse.iitkgp.ac.in/~soumya/hp3/slides/warp-divr.pdf)

[Dissecting the CUDA scheduling hierarchy: a Performance and Predictability Perspective](https://conferences.computer.org/cpsiot/pdfs/RTAS2020-4uXAu5nqG7QNiz5wFYyfj6/549900a210/549900a210.pdf)

[Embedding计算在GPU上的性能优化](https://yywangcs.notion.site/Embedding-GPU-1-GPU-Occupancy-178fc9f5d80580d4affddeb4c40c64e0)