
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

### Cuda Core

#### forward

<p>
$FLOPs = 0$
</p>

#### backword

<p>
$FLOPs = o(0)$
</p>

## rotary_emb(Qwen2RotaryEmbedding)
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

## layers(Qwen2DecoderLayer * num_hidden_layers)
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

##### forward

$output = input @ weight^{T} + bias$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

$dbias_{j} = \sum_{i=0}{batch\_size * seqlen\_q - 1} doutput_{ij}$

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

###### backword

<p>
$dinput:* batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$dweight = 2 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

##### Cuda Core

###### forward

<p>
$FLOPs = batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim)$
</p>

###### backword
<p>
$FLOPs = batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim)$
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

##### forward

$output = input @ weight^{T} + bias$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

$dbias_{j} = \sum_{i=0}{batch\_size * seqlen\_q - 1} doutput_{ij}$

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

<p>
$dinput = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$dweight = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

##### Cuda Core

###### forward

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
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

##### forward

$output = input @ weight^{T} + bias$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

$dbias_{j} = \sum_{i=0}{batch\_size * seqlen\_q - 1} doutput_{ij}$

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

<p>
$dinput: 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$dweight: 2 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)$
</p>

##### Cuda Core

###### forward

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

###### backword

<p>
$FLOPs = batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
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

##### forward

$output = input @ weight^{T}$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

##### Tensor Core
###### forward

<p>
$FLOPs = 2 * batch\_size * seqlen\_q  * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

</p>

###### backword

<p>
$dinput: 2 * batch\_size * seqlen\_q  * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$dweight: 2 * batch\_size * seqlen\_q  * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

<p>
$FLOPs = 4 * batch\_size * seqlen\_q  * hidden\_size * (num\_attention\_heads * head\_dim)$
</p>

#### attention_interface
<p>
$q\_shape = (batch\_size, num\_attention\_heads, seqlen\_q, head\_dim)$
</p>

<p>
$k\_shape = (batch\_size, num\_key\_value\_heads, seqlen\_kv, head\_dim) -> (batch\_size, num\_attention\_heads, seqlen\_kv, head\_dim)$
</p>

<p>
$v\_shape = (batch\_size, num\_key\_value\_heads, seqlen\_kv, head\_dim) -> (batch\_size, num\_attention\_heads, seqlen\_kv, head\_dim)$
</p>

<p>
$output\_shape =(batch\_size, num\_attention\_heads, seqlen\_q, head\_dim)$
</p>

$softmax(x_{ij}) = \frac{exp^{x_{ij}-M}}{\sum_{k=0}^{seqlen\_kv-1}exp^{x_{ik}-M}}$
##### forward

$qk = q@k^{T} * scaling$

$p = softmax(qk)$

$pv = p@v$

$attention(q, k, v, scaling) = softmax(\frac{q@k^{T}} * scaling) @ v$

##### backward

$qk = q@k^{T} * scaling$

$p = softmax(qk)$

$dp = doutput@v^{T}$

$d = p*(dp - sum(doutput*output, dim=-1, keepdim=True)*scaling)$

$dq = d@k$

$dk = d^{T}@q$

$dv = p^{T}@doutput$

##### Tensor Core
###### forward
<p>
$qk: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$pv: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$FLOPs = 4 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

###### backward
<p>
$qk: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$dp: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$dq: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$dk: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$dv: 2 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

<p>
$FLOPs = 10 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

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

<p>
$scaling: batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

<p>
$softmax: 3 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

<p>
$d: 5 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

<p>
$d: 9 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

##### SFU
<p>
$softmax: batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

###### backward

<p>
$softmax: batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

#### apply_rotary_pos_emb
<p>
$q\_embed = (q * cos) + (rotate\_half(q) * sin)$
</p>

<p>
$k\_embed = (k * cos) + (rotate\_half(k) * sin)$
</p>

##### Cuda Core
###### forward

<p>
$q\_embed: 3 * batch\_size * seqlen\_q * (num\_attention\_heads * head\_dim)$
</p>

<p>
$k\_embed: 3 * batch\_size * seqlen\_kv * (num\_key\_value\_heads * head\_dim)$
</p>

#### self_attn Total
##### Tensor Core
###### forward
<p>
$FLOPs = $
</p>

q_proj + o_proj
<p>
$4 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim) $
</p>

k_proj + v_proj
<p>
$+ 4 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim) $
</p> 

attention_interface
<p>
$+ 4 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

###### backward

<p>
$FLOPs =$
</p>

q_proj + o_proj
<p>
$8 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim) $
</p>

k_proj + v_proj
<p>
$+ 8 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim) $
</p>

attention_interface
<p>
$+ 10 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv$
</p>

##### Cuda Core
###### forward
<p>
$FLOPs = $
</p>

q_proj
<p>
$batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim) $
</p> 

k_proj + v_proj
<p>
$+ 2 * batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

attention_interface
<p>
$+ 4 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv $
</p>

q_embed
<p>
$+ 3 * batch\_size * seqlen\_q * (num\_attention\_heads * head\_dim) $
</p>

k_embed
<p>
$+ 3 * batch\_size * seqlen\_kv * (num\_key\_value\_heads * head\_dim) $
</p>

###### backward

<p>
$FLOPs = $
</p>

q_proj
<p>
$batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim) $
</p>

k_proj + v_proj
<p>
$+ 2 * batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim)$
</p>

attention_interface
<p>
$+ 9 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv $
</p>

##### SFU
###### forward

softmax
<p>
$FLOPs = batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

###### backward

softmax
<p>
$FLOPs = batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv$
</p>

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

##### forward

$output = input @ weight^{T}$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$


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

##### forward

$output = input @ weight^{T}$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

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
##### forward
<p>
$silu(x) = \frac{x}{1 + e^{-x}}$
</p>

##### backward
<p>
$silu(x) = \frac{x}{1 + e^{-x}}$
</p>

<p>
$dinput_{ij} = doutput_{ij} * (\frac{1 + x_{ij}}{1 + e^{-x}} - \frac{x_{ij}}{(1 + e^{-x})^{2}})$
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

##### forward

$output = input @ weight^{T}$

##### backward

$dinput = doutput @ weight$

$dweight = doutput^{T} @ input$

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
gate_proj + up_proj + down_proj
<p>
$FLOPs = 6 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

###### backward
gate_proj + up_proj + down_proj
<p>
$FLOPs = 12 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size$
</p>

##### Cuda Core
###### forward
act_fn
<p>
$FLOPs = 2 * batch\_size * seqlen\_q * intermediate\_size$
</p>

###### backward
act_fn
<p>
$FLOPs = 6 * batch\_size * seqlen\_q * intermediate\_size$
</p>

##### SFU
###### forward
act_fn
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
$FLOPs = 0$
</p>

## Total
### Tensor Core
#### forward
self_attn Total
<p>
$FLOPs = $
</p>

q_proj + o_proj
<p>
$+ num\_hidden\_layers * (4 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)) $
</p>

k_proj + v_proj
<p>
$+ num\_hidden\_layers * (4 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)) $
</p> 

attention_interface
<p>
$+ num\_hidden\_layers * (4 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv)$
</p>

mlp Total
gate_proj + up_proj + down_proj
<p>
$+ num\_hidden\_layers * (6 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size)$
</p>

#### backward
self_attn Total
<p>
$FLOPs = $
</p>
q_proj + o_proj
<p>
$num\_hidden\_layers * (8 * batch\_size * seqlen\_q * hidden\_size * (num\_attention\_heads * head\_dim)) $
</p>

k_proj + v_proj
<p>
$+ num\_hidden\_layers * (8 * batch\_size * seqlen\_kv *  hidden\_size * (num\_key\_value\_heads * head\_dim)) $
</p>

attention_interface
<p>
$+ num\_hidden\_layers * (10 * batch\_size * (num\_attention\_heads * head\_dim) * seqlen\_q * seqlen\_kv)$
</p>

mlp Total
gate_proj + up_proj + down_proj 
<p>
$+ num\_hidden\_layers * (12 * batch\_size * seqlen\_q * hidden\_size * intermediate\_size)$
</p>

### Cuda Core
#### forward

$FLOPs = $

embed_tokens

$0$

rotary_emb
$+ hidden\_size * seqlen$

layers
self_attn Total
q_proj
<p>
$+ num\_hidden\_layers * (batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim)) $
</p> 

k_proj + v_proj
<p>
$+ num\_hidden\_layers * (2 * batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim))$
</p>

attention_interface
<p>
$+ num\_hidden\_layers * (4 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv) $
</p>

q_embed
<p>
$+ num\_hidden\_layers * (3 * batch\_size * seqlen\_q * (num\_attention\_heads * head\_dim)) $
</p>

k_embed
<p>
$+ num\_hidden\_layers * (3 * batch\_size * seqlen\_kv * (num\_key\_value\_heads * head\_dim)) $
</p>

mlp Total
act_fn
<p>
$+ num\_hidden\_layers * (2 * batch\_size * seqlen\_q * intermediate\_size)$
</p>

input_layernorm Total
<p>
$+ num\_hidden\_layers * (4 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q)$
</p>

post_attention_layernorm Total
<p>
$+ num\_hidden\_layers * (4 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q)$
</p>

residual Total
<p>
$+ num\_hidden\_layers * (2 * batch\_size * seqlen\_q * hidden\_size)$
</p>

#### backward
$FLOPs = $

embed_tokens
$o(0)$

rotary_emb
$+ 0$

self_attn Total
q_proj
<p>
$+ num\_hidden\_layers * (batch\_size * seqlen\_q  * (num\_attention\_heads * head\_dim))$
</p>

k_proj + v_proj
<p>
$+ num\_hidden\_layers * (2 * batch\_size * seqlen\_kv  * (num\_key\_value\_heads * head\_dim))$
</p>

attention_interface
<p>
$+ num\_hidden\_layers * (9 * batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv)$
</p>

mlp Total
act_fn
<p>
$+ num\_hidden\_layers * (6 * batch\_size * seqlen\_q * intermediate\_size)$
</p>

input_layernorm Total
<p>
$+ num\_hidden\_layers * (11 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q)$
</p>

post_attention_layernorm Total
<p>
$+ num\_hidden\_layers * (11 * batch\_size * seqlen\_q * hidden\_size + 2 * batch\_size * seqlen\_q)$
</p>

residual Total
<p>
$+ 0$
</p>

### SFU
#### forward

<p>
$FLOPs = $
</p>

rotary_emb
<p>
$+ hidden\_size * seqlen$
</p>

self_attn Total

softmax
<p>
$num\_hidden\_layers * (batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv)$
</p>

mlp Total

act_fn
<p>
$+ num\_hidden\_layers * (batch\_size * seqlen\_q * intermediate\_size)$
</p>

input_layernorm Total
<p>
$+ num\_hidden\_layers * (batch\_size * seqlen\_q)$
</p>

post_attention_layernorm Total
<p>
$+ num\_hidden\_layers * (batch\_size * seqlen\_q)$
</p>

#### backward
<p>
$FLOPs = $
</p>

self_attn Total

softmax
<p>
$num\_hidden\_layers * (batch\_size * num\_attention\_heads * seqlen\_q * seqlen\_kv)$
</p>

mlp Total

act_fn
<p>
$+ 0$
</p>

input_layernorm Total
<p>
$+ 0$
</p>

post_attention_layernorm Total

$+ 0$

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