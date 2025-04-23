
# forward

$input \in {R}^{M \times infs}$

$weight \in {R}^{outfs \times infs}$

$bias \in {R}^{outfs}$

$linear \in {R}^{M \times outfs}$

$linear = input@weight^T + bias$

$linear_{ij} = \sum_{k=0}^{infs-1}input_{ik}.weight_{jk} + bias_{j}$

# 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

## input求导

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{k}}{\partial input_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial input_{ij}}$

### $p \neq i$

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}} = 0$
</p>

### $p = i$

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{ik}weight_{qk}}{\partial input_{ij}}$

$=\frac{\partial input_{ij}weight_{qj}}{\partial input_{ij}}$

$=weight_{qj}$

## weight求导

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{k}}{\partial weight_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial weight_{ij}}$

### $q \neq i$

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}} = 0$
</p>

### $q = i$

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{ik}}{\partial weight_{ij}}$

$=\frac{\partial input_{pj}weight_{ij}}{\partial weight_{ij}}$

$=input_{pj}$

## bias求导

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

$=\frac{\partial bias_{q}}{\partial bias_{j}}$

### $q \neq j$

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}=\frac{\partial bias_{q}}{\partial bias_{j}}=0$
</p>

### $q = j$

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}=\frac{\partial bias_{j}}{\partial bias_{i}}=1$
</p>

# 链式法则

## input链式法则

### 元素形式

<p>
$\frac{\partial f(linear(input)_{kl})}{\partial input_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(input)_{kl})}{\partial linear(input)_{pq}}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\sum_{q=0}^{outfs-1}df_{iq}.\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$
</p>

$=\sum_{q=0}^{outfs-1}df_{iq}.weight_{qj}$

### 矩阵形式

$\frac{\partial f(input)}{\partial input}$

$=df@weight$

## weight链式法则

### 元素形式

<p>
$\frac{\partial f(linear(weight)_{kl})}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(weight)_{kl})}{\partial linear(weight)_{pq}}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}df_{pi}.\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$
</p>

$=\sum_{p=0}^{M-1}df_{pi}.input_{pj}$

### 矩阵形式

$\frac{\partial f(weight)}{\partial weight}$

$=df^{T}@intput$

## bias链式法则

### 元素形式

<p>
$\frac{\partial f(linear(bias)_{kl})}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(bias)_{kl})}{\partial linear(bias)_{pq}}.\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}df_{pj}.\frac{\partial linear(bias)_{j}}{\partial bias_{j}}$
</p>

$=\sum_{p=0}^{M-1}df_{pj}$

### 矩阵形式

$\frac{\partial f(bias)}{\partial bias} = \sum_{p=0}^{M-1} df_{p,j \in (outfs-1)}$

# 链式法则二

$A \in {R}^{M \times K}$

$B \in {R}^{K \times N}$

$C = AB$

矩阵乘法链式法则(matmul.md)

$\frac{\partial f(A)}{\partial A} = \frac{\partial f(C)}{\partial C}@B^T$

$\frac{\partial f(B)}{\partial B} = A^T @ \frac{\partial f(C)}{\partial C}$ 

$A = input$

$B = weight^T$

$\frac{\partial f(input)}{\partial input} = \frac{\partial f(linear)}{\partial linear}@(weight^T)^T = \frac{\partial f(linear)}{\partial linear}@weight$

$\frac{\partial f(weight^T)}{\partial weight^T} = A^T @ \frac{\partial f(linear)}{\partial C}$ 

$\frac{\partial f(weight)}{\partial weight} = (\frac{\partial f(linear)}{\partial C}) ^ T @ A$ 

# Tensor Parallel
## Column Parallel
### forward

$input \in {R}^{M \times infs}$

$weight \in {R}^{outfs \times infs}$

$tp$ Tensor Parallel Size

$$
weight =  \begin{bmatrix}
weight_{0} \\
weight_{1} \\
\vdots \\
weight_{tp-1}
\end{bmatrix}, weight_{i} \in {R}^{\frac{outfs}{tp} \times infs}
$$

$$
weight^{T} =  \begin{bmatrix}
weight_{0}^{T} ,
weight_{1}^{T} ,
\cdots ,
weight_{tp-1}^{T}
\end{bmatrix}, weight_{i}^{T} \in {R}^{infs \times \frac{outfs}{tp}}
$$

$bias \in {R}^{outfs}$

$$
bias =  \begin{bmatrix}
bias_{0},
bias_{1},
\cdots,
bias_{tp-1}
\end{bmatrix}, bias_{i} \in {R}^{\frac{outfs}{tp}}
$$

$output \in {R}^{M \times outfs}$

$$
output =  \begin{bmatrix}
output_{0}, 
output_{1}, 
\cdots,
output_{tp-1}
\end{bmatrix}, output_{i} \in {R}^{M \times \frac{outfs}{tp}}
$$

$$
output = input@weight^T + bias =  input@\begin{bmatrix}
weight_{0}^{T} ,
weight_{1}^{T} ,
\cdots ,
weight_{tp-1}^{T}
\end{bmatrix} + \begin{bmatrix}
bias_{0},
bias_{1},
\cdots,
bias_{tp-1}
\end{bmatrix}
$$

### backward
#### input链式法则
$\frac{\partial f(linear(input))}{\partial input}$

$=doutput@weight$

$$
doutput =  \begin{bmatrix}
doutput_{0}, 
doutput_{1}, 
\cdots,
doutput_{tp-1}
\end{bmatrix}, output_{i} \in {R}^{M \times \frac{outfs}{tp}}
$$

$$
weight =  \begin{bmatrix}
weight_{0} \\
weight_{1} \\
\vdots \\
weight_{tp-1} \\
\end{bmatrix}, weight_{i} \in {R}^{\frac{outfs}{tp} \times infs}
$$

$$
dinput =  \begin{bmatrix}
doutput_{0}, 
doutput_{1}, 
\cdots,
doutput_{tp-1} \\
\end{bmatrix} @ \begin{bmatrix}
weight_{0} \\
weight_{1} \\
\vdots \\
weight_{tp-1} \\
\end{bmatrix} = \sum_{i=0}^{tp-1}doutput_{i}@weight_{i}
$$

#### weight链式法则

$\frac{\partial f(linear(weight))}{\partial weight}$

$=doutput^{T}@intput$

$$
doutput^{T} =  \begin{bmatrix}
doutput_{0}^{T} \\
doutput_{1}^{T} \\
\vdots \\
doutput_{tp-1}^{T}
\end{bmatrix}, doutput_{i}^{T} \in {R}^{\frac{outfs}{tp}}  \times M
$$

$input \in {R}^{M \times infs}$

$$
dweight=doutput^{T}@intput =  \begin{bmatrix}
doutput_{0}^{T} \\
doutput_{1}^{T} \\
\vdots \\
doutput_{tp-1}^{T} \\
\end{bmatrix} @ input
$$

#### bias链式法则

$\frac{\partial f(linear(bias))}{\partial bias} = \sum_{p=0}^{M-1} doutput_{p,:}$

$$
dbias =  \begin{bmatrix}
dbias_{0},
dbias_{1},
\cdots,
dbias_{tp-1}
\end{bmatrix}, dbias_{j} = \sum_{p=0}^{M-1} doutput_{p,j*\frac{outfs}{tp}:(j+1)*\frac{outfs}{tp}}, dbias_{i} \in {R}^{\frac{outfs}{tp}}
$$

$$
dbias = \sum_{p=0}^{M-1} doutput_{p,:} =  \begin{bmatrix}
\sum_{p=0}^{M-1} doutput_{p,0:\frac{outfs}{tp}},
\sum_{p=0}^{M-1} doutput_{p,\frac{outfs}{tp}:2*\frac{outfs}{tp}},
\cdots,
\sum_{p=0}^{M-1} doutput_{p,(tp-1)*\frac{outfs}{tp}:tp} \\
\end{bmatrix}
$$

# Arithmetic intensity
## Hardware
RTX 3090 

Memory Bandwidth (GB/sec): 936 GB/sec

Peak FP16 Tensor TFLOPS with FP32 Accumulate: 71 TFLOPS

Peak FP32 TFLOPS (non-Tensor): 35.6 TFLOPS
SMs: 82

CUDA Cores / SM: 128

CUDA Cores / GPU: 10496

Tensor Cores / SM: 4 (3rd Gen)

Tensor Cores / GPU: 328 (3rd Gen)

L1 Data Cache/Shared Memory: 10752 KB

L2 Cache Size: 6144 KB

Register File Size: 20992 KB

GPU Boost Clock (MHz): 1695

Peak FP32 TFLOPS (non-Tensor) = 2 * (GPU Boost Clock (MHz)) * (CUDA Cores / GPU) = 2 * 1695 * 1e6 * 10496 = 35.581440e12

Peak FP16 Tensor TFLOPS with FP32 Accumulate = 2 * Peak FP32 TFLOPS (non-Tensor)  = 71.16288e12

Peak FP32 TFLOPS / SM / clock = 2 * (CUDA Cores / SM) = 2 * 128 FLOPS = 256 FLOPS

Peak FP16 Tensor TFLOPS with FP32 Accumulate / SM  / clock = 2 * 128 * 2  = 512 FLOPS

Peak FP16 Tensor TFLOPS with FP32 Accumulate / Tensor Cores / clock = 128 FLOPS

Shared Memory / SM: Shared Memory / SMs = 10752 KB / 82 = 131.122 KB

Shared Memory Bandwidth: 128 bytes/clock per SM

Shared Memory Bandwidth: (128 bytes/clock per SM) * (GPU Boost Clock (MHz)) * SMs = 128 * 1695 * 1e6 * 82 = 17.80e12

Memory Interface: 348 bit

Memory Clock (Data Rate): 19.5 Gbps

Memory Bandwidth (GB/sec): 936 GB/sec

## basic triple loop nest
```
for (int cta_n = 0; cta_n < GemmN; cta_n += CtaTileN) {                     // for each threadblock_y           } threadblock-level concurrency
  for (int cta_m = 0; cta_m < GemmM; cta_m += CtaTileM) {                   //    for each threadblock_x        }

    for (int cta_k = 0; cta_k < GemmK; cta_k += CtaTileK) {                 //       "GEMM mainloop" - no unrolling
                                                                            //                       - one iteration of this loop is one "stage"
                                                                            //
      for (int warp_n = 0; warp_n < CtaTileN; warp_n += WarpTileN) {        // for each warp_y                  } warp-level parallelism
        for (int warp_m = 0; warp_m < CtaTileM; warp_m += WarpTileM) {      //    for each warp_x               }
                                                                            //
          for (int warp_k = 0; warp_k < CtaTileK; warp_k += WarpTileK) {         //       fully unroll across CtaTileK
                                                                            //         - one iteration of this loop is one "k Group"
                                                                            //
            for (int mma_k = 0; mma_k < WarpTileK; mma_k += MmaK) {         // for each mma instruction         } instruction-level parallelism
              for (int mma_n = 0; mma_n < WarpTileN; mma_n += MmaN) {       //    for each mma instruction      }
                for (int mma_m = 0; mma_m < WarpTileM; mma_m += MmaM) {     //        for each mma instruction  }
                                                                            //
                  mma_instruction(d, a, b, c);                              //            TensorCore matrix computation

                }   // for mma_m
              }   // for mma_n
            }   // for mma_k

          }   // for warp_k
        }   // for warp_m
      }   // for warp_n

    }   // for cta_k
  }   // for cta_m
}   // for cta_n

```

### Thread-level GEMM (instruction-level parallelism)
```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3};

mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3};
```

数据在Register File中， 同一个WarpBlock在同一个SM中计算

Shared Memory Bandwidth 是 128 bytes/clock per SM, 没有找到Register File Bandwidth. 一般情况下Register File Bandwidth 比 Shared Memory Bandwidth 高一个数量级

$InstructionShape=(MmaM， MmaN, MmaK)=(16, 8, 16)$

$WarpShape=(WarpTileM, WarpTileN, WarpTileK)=(num\_MmaM * MmaM, num\_MmaN, num\_MmaK)$

<p>
$FLOPs\_per\_SM = 2 * WarpTileM * WarpTileN * WarpTileK$
</p>

<p>
$clks = \frac{FLOPs\_per\_SM}{512}$
</p>

<p>
$Bytes\_per\_SM = WarpTileM * WarpTileK + WarpTileK * WarpTileN + 2 * num\_MmaK * WarpTileM * MmaN$
</p>

<p>
$= WarpTileM * WarpTileK + WarpTileK * WarpTileN + 2 * (WarpTileK / MmaK)  * WarpTileM * WarpTileN$
</p>

<p>
$Bytes\_per\_SM\_per\_clc = \frac{Bytes\_per\_SM}{clks}$
</p>

<p>
$= \frac{Bytes\_per\_SM}{FLOPs\_per\_SM / 512}$
</p>

<p>
$= \frac{512}{Arithmetic\_intensity}$
</p>

<p>
$Arithmetic\_intensity$ 
</p>

<p>
$=\frac{FLOPs\_per\_SM} {Bytes\_per\_SM}$
</p>

<p>
$\frac{2 * WarpTileM * WarpTileN * WarpTileK}{WarpTileM * WarpTileK + WarpTileK * WarpTileN + 2 * num\_MmaK * WarpTileM * WarpTileN}$
</p>

<p>
$=\frac{1}{1 / 2WarpTileM + 1 / 2WarpTileN + num\_MmaK / WarpTileK}$
</p>

<p>
$=\frac{1}{1 / 2WarpTileM + 1 / 2WarpTileN + 1 / MmaK}$
</p>



```
InstructionShape = [
    (16, 8, 16),
]

WarpShape = [
    (16, 8, 16),
    (64, 64, 32),
    (32, 128, 32),
    (16, 64, 32),
    (16, 128, 32),
    (32, 32, 32),
    (64, 64, 64),
    (32, 32, 64),
    (64, 32, 32),

]

for (MmaM, MmaN, MmaK) in InstructionShape:
    for (WarpTileM, WarpTileN, WarpTileK) in WarpShape:
        FLOPs_per_SM = 2 * WarpTileM * WarpTileN * WarpTileK
        clks = FLOPs_per_SM / 512
        Bytes_per_SM = WarpTileM * WarpTileK + WarpTileK * WarpTileN + 2 * (WarpTileK / MmaK)  * WarpTileM * WarpTileN
        Bytes_per_clc = Bytes_per_SM / clks
        Arithmetic_intensity = 1 / (1 / (2 * WarpTileM) + 1 / (2 * WarpTileN) + 1 / MmaK)
        print(f'{(WarpTileM, WarpTileN, WarpTileK)}, clks: {clks:.3f}, Arithmetic_intensity: {Arithmetic_intensity:.3f}, Bytes_per_clc: {Bytes_per_clc:.3f}')

```

| WarpShape     | InstructionShape | Arithmetic intensity | Bytes_per_clc | clks   |
|---------------|------------------|----------------------|---------------|--------|
| (16, 8, 16)   | (16, 8, 16)      | 6.4                  | 80            | 8      |
| (64, 64, 32)  | (16, 8, 16)      | 12.800               | 40            | 512    |
| (32, 128, 32) | (16, 8, 16)      | 12.190               | 42            | 512    |
| (16, 64, 32)  | (16, 8, 16)      | 9.846                | 52            | 128    |
| (16, 128, 32) | (16, 8, 16)      | 10.240               | 50            | 256    |
| (32, 32, 32)  | (16, 8, 16)      | 10.667               | 48            | 128    |
| (64, 64, 64)  | (16, 8, 16)      | 12.800               | 40            | 1024   |
| (32, 32, 64)  | (16, 8, 16)      | 10.667               | 48            | 256    |
| (64, 32, 32)  | (16, 8, 16)      | 11.636               | 44            | 256    |

## Warp-level GEMM (warp-level parallelism)
```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {d0, d1, d2, d3}, [addr];

ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {d0, d1}, [addr];
```
数据在 Shared Memory 中, Shared Memory Bandwidth 是 128 bytes/clock per SM

<p>
$WarpShape=(WarpTileM, WarpTileN, WarpTileK)=(num\_MmaM * MmaM, num\_MmaN, num\_MmaK)$
</p>

<p>
$ThreadblockShape=(CtaTileM, CtaTileN, CtaTileK)=(num\_WarpTileM * WarpTileM, num\_WarpTileN * WarpTileN, num\_WarpTileK * WarpTileK)$
</p>

<p>
$FLOPs\_per\_SM = 2 * CtaTileM * CtaTileN * CtaTileK$
</p>

<p>
$clks = \frac{FLOPs\_per\_SM}{512}$
</p>

<p>
$Bytes\_per\_SM = num\_WarpTileN * CtaTileM * CtaTileK + num\_WarpTileM * CtaTileK * CtaTileN + 2 * CtaTileM * CtaTileN$
</p>

<p>
$= (CtaTileN / WarpTileN) * CtaTileM * CtaTileK + (CtaTileM / WarpTileM) * CtaTileK * CtaTileN + 2 * CtaTileM * CtaTileN$
</p>

<p>
$Bytes\_per\_SM\_per\_clc = \frac{Bytes\_per\_SM}{clks}$
</p>

<p>
$= \frac{Bytes\_per\_SM}{FLOPs\_per\_SM / 512}$
</p>

<p>
$= \frac{512}{Arithmetic\_intensity}$
</p>

<p>
$Arithmetic\_intensity=$ 
</p>

<p>
$\frac{FLOPs\_per\_SM} {Bytes\_per\_SM}$
</p>

<p>
$\frac{2 * CtaTileM * CtaTileN * CtaTileK}{(CtaTileN / WarpTileN) * CtaTileM * CtaTileK + (CtaTileM / WarpTileM) * CtaTileK * CtaTileN + 2 * CtaTileM * CtaTileN}$
</p>

<p>
$=\frac{1}{1 / 2WarpTileN + 1 / 2WarpTileM + 1 / CtaTileK}$
</p>

```
InstructionShape = [
    (16, 8, 16),
]

WarpShape_and_ThreadblockShape = [
    ((16, 8, 16), (16, 8, 16)),
    ((64, 64, 32), (64, 128, 32)),
    ((64, 64, 32), (64, 256, 32)),
    ((64, 64, 32), (128, 128, 32)),
    ((64, 64, 32), (128, 64, 32)),
    ((32, 128, 32), (64, 128, 32)),
    ((16, 64, 32), (64, 64, 32)),
    ((16, 128, 32), (64, 128, 32)),
    ((32, 32, 32), (64, 64, 32)),
    ((64, 64, 64), (128, 128, 64)),
    ((32, 32, 64), (64, 64, 64)),
    ((64, 32, 32), (128, 64, 32)),
    ((32, 64, 32), (64, 64, 32))
]

for ((WarpTileM, WarpTileN, WarpTileK), (CtaTileM, CtaTileN, CtaTileK)) in WarpShape_and_ThreadblockShape:
    FLOPs_per_SM = 2 * CtaTileM * CtaTileN * CtaTileK
    clks = FLOPs_per_SM / 512
    Bytes_per_SM = (CtaTileN / WarpTileN) * CtaTileM * WarpTileK + (CtaTileM / WarpTileM) * CtaTileK * CtaTileN + 2 * CtaTileM * CtaTileN
    Bytes_per_clc = Bytes_per_SM / clks
    Arithmetic_intensity = 1 / (1 / (2 * WarpTileM) + 1 / (2 * WarpTileN) + 1 / CtaTileK)
    print(f'{(WarpTileM, WarpTileN, WarpTileK)}, {(CtaTileM, CtaTileN, CtaTileK)}, clks: {clks:.3f}, Arithmetic_intensity: {Arithmetic_intensity:.3f}, Bytes_per_clc: {Bytes_per_clc:.3f}')
```

| WarpShape     | InstructionShape | ThreadblockShape | Arithmetic intensity | Bytes_per_clc | clks    |
|---------------|------------------|------------------|----------------------|---------------|---------|
| (16, 8, 16)   | (16, 8, 16)      | (16, 8, 16)      | 6.4                  | 80            | 8       |
| (64, 64, 32)  | (16, 8, 16)      | (64, 128, 32)    | 21.333               | 24            | 1024    |
| (64, 64, 32)  | (16, 8, 16)      | (64, 256, 32)    | 21.333               | 24            | 2048    |
| (64, 64, 32)  | (16, 8, 16)      | (128, 128, 32)   | 21.333               | 24            | 2048    |
| (64, 64, 32)  | (16, 8, 16)      | (128, 64, 32)    | 21.333               | 24            | 1024    |
| (32, 128, 32) | (16, 8, 16)      | (64, 128, 32)    | 19.692               | 26            | 1024    |
| (16, 64, 32)  | (16, 8, 16)      | (64, 64, 32)     | 14.222               | 36            | 512     |
| (16, 128, 32) | (16, 8, 16)      | (64, 128, 32)    | 15.059               | 34            | 1024    |
| (32, 32, 32)  | (16, 8, 16)      | (64, 64, 32)     | 16.000               | 32            | 512     |
| (64, 64, 64)  | (16, 8, 16)      | (128, 128, 64)   | 32.000               | 16            | 4096    |
| (32, 32, 64)  | (16, 8, 16)      | (64, 64, 64)     | 21.333               | 24            | 1024    |
| (64, 32, 32)  | (16, 8, 16)      | (128, 64, 32)    | 18.286               | 28            | 1024    |
| (32, 64, 32)  | (16, 8, 16)      | (64, 64, 32)     | 18.286               | 28            | 512     |

## Threadblock-level GEMM (threadblock-level)
```
cp.async.cg.shared.global [%1], [%2], %3

cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2
```
数据在 Global Memory 或者 L2 cache 中, Global Memory Bandwidth 是 936 GB/sec

RTX 3090 L2 cache Bandwidth 的查不到. 

[A100 L2 cache Bandwidth](https://forums.developer.nvidia.com/t/how-to-reach-peak-bandwidth-of-l2-cache-on-a100/198560)

A100 L2 cache Bandwidth: 5120 Bytes/clk = 5120 * 1414 * 1e6 = 7239.68 * 1e9 Bytes / second = 6742.477 GB / second

A100 L2 cache Bandwidth: 4830 GB/s

以下分析假设 RTX 3090 L2 cache Bandwidth 和 A100 L2 cache Bandwidth 一样

Global Memory Bandwidth = 936 GB/sec

ThreadblockShape=(CtaTileM, CtaTileN, CtaTileK)

qwen2.5 72B
ProblemShape=(2048, 8192, 29696)

<p>
$ThreadblockShape=(CtaTileM, CtaTileN, CtaTileK)$
</p>

<p>
$ProblemShape=(GemmM, GemmN, GemmK)=(num\_CtaTileM * CtaTileM, num\_CtaTileN * CtaTileN, num\_CtaTileK * CtaTileK)$
</p>

<p>
$FLOPs = (2 * GemmM * GemmN * GemmK)$
</p>

<p>
$seconds = \frac{FLOPs}{71.16288e12}$
</p>

<p>
$Bytes = (num\_CtaTileN * GemmM * GemmK + num\_CtaTileM * GemmK * GemmN + 2 * GemmM * GemmN)$
</p>

<p>
$= (GemmN / CtaTileN) * GemmM * GemmK + (GemmM / CtaTileM) * GemmK * GemmN + 2 * CtaTileM * CtaTileN$
</p>

<p>
$Bytes\_per\_second = \frac{Bytes}{seconds}$
</p>

<p>
$= \frac{Bytes\_second}{FLOPs\_per\_second / 71.16288e12}$
</p>

<p>
$= \frac{Bytes}{FLOPs / 71.16288e12}$
</p>

<p>
$= \frac{71.16288e12}{Arithmetic\_intensity}$
</p>

<p>
$Arithmetic\_intensity=$ 
</p>

<p>
$\frac{FLOPs\_per\_SM} {Bytes\_per\_SM}$
</p>

<p>
$\frac{FLOPs} {Bytes}$
</p>

<p>
$\frac{2 * GemmM * GemmN * GemmK}{(GemmN / CtaTileN) * GemmM * GemmK + (GemmM / CtaTileM) * GemmK * GemmN + 2 * CtaTileM * CtaTileN}$
</p>

<p>
$=\frac{1}{1 / 2CtaTileN + 1 / 2CtaTileM + 1 / GemmK}$
</p>

<p>
$Global\_Memory\_Bandwidth * (1 - (L2\_cache\_hit)) + (L2\_cache\_Bandwidth) * (L2\_cache\_hit) = GB\_per\_second$
</p>

<p>
$(L2\_cache\_Bandwidth - Global\_Memory\_Bandwidth) * L2\_cache\_hit = GB\_per\_second - Global\_Memory\_Bandwidth$
</p>

<p>
$L2\_cache\_hit = \frac{GB\_per\_second - Global\_Memory\_Bandwidth}{L2\_cache\_Bandwidth - Global\_Memory\_Bandwidth}$
</p>

```
ThreadblockShape_and_ProblemShape = [
    ((16, 8, 16), (1, 8192, 29696)),
    ((16, 64, 32), (1, 8192, 29696)),
    ((16, 64, 64), (1, 8192, 29696)),
    ((16, 128, 32), (1, 8192, 29696)),
    ((16, 128, 64), (1, 8192, 29696)),
    ((16, 256, 32), (1, 8192, 29696)),

    ((16, 8, 16), (2, 8192, 29696)),
    ((16, 64, 32), (2, 8192, 29696)),
    ((16, 64, 64), (2, 8192, 29696)),
    ((16, 128, 32), (2, 8192, 29696)),
    ((16, 128, 64), (2, 8192, 29696)),
    ((16, 256, 32), (2, 8192, 29696)),

    ((16, 8, 16), (3, 8192, 29696)),
    ((16, 64, 32), (3, 8192, 29696)),
    ((16, 64, 64), (3, 8192, 29696)),
    ((16, 128, 32), (3, 8192, 29696)),
    ((16, 128, 64), (3, 8192, 29696)),
    ((16, 256, 32), (3, 8192, 29696)),

    ((16, 8, 16), (4, 8192, 29696)),
    ((16, 64, 32), (4, 8192, 29696)),
    ((16, 64, 64), (4, 8192, 29696)),
    ((16, 128, 32), (4, 8192, 29696)),
    ((16, 128, 64), (4, 8192, 29696)),
    ((16, 256, 32), (4, 8192, 29696)),

    ((16, 8, 16), (8, 8192, 29696)),
    ((16, 64, 32), (8, 8192, 29696)),
    ((16, 64, 64), (8, 8192, 29696)),
    ((16, 128, 32), (8, 8192, 29696)),
    ((16, 128, 64), (8, 8192, 29696)),
    ((16, 256, 32), (8, 8192, 29696)),

    ((16, 8, 16), (16, 8192, 29696)),
    ((16, 64, 32), (16, 8192, 29696)),
    ((16, 64, 64), (16, 8192, 29696)),
    ((16, 128, 32), (16, 8192, 29696)),
    ((16, 128, 64), (16, 8192, 29696)),
    ((16, 256, 32), (16, 8192, 29696)),

    ((16, 8, 16), (2048, 8192, 29696)),
    ((64, 64, 32), (2048, 8192, 29696)),
    ((64, 64, 64), (2048, 8192, 29696)),
    ((64, 128, 32), (2048, 8192, 29696)),
    ((64, 256, 32), (2048, 8192, 29696)),
    ((128, 64, 32), (2048, 8192, 29696)),
    ((128, 128, 32), (2048, 8192, 29696)),
    ((128, 128, 64), (2048, 8192, 29696)),
]

SMs = 82

for ((CtaTileM, CtaTileN, CtaTileK), (GemmM, GemmN, GemmK)) in ThreadblockShape_and_ProblemShape:
    FLOPs = 2 * GemmM * GemmN * GemmK 
    FLOPS = 71.16288e9 if GemmM > 16 else 71.16288e9 * (GemmM / 16)
    print(f'FLOPS: {FLOPS}')
    ms = FLOPs / FLOPS
    Bytes = ((GemmN / CtaTileN) * GemmM * GemmK + ((GemmM + CtaTileM - 1) // CtaTileM) * GemmK * GemmN + 2 * GemmM * GemmN)
    GB_per_second = (Bytes / ms * 1e3) / 1024 / 1024 / 1024
    if GemmM < CtaTileM:
        Arithmetic_intensity = 1 / (1 / (2 * CtaTileN) + ((GemmM + CtaTileM - 1) // CtaTileM) / (2 * GemmM) + 1 / GemmK)
    else:
        Arithmetic_intensity = 1 / (1 / (2 * CtaTileN) + 1 / (2 * CtaTileM) + 1 / GemmK)
    L2_cache_Bandwidth = 4830
    Global_Memory_Bandwidth = 936
    L2_cache_hit = 100 * (GB_per_second - Global_Memory_Bandwidth) /(L2_cache_Bandwidth - Global_Memory_Bandwidth)
    if L2_cache_hit < 0:
        L2_cache_hit = 0.0
        MFU = (FLOPs / ms) / 71.16288e9 
    elif L2_cache_hit > 100.0:
        L2_cache_hit = 100.0
        MFU = FLOPs * (Global_Memory_Bandwidth / GB_per_second) / ms / 71.16288e9 
    else:
        MFU = (FLOPs / ms) / 71.16288e9 
    MFU *= 100
    print(f'{(CtaTileM, CtaTileN, CtaTileK)}, {(GemmM, GemmN, GemmK)}, ms: {ms:.3f}, Arithmetic_intensity: {Arithmetic_intensity:.3f}, GB_per_second: {GB_per_second:.3f}, L2_cache_hit: {L2_cache_hit:.3f}, MFU: {MFU}')
```

| ThreadblockShape | GemmShape           | Arithmetic intensity | GB_per_second | ms      | L2 cache hit(%) |  MFU(%) |
|------------------|---------------------|----------------------|---------------|---------|--------------|---------|
| (16, 8, 16)      | (1, 8192, 29696)    | 1.778                | 2330.141      | 0.109   | 35.802       | 6.25    |
| (16, 64, 32)     | (1, 8192, 29696)    | 1.969                | 2103.613      | 0.109   | 29.985       | 6.25    |
| (16, 64, 64)     | (1, 8192, 29696)    | 1.969                | 2103.613      | 0.109   | 29.985       | 6.25    |
| (16, 128, 32)    | (1, 8192, 29696)    | 1.984                | 2087.432      | 0.109   | 29.569       | 6.25    |
| (16, 128, 64)    | (1, 8192, 29696)    | 1.984                | 2087.432      | 0.109   | 29.569       | 6.25    |
| (16, 256, 32)    | (1, 8192, 29696)    | 1.992                | 2097.342      | 0.109   | 29.362       | 6.25    |
| (16, 8, 16)      | (2, 8192, 29696)    | 3.200                | 2589.169      | 0.109   | 42.454       | 12.25   |
| (16, 64, 32)     | (2, 8192, 29696)    | 3.878                | 2136.114      | 0.109   | 30.820       | 12.25   |
| (16, 64, 64)     | (2, 8192, 29696)    | 3.878                | 2136.114      | 0.109   | 30.820       | 12.25   |
| (16, 128, 32)    | (2, 8192, 29696)    | 3.938                | 2103.753      | 0.109   | 29.989       | 12.25   |
| (16, 128, 64)    | (2, 8192, 29696)    | 3.938                | 2103.753      | 0.109   | 29.989       | 12.25   |
| (16, 256, 32)    | (2, 8192, 29696)    | 3.968                | 2087.572      | 0.109   | 29.573       | 12.25   |
| (16, 8, 16)      | (3, 8192, 29696)    | 4.363                | 2848.198      | 0.109   | 49.106       | 18.75   |
| (16, 64, 32)     | (3, 8192, 29696)    | 5.730                | 2168.614      | 0.109   | 31.654       | 18.75   |
| (16, 64, 64)     | (3, 8192, 29696)    | 5.730                | 2168.614      | 0.109   | 31.654       | 18.75   |
| (16, 128, 32)    | (3, 8192, 29696)    | 5.861                | 2120.073      | 0.109   | 30.408       | 18.75   |
| (16, 128, 64)    | (3, 8192, 29696)    | 5.861                | 2120.073      | 0.109   | 30.408       | 18.75   |
| (16, 256, 32)    | (3, 8192, 29696)    | 5.929                | 2095.802      | 0.109   | 29.784       | 18.75   |
| (16, 8, 16)      | (4, 8192, 29696)    | 5.332                | 3107.227      | 0.109   | 55.758       | 25.0    |
| (16, 64, 32)     | (4, 8192, 29696)    | 7.528                | 2201.115      | 0.109   | 32.489       | 25.0    |
| (16, 64, 64)     | (4, 8192, 29696)    | 7.528                | 2201.115      | 0.109   | 32.489       | 25.0    |
| (16, 128, 32)    | (4, 8192, 29696)    | 7.756                | 2136.393      | 0.109   | 30.827       | 25.0    |
| (16, 128, 64)    | (4, 8192, 29696)    | 7.756                | 2136.393      | 0.109   | 30.827       | 25.0    |
| (16, 256, 32)    | (4, 8192, 29696)    | 7.875                | 2104.031      | 0.109   | 29.996       | 25.0    |
| (16, 8, 16)      | (8, 8192, 29696)    | 7.998                | 4143.341      | 0.109   | 82.366       | 50.0    |
| (16, 64, 32)     | (8, 8192, 29696)    | 14.215               | 2331.117      | 0.109   | 35.827       | 50.0    |
| (16, 64, 64)     | (8, 8192, 29696)    | 14.215               | 2331.117      | 0.109   | 35.827       | 50.0    |
| (16, 128, 32)    | (8, 8192, 29696)    | 15.051               | 2201.673      | 0.109   | 32.503       | 50.0    |
| (16, 128, 64)    | (8, 8192, 29696)    | 15.051               | 2201.673      | 0.109   | 32.503       | 50.0    |
| (16, 256, 32)    | (8, 8192, 29696)    | 15.507               | 2136.951      | 0.109   | 30.841       | 50.0    |
| (16, 8, 16)      | (16, 8192, 29696)   | 7.998                | 4143.341      | 0.109   | 100.00       | 15.059 |
| (16, 64, 32)     | (16, 8192, 29696)   | 14.215               | 2331.117      | 0.109   | 42.504       | 100.0  |
| (16, 64, 64)     | (16, 8192, 29696)   | 14.215               | 2331.117      | 0.109   | 42.504       | 100.0  |
| (16, 128, 32)    | (16, 8192, 29696)   | 15.051               | 2201.673      | 0.109   | 35.856       | 100.0  |
| (16, 128, 64)    | (16, 8192, 29696)   | 15.051               | 2201.673      | 0.109   | 35.856       | 100.0  |
| (16, 256, 32)    | (16, 8192, 29696)   | 15.507               | 2136.951      | 0.109   | 32.532       | 100.0  |
| (16, 8, 16)      | (2048, 8192, 29696) | 10.663               | 6215.569      | 14.002  | 100.000      | 15.059 |
| (64, 64, 32)     | (2048, 8192, 29696) | 63.862               | 1037.788      | 14.002  | 2.614        | 100.0  |
| (64, 64, 64)     | (2048, 8192, 29696) | 63.862               | 1037.788      | 14.002  | 2.614        | 100.0  |
| (64, 128, 32)    | (2048, 8192, 29696) | 85.089               | 778.899       | 14.002  | 0            | 100.0  |
| (64, 256, 32)    | (2048, 8192, 29696) | 102.048              | 649.454       | 14.002  | 0            | 100.0  |
| (128, 64, 32)    | (2048, 8192, 29696) | 85.089               | 778.899       | 14.002  | 0            | 100.0  |
| (128, 128, 32)   | (2048, 8192, 29696) | 127.451              | 520.010       | 14.002  | 0            | 100.0  |
| (128, 128, 64)   | (2048, 8192, 29696) | 127.451              | 520.010       | 14.002  | 0            | 100.0  |


# 实现

实际实现，M<=16 时，MFU远远没有达到理论值
## block matmul

[Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)

## persistent matmul

[persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)

## Stream-K matmul

[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/pdf/2301.03598)

# 参考文献
[pytorch linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

[Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)

[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/pdf/2301.03598)

[CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)

[Warp Scheduling Basics](https://www.cs.ucr.edu/~nael/217-f19/lectures/WarpScheduling.pptx)

[Dissecting the CUDA scheduling hierarchy: a Performance and Predictability Perspective](https://conferences.computer.org/cpsiot/pdfs/RTAS2020-4uXAu5nqG7QNiz5wFYyfj6/549900a210/549900a210.pdf)

[DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100](https://developer.download.nvidia.cn/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)

[GPU抢占式任务调度](https://zhuanlan.zhihu.com/p/657205626)

[CUDA C/C++ Streams and Concurrency](https://developer.download.nvidia.cn/CUDA/training/StreamsAndConcurrencyWebinar.pdf)