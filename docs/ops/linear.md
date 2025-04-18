
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

# MFU
# Arithmetic_intensity
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

Peak FP32 TFLOPS / SM = 2 * (CUDA Cores / GPU) = 2 * 1695 FLOPS

Peak FP16 Tensor TFLOPS with FP32 Accumulate / SM = 2 * 1695 * 2  = 6780 FLOPS

Peak FP16 Tensor TFLOPS with FP32 Accumulate / Tensor Cores = 1695 FLOPS

Shared Memory / SM: Shared Memory / SMs = 10752 KB / 82 = 131.122 KB

Shared Memory Bandwidth: 128 bytes/clock per SM

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

数据在Register File中， 

Shared Memory Bandwidth 是 128 bytes/clock per SM, 没有找到Register File Bandwidth. 一般情况下Register File Bandwidth 比 Shared Memory Bandwidth 高一个数量级

$InstructionShape=(MmaM， MmaN, MmaK)=(16, 8, 16)$

$WarpShape=(WarpTileM, WarpTileN, WarpTileK)=(num_MmaM * MmaM, num_MmaN, num_MmaK)$

$FLOPs\_per\_SM = 2 * WarpTileM * WarpTileN * WarpTileK$

$clcs = \frac{FLOPs\_per\_SM}{6780}$

$Arithmetic\_intensity = $ 

$\frac{FLOPs\_per\_SM} {2 * (WarpTileM * WarpTileK + WarpTileK * WarpTileN + (num\_MmaK) * WarpTileM * WarpTileN)}$

$\frac{WarpTileM * WarpTileN * WarpTileK}{WarpTileM * WarpTileK + WarpTileK * WarpTileN + num\_MmaK * WarpTileM * MmaN}$

$=\frac{1}{1 / WarpTileM + 1 / WarpTileN + num\_MmaK / WarpTileK}$

$=\frac{1}{1 / WarpTileM + 1 / WarpTileN + 1 / MmaK}$

$Bytes\_per\_clc = \frac{FLOPs\_per\_SM}{Arithmetic\_intensity}/clcs$

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
        clcs = FLOPs_per_SM / 6780
        Arithmetic_intensity = 1 / (1 / WarpTileM + 1 / WarpTileN + 1 / MmaK)
        Bytes_per_clc = FLOPs_per_SM / Arithmetic_intensity /clcs
        print(f'{(WarpTileM, WarpTileN, WarpTileK)}, Arithmetic_intensity: {Arithmetic_intensity:.3f}, Bytes_per_clc: {Bytes_per_clc:.3f}')

```

| WarpShape     | InstructionShape | Arithmetic intensity | Bytes_per_clc |
|---------------|------------------|----------------------|---------------|
| (16, 8, 16)   | (16, 8, 16)      | 4.0                  | 1695          |
| (64, 64, 32)  | (16, 8, 16)      | 10.667               | 635.625       |
| (32, 128, 32) | (16, 8, 16)      | 9.846                | 688.594       |
| (16, 64, 32)  | (16, 8, 16)      | 7.111                | 953.438       |
| (16, 128, 32) | (16, 8, 16)      | 7.529                | 900.469       |
| (32, 32, 32)  | (16, 8, 16)      | 8.000                | 847.500       |
| (64, 64, 64)  | (16, 8, 16)      | 10.667               | 635.625       |
| (32, 32, 64)  | (16, 8, 16)      | 8.000                | 847.500       |
| (64, 32, 32)  | (16, 8, 16)      | 9.143                | 741.562       |



# 实现
## block matmul

## persistent matmul

## Stream-K matmul

# 参考文献
[pytorch linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

[Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)

[Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU](https://arxiv.org/pdf/2301.03598)

[CUTLASS Tutorial: Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
