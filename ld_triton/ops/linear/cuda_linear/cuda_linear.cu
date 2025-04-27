
#ifndef LD_TRITON_CUDA_LINEAR_CUH
#define LD_TRITON_CUDA_LINEAR_CUH
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

#ifdef DEBUG_MAIN
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#endif // DEBUG_MAIN

constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

__device__ inline bool thread0() {
    return (threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y == static_cast<unsigned int>(0)) && \
           (blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y) == static_cast<unsigned int>(0);
}

template <
  const int threads, // number of threads in a threadblock
  const int thread_m_blocks, // number of 16x16 blocks in the m dimension (batchsize) of the threadblock 
  const int thread_n_blocks, // same for n dimension (output) 
  const int thread_k_blocks, // same for k dimension (reduction)
  const int stages, // number of stages for the async global->shared fetch pipeline
  const int group_blocks = -1 // number of consecutive 16x16 blocks with a separate quantization scale
>
__global__ void cuda_linear_kernel(
    const half* __restrict__ A, // fp16 input matrix of shape mxk 
    const half* __restrict__ B, // 4bit quantized weight matrix of shape kxn 
          half* __restrict__ C, // fp16 output buffer of shape mxn
    int  prob_m, // batch dimension m
    int  prob_n, // output dimension n
    int  prob_k, // reduction dimension k
    int* locks // extra global storage for barrier synchronization 
) {
    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    if (thread0()) {
        printf("prob: (%d, %d, %d)\n", prob_m, prob_n, prob_k);
    }
}

const int THREADS = 256;
const int STAGES = 4;
const int SHARED_MEM = 96 * 1024;

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int linear_cuda(
    const void *A,
    const void *B,
          void *C,
    int prob_m,
    int prob_n,
    int prob_k,
    void *workspace,
    int groupsize = -1,
    int dev = 0,
    cudaStream_t stream = 0,
    int thread_k = -1,
    int thread_n = -1,
    int sms = -1
) {
    int thread_m = 16;
    int tot_m = prob_m;
    int tot_m_blocks = ceildiv(tot_m, thread_m);
    int pad = thread_m * tot_m_blocks - tot_m;

    if (sms == -1) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    printf("sms: %d\n", sms);
    if (thread_k == -1 || thread_n == -1) {
        if (prob_m <= thread_m) {
            thread_k = 128;
            thread_n = 128;
        } else {
            thread_k = 64;
            thread_n = 256;
        }
    }

    int thread_k_blocks = prob_k / thread_k;
    int thread_n_blocks = prob_n / thread_n;
    int group_blocks = (groupsize == -1) ? -1 : groupsize / thread_m;
    int blocks = sms;

    if (prob_n % thread_n != 0 || prob_k % thread_k != 0 || (group_blocks != -1 && prob_k % group_blocks != 0))
        return ERR_PROB_SHAPE;
    if (prob_m == 0 || prob_n == 0 || prob_k == 0)
        return 0;

    if (tot_m_blocks > 4) {
        return ERR_PROB_SHAPE;
    }
    int thread_m_blocks = tot_m_blocks;
    int* locks = (int*) workspace;
    half * A_ptr = (half *) A;
    half * B_ptr = (half *) B;
    half * C_ptr = (half *) C;
    if (false) {

    } else if (thread_m_blocks == 1 && thread_n_blocks == 8 && thread_k_blocks == 8 && group_blocks == -1) {
        // M = 16, N = 128 * 8, k = 128 * 8
        constexpr int THREAD_M_BLOCKS = 1;
        constexpr int THREAD_N_BLOCKS = 8;
        constexpr int THREAD_K_BLOCKS = 8;
        constexpr int GROUP_BLOCKS = -1;
        cudaFuncSetAttribute( 
            cuda_linear_kernel<THREADS, 1, 8, 8, STAGES, -1>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            SHARED_MEM 
        );
        cuda_linear_kernel<
            THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS 
        ><<<blocks, THREADS, SHARED_MEM, stream>>>(
            A_ptr, B_ptr, C_ptr,
            prob_m, prob_n, prob_k,
            locks
        ); 
    }

}

// template <typename T>

#ifdef DEBUG_MAIN
// nvcc --std=c++17 -DDEBUG_MAIN -o cuda_linear cuda_linear.cu
int main(int argc, char ** argv) {
    // M = 16, N = 128 * 8, k = 128 * 8
    int M = 16;
    int N = 128 * 8;
    int K = 128 * 8;

    using TA = half;
    using TB = half;
    using TC = float;

    thrust::host_vector<TA> h_A(M*K);
    thrust::host_vector<TB> h_B(N*K);
    thrust::host_vector<TC> h_C(M*N);

    for (int j = 0; j < M*K; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < N*K; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < M*N; ++j) h_C[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    linear_cuda(
        d_A.data().get(),
        d_B.data().get(),
        d_C.data().get(),
        M, N, K,
        nullptr
    );
    return 0;
}
#endif


#endif // LD_TRITON_CUDA_LINEAR_CUH