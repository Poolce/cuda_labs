#include "../inc/CudaMatMul.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <cuda_profiler_api.h>

template<int block_size, typename T>
__global__ void  ObviousMatrixMulKernel(const T* A, std::size_t A_m, const T* B, std::size_t B_m, T* C){
    std::size_t id_x = blockIdx.x * block_size + threadIdx.x;
    std::size_t id_y = blockIdx.y * block_size + threadIdx.y;
    
    T C_XY_Element = 0;
    #pragma unroll
    for(std::size_t i = 0; i < A_m; i++)
        C_XY_Element += A[A_m * id_x + i]*B[B_m * i + id_y];

    C[id_x * B_m + id_y] = C_XY_Element;
}

template<int block_size, typename T>
__global__ void  MatrixMulKernel(const T* A, std::size_t A_m, const T* B, std::size_t B_m, T* C){
    
    int Bx = blockIdx.x;
    int By = blockIdx.y;

    int Tx = threadIdx.x;
    int Ty = threadIdx.y;
    
    
    std::size_t ABlocksLineBegin = Bx * A_m * block_size;
    std::size_t ABlocksLineEnd = ABlocksLineBegin + A_m;
    std::size_t ABlocksLineStep = block_size;

    std::size_t BBlocksLineBegin = By * block_size;
    std::size_t BBlocksLineStep = B_m * block_size;

    T C_XY_Element = 0;

    std::size_t A_block_iter = ABlocksLineBegin;
    std::size_t B_block_iter = BBlocksLineBegin;

    for(;A_block_iter < ABlocksLineEnd; A_block_iter += ABlocksLineStep, B_block_iter += BBlocksLineStep){
        
        __shared__ T A_Shared[block_size][block_size];
        __shared__ T B_Shared[block_size][block_size];
        
        A_Shared[Tx][Ty] = A[A_block_iter + A_m * Tx + Ty];
        B_Shared[Tx][Ty] = B[B_block_iter + B_m * Tx + Ty];
        
        __syncthreads();
    #pragma unroll
        for(std::size_t k = 0; k < block_size; k++)
            C_XY_Element += A_Shared[Tx][k] * B_Shared[k][Ty];

        __syncthreads();

    }
    std::size_t CBlockBegin = B_m * block_size * Bx + block_size * By;

    C[CBlockBegin + B_m * Ty + Tx] = C_XY_Element;
}



template<typename T>
void launch_cuda_mmul(const T* A, std::size_t A_n, std::size_t A_m, const T* B, std::size_t B_n, std::size_t B_m, T* C, bool is_obvious){
    const std::size_t block_size = 16;

    cudaError_t err = cudaSuccess;
    T *gpuA, *gpuB, *gpuC;

    // MEMORY ALLOC
    err = cudaMalloc((void**)&gpuA, A_n * A_m * sizeof(T));
    if (err != cudaSuccess){
        printf("gpuA memory allocation error. ");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&gpuB, B_n * B_m * sizeof(T));
    if (err != cudaSuccess){
        printf("gpuA memory allocation error. ");
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void**)&gpuC, A_n * B_m * sizeof(T));
    if (err != cudaSuccess){
        printf("gpuC memory allocation error. ");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpuA, A, A_n * A_m * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("gpuA memory relocation error. Host to device.");
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(gpuB, B, B_n * B_m * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("gpuB memory relocation error. Host to device.");
        exit(EXIT_FAILURE);
    }


    dim3 blocks(block_size, block_size);
    dim3 grid(A_n/block_size, B_m/block_size);

    if(is_obvious){
        ObviousMatrixMulKernel<block_size,T><<<grid, blocks>>>(gpuA, A_m, gpuB, B_m, gpuC);
    } else {
        MatrixMulKernel<block_size,T><<<grid, blocks>>>(gpuA, A_m, gpuB, B_m, gpuC);
    }

    err = cudaMemcpy(C, gpuC, A_n * B_m * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("gpuC memory relocation error. Device to host. %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaFree(gpuA);
    if (err != cudaSuccess){
        printf("gpuA destruction error. ");
        exit(EXIT_FAILURE);
    }
    err = cudaFree(gpuB);
    if (err != cudaSuccess){
        printf("gpuB destruction error. ");
        exit(EXIT_FAILURE);
    }
    err = cudaFree(gpuC);
    if (err != cudaSuccess){
        printf("gpuC destruction error. ");
        exit(EXIT_FAILURE);
    }
}


void obvious_cuda_mmul_fp32(const float* A, std::size_t A_n, std::size_t A_m, const float* B, std::size_t B_n, std::size_t B_m, float* C){
    launch_cuda_mmul<float>(A, A_n, A_m, B, B_n, B_m, C, true);
}

void cuda_mmul_fp32(const float* A, std::size_t A_n, std::size_t A_m, const float* B, std::size_t B_n, std::size_t B_m, float* C){
    launch_cuda_mmul<float>(A, A_n, A_m, B, B_n, B_m, C, false);
}

void obvious_cuda_mmul_d64(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C){
    launch_cuda_mmul<double>(A, A_n, A_m, B, B_n, B_m, C, true);
}

void cuda_mmul_d64(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C){
    launch_cuda_mmul<double>(A, A_n, A_m, B, B_n, B_m, C, false);
}