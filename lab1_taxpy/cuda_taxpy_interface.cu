#include <cuda_runtime.h>
#include <stdio.h>
#include "omp.h"
#include "cuda_taxpy_template.h"

//Cuda version

template <typename T>
__global__ void taxpy_kernel(int n, T* X, int Xinc, T* Y, int Yinc, T alpfa){
    int op_nom = std::ceil(((double)(n) / (double)(max(Xinc,Yinc))));
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < op_nom)
        Y[i*Yinc]+=alpfa*X[i*Xinc];
}

template <typename T>
double cuda_taxpy(int n, T* X, int Xinc, T* Y, int Yinc, T alpfa){
    cudaError_t err = cudaSuccess;

    //memory allocation
    T *gpuX, *gpuY;
    err = cudaMalloc((void**)&gpuX, n*sizeof(T));
    if (err != cudaSuccess){
        printf("gpuX memory allocation error. ");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&gpuY, n*sizeof(T));
    if (err != cudaSuccess){
        printf("gpuY memory allocation error. ");
        exit(EXIT_FAILURE);
    }

    //memory relocation Host to device
    err = cudaMemcpy(gpuX, X, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("gpuX memory relocation error. Host to device.");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpuY, X, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("gpuY memory relocation error. Host to device.");
        exit(EXIT_FAILURE);
    }

    //Launch kernel and mark the time
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    double start = omp_get_wtime();
    taxpy_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(n, gpuX, Xinc, gpuY, Yinc, alpfa);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error");
        exit(EXIT_FAILURE);
    }

    //memory relocation Device to host
    err = cudaMemcpy(Y, gpuY, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("gpuY memory relocation error. Device to host.");
        exit(EXIT_FAILURE);
    }
    double end = omp_get_wtime();

    //freeing memory 
    err = cudaFree(gpuX);
    if (err != cudaSuccess){
        printf("gpuX destruction error. ");
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(gpuY);
    if (err != cudaSuccess){
        printf("gpuY destruction error. ");
        exit(EXIT_FAILURE);
    }

    //time return
    return end - start;
}


/* ????? To correct ????? */
double cuda_daxpy(int n, double* X, int Xinc, double* Y, int Yinc, double alpfa){
    return cuda_taxpy<double> (n, X, Xinc,Y, Yinc, alpfa);
}

double cuda_faxpy(int n, float* X, int Xinc, float* Y, int Yinc, float alpfa){
    return cuda_taxpy<float> (n, X, Xinc, Y, Yinc, alpfa);
}
/*------------------------*/