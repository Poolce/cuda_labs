#include <cuda_runtime.h>
#include <stdio.h>
#include "taxpy_interface.h"
#include "omp.h"

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

    int devCount = -1;
    cudaGetDeviceCount(&devCount);
    printf("\n%d\n", devCount);

    //memory allocation
    T *gpuX, *gpuY;
    err = cudaMalloc((void**)&gpuX, n*sizeof(T));
    if (err != cudaSuccess){
        fprintf(stderr, "gpuX memory allocation error. ", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&gpuY, n*sizeof(T));
    if (err != cudaSuccess){
        fprintf(stderr, "gpuY memory allocation error. ", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //memory relocation Host to device
    err = cudaMemcpy(gpuX, X, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "gpuX memory relocation error. Host to device.", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(gpuY, X, n*sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        fprintf(stderr, "gpuY memory relocation error. Host to device.", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Launch kernel and mark the time
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    double start = omp_get_wtime();
    taxpy_kernel<T><<<blocksPerGrid, threadsPerBlock>>>(n, gpuX, Xinc, gpuY, Yinc, alpfa);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr, "Kernel launch error", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    double end = omp_get_wtime();


    //memory relocation Device to host
    err = cudaMemcpy(Y, gpuY, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        fprintf(stderr, "gpuY memory relocation error. Device to host.", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //freeing memory 
    err = cudaFree(gpuX);
    if (err != cudaSuccess){
        fprintf(stderr, "gpuX destruction error. ", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaFree(gpuX);
    if (err != cudaSuccess){
        fprintf(stderr, "gpuY destruction error. ", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //time return
    return end - start;
}



///////////////////////EXP_RESULTS///////////////////////////////
template <typename T>
timeResult GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa){
    T* vecA = get_rand_vector<T>(vectorSize);
    T* vecB = get_rand_vector<T>(vectorSize);

    //////////////SEQ
    omp_set_num_threads(1);
    double TimeStart = omp_get_wtime();
    taxpy(vectorSize, vecA, Xinc, vecB, Yinc, alpfa);
    double TimeEnd = omp_get_wtime();
    double seqResult = TimeEnd - TimeStart;

    //////////////OMP_PARALLEL
    omp_set_num_threads(4);
    TimeStart = omp_get_wtime();
    taxpy(vectorSize, vecA, Xinc, vecB, Yinc, alpfa);
    TimeEnd = omp_get_wtime();
    double ompResult = TimeEnd - TimeStart;

    ////////////CUDA
    TimeStart = omp_get_wtime();
    double CudaTimeWithoutLoad = cuda_taxpy(vectorSize, vecA, Xinc, vecB, Yinc, alpfa);
    TimeEnd = omp_get_wtime();
    double CudaTimeWithLoad = TimeEnd - TimeStart;

    timeResult res;
    {
        res.vectorSize = vectorSize;
        res.seqTimeResult = seqResult;
        res.ompTimeResult = ompResult;
        res.cudaTimeResultWithLoad = CudaTimeWithLoad;
        res.cudaTimeResultWithoutLoad = CudaTimeWithoutLoad;
    }
    return res;
}


timeResult GetDResultEXP(int vectorSize, int Xinc, int Yinc, double alpfa){
    return GetResultExp<double>(vectorSize, Xinc, Yinc, alpfa);
}

timeResult GetFResultEXP(int vectorSize, int Xinc, int Yinc, float alpfa){
    return GetResultExp<float>(vectorSize, Xinc, Yinc, alpfa);
}