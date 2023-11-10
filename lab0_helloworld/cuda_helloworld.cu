#include <cuda_runtime.h>
#include <stdio.h>

__global__ void taxpy_kernel(int arr_size, int* gpu_arr){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world. I am from %d block, %d thread (global index: %d)\n", blockIdx.x, threadIdx.x, i);
    if(i<arr_size) gpu_arr[i]+=i;
}

void cuda_helloworld(int arr_size,int* arr, int blocksPerGrid, int threadsPerBlock){
    cudaError_t err = cudaSuccess;

    //Allocation memory
    int* gpu_arr;
    err = cudaMalloc((void**)&gpu_arr, arr_size*sizeof(int));
    if (err != cudaSuccess){
        printf("gpuX memory allocation error. ");
        exit(EXIT_FAILURE);
    }

    //Relocation memory
    err = cudaMemcpy(gpu_arr, arr, arr_size*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("Array memory relocation error. Host to device.\n%s",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //Launch kernel
    taxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(arr_size, gpu_arr);
    cudaDeviceSynchronize();

    //memory relocation Device to host
    err = cudaMemcpy(arr, gpu_arr, arr_size*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("Array memory relocation error. Device to host.\n%s",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //freeing memory 
    err = cudaFree(gpu_arr);
    if (err != cudaSuccess){
        printf("Array destruction error.\n%s",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(){
    int arr_size = 20;
    int* arr = new int[arr_size];
    for(int i = 0; i < arr_size; i++)
        arr[i] = i;

    cuda_helloworld(arr_size, arr, 5, 4);

    printf("[");
    for(int i = 0; i < arr_size; i++){
        if(i==(arr_size-1)){
            printf("%d]\n",arr[i]);
            break;
        } else {
            printf("%d, ",arr[i]);
        }
    }
}