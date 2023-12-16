#ifndef CUDAMATMULTEMP_H
#define CUDAMATMULTEMP_H
#include<iostream>
#include "./CudaMatMul.h"

template<typename T>
void t_cuda_mmul(const T* A, std::size_t A_n, std::size_t A_m, const T* B, std::size_t B_n, std::size_t B_m, T* C, bool _obvious = false){
    throw std::bad_typeid();
}


template<>
void t_cuda_mmul<double>(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C, bool _obvious){
    if(_obvious){
        obvious_cuda_mmul_d64(A, A_n, A_m, B, B_n, B_m, C);
    } else {
        cuda_mmul_d64(A, A_n, A_m, B, B_n, B_m, C);
    }
}

template<>
void t_cuda_mmul<float>(const float* A, std::size_t A_n, std::size_t A_m, const float* B, std::size_t B_n, std::size_t B_m, float* C, bool _obvious){
    if(_obvious){
        obvious_cuda_mmul_fp32(A, A_n, A_m, B, B_n, B_m, C);
    } else {
        cuda_mmul_fp32(A, A_n, A_m, B, B_n, B_m, C);
    }
}

#endif //CUDAMATMULTEMP_H