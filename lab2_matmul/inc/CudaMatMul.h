#ifndef CUDAMATMUL_H
#define CUDAMATMUL_H
#include <iostream>

void cuda_mmul_d64(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C);
void cuda_mmul_fp32(const float* A, std::size_t A_n, std::size_t A_m, const float* B, std::size_t B_n, std::size_t B_m, float* C);

void obvious_cuda_mmul_fp32(const float* A, std::size_t A_n, std::size_t A_m, const float* B, std::size_t B_n, std::size_t B_m, float* C);
void obvious_cuda_mmul_d64(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C);


#endif //CUDAMATMUL_H