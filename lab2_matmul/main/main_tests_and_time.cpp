#include "../inc/Matrix.h"
#include "../test/test.h"
#include "omp.h"
#include <iostream>

void RUN_TESTS(){
    test((char*)"OMP AND SEQUENTIAL TEST MATRIX A(900,100) B(100,900)", [](){
        Matrix<float> A(900,100,true);
        Matrix<float> B(100,900,true);
        Matrix<float> C = omp_mmul(A,B,6);
        Matrix<float> D = seq_mmul(A,B);
        return C==D;
    });

    test((char*)"CUDA-OBVIOUS AND OMP TEST MATRIX A(2048,1024) B(1024,512)", [](){
        Matrix<float> A(2048,1024,true);
        Matrix<float> B(1024,512,true);

        auto C = omp_mmul(A,B,6);
        auto D = obvious_cuda_mmul(A,B);

        return C==D;
    });

    test((char*)"CUDA-OBVIOUS AND CUDA-SHARED TEST MATRIX A(2048,1024) B(1024,512) FLOAT", [](){
        Matrix<float> A(2048,1024,true);
        Matrix<float> B(1024,512,true);

        auto C = shared_cuda_mmul(A,B);
        auto D = obvious_cuda_mmul(A,B);

        return C==D;
    });
    
    test((char*)"CUDA-OBVIOUS AND CUDA-SHARED TEST MATRIX A(2048,1024) B(1024,512) DOUBLE", [](){
        Matrix<double> A(2048,1024,true);
        Matrix<double> B(1024,512,true);

        auto C = shared_cuda_mmul(A,B);
        auto D = obvious_cuda_mmul(A,B);
        
        return C==D;
    });
}

template<typename T>
void TimeExperiment(std::size_t A_n, std::size_t A_m, std::size_t B_n, std::size_t B_m){
    Matrix<T> A(A_n, A_m, true);
    Matrix<T> B(B_n, B_m, true);
    
    double TimeStart = omp_get_wtime();
    Matrix OmpObvRes = omp_mmul(A,B,6);
    double OMPTimeResult = omp_get_wtime() - TimeStart;

    TimeStart = omp_get_wtime();
    Matrix CudaObvRes = obvious_cuda_mmul(A,B);
    double obviousCudaTimeResult = omp_get_wtime() - TimeStart;

    TimeStart = omp_get_wtime();
    Matrix shared_cuda_res = shared_cuda_mmul(A,B);
    double sharedCudaTimeResult = omp_get_wtime() - TimeStart;

    printf("\t----------------------------------------------------\n\n");
    printf("\tOMPTimeResult \t\t\t\t%lfsec\n",OMPTimeResult);
    printf("\tobviousCudaTimeResult \t\t\t%lfsec\n",obviousCudaTimeResult);
    printf("\tsharedCudaTimeResult \t\t\t%lfsec\n",sharedCudaTimeResult);
    printf("\t----------------------------------------------------\n\n");
}


int main(){
    RUN_TESTS();


    printf("\t\x1b[32;47mTime Experiment Float type A(%i, %i) B(%i,%i)\t\x1b[35;47m\x1b[0m\n",1024, 256, 256, 512);
    TimeExperiment<float>(1024, 256, 256, 512);

    printf("\t\x1b[32;47mTime Experiment Double type A(%i, %i) B(%i,%i)\t\x1b[35;47m\x1b[0m\n", 1024, 256, 256, 512);
    TimeExperiment<double>(1024, 256, 256, 512);

    printf("\t\x1b[32;47mTime Experiment Float type A(%i, %i) B(%i,%i)\t\x1b[35;47m\x1b[0m\n", 1024, 512, 512, 1024);
    TimeExperiment<float>(1024, 512, 512, 1024);

    printf("\t\x1b[32;47mTime Experiment Double type A(%i, %i) B(%i,%i)\t\x1b[35;47m\x1b[0m\n", 1024, 512, 512, 1024);
    TimeExperiment<double>(1024, 512, 512, 1024);
}