#include <iostream>
#include "../inc/Matrix.h"

int main(){
    auto A = Matrix<float>(16384,8192);
    auto B = Matrix<float>(8192,4096);

    auto C = obvious_cuda_mmul(A,B);
}