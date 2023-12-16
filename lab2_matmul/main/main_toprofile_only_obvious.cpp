#include <iostream>
#include "../inc/Matrix.h"

int main(){
    printf("\t\x1b[32;47mPROFILING OBVIOUS KERNEL\t\x1b[35;47m\x1b[0m\n");
    auto A = Matrix<float>(16384,8192);
    auto B = Matrix<float>(8192,4096);

    auto C = obvious_cuda_mmul(A,B);
}