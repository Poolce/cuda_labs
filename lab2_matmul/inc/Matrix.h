#ifndef MATRIX_H
#define MATRIX_H
#include <random>
#include <iostream>
#include <omp.h>
#include <exception>

#include "./Buffer2D.h"
#include "./CudaMatMulTemp.h"



template<typename T>
class Matrix{
protected:
    Buffer2D<T> data;

public:
    Matrix(int n = 0, int m = 0, bool randomize = false);

    Matrix<T> get_transpose();

    friend bool operator==(const Matrix& lhs, const Matrix& rhs){
        int m, n;
        if((m = lhs.data.m != rhs.data.m) || (n = lhs.data.n != rhs.data.n))
            return false;

        for(int i = 0; i < n; i++)
            for(int j = 0; j < m; j++)
                if(lhs.data.at(i,j) != rhs.data.at(i,j))
                    return false;


        return true;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix& rhs){
        for(int i = 0; i < rhs.data.n; i++){
            os<<"\t";
            for(int j = 0; j < rhs.data.m; j++){
                os<<rhs.data.at(i,j)<<" ";
            }
            os<<"\n";
        }
        return os;
    }

    friend Matrix seq_mmul(const Matrix<T>& A, const Matrix<T>& B){
        if (A.data.m != B.data.n)
            throw std::invalid_argument("Length of A matrix row should equel length of B matrix column.");

        Matrix<T> C(A.data.n, B.data.m);

        for(int i = 0; i < A.data.n; i++)
            for(int j = 0; j < B.data.m; j++)
                for(int k = 0; k < A.data.m; k++)
                    C.data.at(i,j) += A.data.at(i,k) * B.data.at(k,j);

        return C;
    }

    friend Matrix<T> omp_mmul(const Matrix<T>& A, const Matrix<T>& B, int omp_thread_nom = 6){
        if (A.data.m != B.data.n)
            throw std::invalid_argument("Length of A matrix row should equel length of B matrix column.");

        Matrix<T> C(A.data.n, B.data.m);

        omp_set_num_threads(omp_thread_nom);
        #pragma omp parallel for
        for(int i = 0; i < A.data.n; i++)
            for(int j = 0; j < B.data.m; j++)
                for(int k = 0; k < A.data.m; k++)
                    C.data.at(i,j) += A.data.at(i,k) * B.data.at(k,j);

        return C;
    }

    friend Matrix<T> obvious_cuda_mmul(const Matrix<T>& A, const Matrix<T>& B){
        if (A.data.m != B.data.n)
            throw std::invalid_argument("Length of A matrix row should equel length of B matrix column.");
        
        Matrix<T> C(A.data.n, B.data.m);

        T* Abuf = A.data.buf.buf;
        T* Bbuf = B.data.buf.buf;
        T* Cbuf = C.data.buf.buf;

        t_cuda_mmul<T>(Abuf, A.data.n, A.data.m, Bbuf, B.data.n, B.data.m, Cbuf,true);


        return C;
    }
    
    friend Matrix<T> shared_cuda_mmul(const Matrix<T>& A, const Matrix<T>& B){
        if (A.data.m != B.data.n)
            throw std::invalid_argument("Length of A matrix row should equel length of B matrix column.");
        
        Matrix<T> C(A.data.n, B.data.m);

        T* Abuf = A.data.buf.buf;
        T* Bbuf = B.data.buf.buf;
        T* const Cbuf = C.data.buf.buf;
        
        t_cuda_mmul<T>(Abuf, A.data.n, A.data.m, Bbuf, B.data.n, B.data.m, Cbuf,false);

        return C;
    }
};


template<typename T>
Matrix<T>::Matrix(int n, int m, bool randomize):data(n,m){
    T Tzero;
    if (!(Tzero = static_cast<T>(1))){
        throw std::invalid_argument("Type of matrix elements should have standart zero.");
    }
    std::random_device dev;
    std::mt19937 gen(dev());
    if(randomize){
        for (int i = 0; i < n; i++) { 
            for (int j = 0; j < m; j++){
                data.at(i,j) = static_cast<T>((gen()%1000))/100;
            }
        }
    } else {
        for (int i = 0; i < n; i++) { 
            for (int j = 0; j < m; j++){
                data.at(i,j) = Tzero;
            }
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::get_transpose(){
    Matrix<T> Result(data.m,data.n);

    for(int i = 0; i < data.n; i++)
        for(int j = 0; j < data.m; j++)
            Result.data.at(j,i) = data.at(i,j);
    
    return Result;
}


#endif //MATRIX_H