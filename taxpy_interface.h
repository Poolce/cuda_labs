#ifndef TAXPY_INTERFACE__H
#define TAXPY_INTERFACE__H

#include <stdio.h>
#include <random>
#include <omp.h>

//Result struct
struct timeResult{
    int vectorSize;
    double seqTimeResult;
    double ompTimeResult;
    double cudaTimeResultWithLoad;
    double cudaTimeResultWithoutLoad;

    void out(){
        printf("----------------------------------------------------\n");
        printf("seqTimeResult \t\t%lf\n",seqTimeResult);
        printf("ompTimeResult \t\t%lf\n",ompTimeResult);
        printf("cudaTimeResultWithLoad \t\t%lf\n",cudaTimeResultWithLoad);
        printf("cudaTimeResultWithoutLoad \t\t%lf\n",cudaTimeResultWithoutLoad);
        printf("----------------------------------------------------\n\n");
    }
};

//Vector operations
template <typename T>
void vector_outs(int size, T* vec){
  printf("{");
  for(int i = 0; i < size; i++)
    printf(" %lf",vec[i]," ");
    printf("}\n");
}

template <typename T>
extern T* get_rand_vector(int size){
  T* res = new T[size];
  for(int i = 0; i < size; i++){
    res[i] = (T)rand()/(T)rand();
  }
  return res;
}

//OpenMP version (for sequential execution set parameter nom_threads = 1)
template <typename T>
void taxpy(int n, T* X, int Xinc, T* Y, int Yinc, T alpfa){
    int op_nom = std::ceil((T)((double)(n) / (double)(std::max(Xinc,Yinc))));
    #pragma omp parallel for
    for(int i = 0; i < op_nom; i++){
        Y[i*Yinc]+=alpfa*X[i*Xinc];
    }
}

#endif //TAXPY_INTERFACE__H