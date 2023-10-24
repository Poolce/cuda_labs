#include <random>
#include "taxpy_interface.h"
#include "cuda_taxpy_template.h"
#include "test.h"

void Run_tests(){
    test((char*)"FLOAT CPU AND GPU 100 ELEMENTS", [](){
        //init
        omp_set_num_threads(1);
        int vec_size = 100;
        float* vecA = get_rand_vector<float>(vec_size);
        float* vecB = get_rand_vector<float>(vec_size);

        //run
        float* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<float>(100, vecA, 2, vecB, 3, 4.53);

        //CPU res in copyB
        taxpy<float>(100, vecA, 2, copyB, 3, 4.53);
        //data destruction
        delete[] vecA;
        delete[] vecB;
        delete[] copyB;
        return is_equal(vec_size, vecB, copyB);
    });

    test((char*)"FLOAT CPU OMP4 AND GPU 1000 ELEMENTS", [](){
        //init
        omp_set_num_threads(4);
        int vec_size = 1000;
        float* vecA = get_rand_vector<float>(vec_size);
        float* vecB = get_rand_vector<float>(vec_size);

        //run
        float* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<float>(100, vecA, 4, vecB, 7, 4.53);

        //CPU res in copyB
        taxpy<float>(100, vecA, 2, copyB, 3, 4.53);
        //data destruction
        delete[] vecA;
        delete[] vecB;
        delete[] copyB;
        return is_equal(vec_size, vecB, copyB);
    });
    test((char*)"FLOAT CPU OMP4 AND GPU RANDOM PARAMETERS", [](){
        //init
        omp_set_num_threads(4);
        int vec_size = rand()%1800+200;
        float* vecA = get_rand_vector<float>(vec_size);
        float* vecB = get_rand_vector<float>(vec_size);

        //run
        float* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<float>(vec_size, vecA, rand()%20+5, vecB, rand()%20+5, 4.53);

        //CPU res in copyB
        taxpy<float>(vec_size, vecA, rand()%20+5, copyB, rand()%20+5, 4.53);

        return is_equal(vec_size, vecB, copyB);
    });
    test((char*)"DOUBLE CPU AND GPU 100 ELEMENTS", [](){
        //init
        omp_set_num_threads(1);
        int vec_size = 100;
        double* vecA = get_rand_vector<double>(vec_size);
        double* vecB = get_rand_vector<double>(vec_size);

        //run
        double* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<double>(100, vecA, 2, vecB, 3, 4.53);

        //CPU res in copyB
        taxpy<double>(100, vecA, 2, copyB, 3, 4.53);
        //data destruction
        delete[] vecA;
        delete[] vecB;
        delete[] copyB;
        return is_equal(vec_size, vecB, copyB);
    });
    
    test((char*)"DOUBLE CPU OMP4 AND GPU 1000 ELEMENTS", [](){
        //init
        omp_set_num_threads(4);
        int vec_size = 1000;
        double* vecA = get_rand_vector<double>(vec_size);
        double* vecB = get_rand_vector<double>(vec_size);

        //run
        double* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<double>(100, vecA, 4, vecB, 7, 4.53);

        //CPU res in copyB
        taxpy<double>(100, vecA, 2, copyB, 3, 4.53);
        //data destruction
        delete[] vecA;
        delete[] vecB;
        delete[] copyB;
        return is_equal(vec_size, vecB, copyB);
    });
    test((char*)"DOUBLE CPU OMP4 AND GPU RANDOM PARAMETERS", [](){
        //init
        omp_set_num_threads(4);
        int vec_size = rand()%1800+200;
        double* vecA = get_rand_vector<double>(vec_size);
        double* vecB = get_rand_vector<double>(vec_size);

        //run
        double* copyB = copy_vector(vec_size, vecB);
        //GPU res in vecB
        cuda_t_axpy<double>(vec_size, vecA, rand()%20+5, vecB, rand()%20+5, 4.53);

        //CPU res in copyB
        taxpy<double>(vec_size, vecA, rand()%20+5, copyB, rand()%20+5, 4.53);

        return is_equal(vec_size, vecB, copyB);
    });
}

int main(){
    //tests
    Run_tests();

    //Experiments
    //GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa, char* descr, int omp_thread_nom)
    GetResultExp<double>(10000, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 10000 ELEMENTS", 6).out();
    GetResultExp<double>(10000000, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 10000000 ELEMENTS", 6).out();
    GetResultExp<double>(100000000, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 100000000 ELEMENTS", 6).out();
    GetResultExp<float>(10000, 3, 7, 5.31432, (char*)"FLOAT EXPERIMENT 10000 ELEMENTS", 6).out();
    GetResultExp<float>(10000000, 5, 2, 5.31432, (char*)"FLOAT EXPERIMENT 100000000 ELEMENTS", 6).out();
    GetResultExp<float>(100000000, 5, 2, 5.31432, (char*)"FLOAT EXPERIMENT 1000000000 ELEMENTS", 6).out();

}
