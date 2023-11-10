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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_t_axpy<float>(vec_size, vecA, 2, vecB, 3, 4.53, blocksPerGrid, threadsPerBlock);

        //CPU res in copyB
        taxpy<float>(vec_size, vecA, 2, copyB, 3, 4.53);
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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;

        cuda_t_axpy<float>(vec_size, vecA, 4, vecB, 7, 4.53, blocksPerGrid, threadsPerBlock);

        //CPU res in copyB
        taxpy<float>(vec_size, vecA, 2, copyB, 3, 4.53);
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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_t_axpy<float>(vec_size, vecA, rand()%20+5, vecB, rand()%20+5, 4.53, blocksPerGrid, threadsPerBlock);

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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_t_axpy<double>(vec_size, vecA, 2, vecB, 3, 4.53, blocksPerGrid, threadsPerBlock);

        //CPU res in copyB
        taxpy<double>(vec_size, vecA, 2, copyB, 3, 4.53);
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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_t_axpy<double>(vec_size, vecA, 4, vecB, 7, 4.53, blocksPerGrid, threadsPerBlock);

        //CPU res in copyB
        taxpy<double>(vec_size, vecA, 2, copyB, 3, 4.53);
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
        int threadsPerBlock = 256;
        int blocksPerGrid = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
        cuda_t_axpy<double>(vec_size, vecA, rand()%20+5, vecB, rand()%20+5, 4.53, blocksPerGrid, threadsPerBlock);

        //CPU res in copyB
        taxpy<double>(vec_size, vecA, rand()%20+5, copyB, rand()%20+5, 4.53);

        return is_equal(vec_size, vecB, copyB);
    });
}

int main(){
    //tests
    Run_tests();
    int blocksPerGrid;
    int threadsPerBlock = 256;
    int num_elements;
    //Experiments
    //GetResultExp(int vectorSize, int Xinc, int Yinc, T alpfa, char* descr, int omp_thread_nom)


    num_elements = 10000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 10000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    num_elements = 10000000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 100000000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    num_elements = 100000000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 1000000000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    num_elements = 10000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 3, 7, 5.31432, (char*)"FLOAT EXPERIMENT 10000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    num_elements = 10000000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 3, 7, 5.31432, (char*)"FLOAT EXPERIMENT 100000000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    num_elements = 100000000;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 3, 7, 5.31432, (char*)"FLOAT EXPERIMENT 1000000000 ELEMENTS", 6, blocksPerGrid, threadsPerBlock).out();

    printf("\n\n\t\t\t\x1b[32;47mBLOCKS CHANGE EXPERIMENTS DOUBLE\x1b[0m\n\n");
    //1
    num_elements = 6000000;
    threadsPerBlock = 8;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 8 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();
    
    threadsPerBlock = 16;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 16 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 32;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 32 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 64;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 64 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 128;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 128 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 256;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<double>(num_elements, 7, 3, 5.31432, (char*)"DOUBLE EXPERIMENT 256 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    printf("\n\n\t\t\t\x1b[32;47mBLOCKS CHANGE EXPERIMENTS FLOAT\x1b[0m\n\n");
    //2
    num_elements = 6000000;
    threadsPerBlock = 8;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 8 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();
    
    threadsPerBlock = 16;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 16 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 32;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 32 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 64;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 64 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 128;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 128 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

    threadsPerBlock = 256;
    blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    GetResultExp<float>(num_elements, 7, 3, 5.31432, (char*)"FLOAT EXPERIMENT 256 BLOCKS IN GRID", 6, blocksPerGrid, threadsPerBlock).out();

}
