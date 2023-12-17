CUDA_BIN_PATH="/usr/local/cuda-12.3/bin"


echo "PATH TO CUDA BIN $CUDA_BIN_PATH"

echo "COMPILATION START:"

rm -rf $PWD/bin
mkdir $PWD/bin

echo $CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_tests_and_time.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/tests_and_time
$CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_tests_and_time.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/tests_and_time

echo $CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_toprofile_only_obvious.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/toprofile_only_obvious
$CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_toprofile_only_obvious.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/toprofile_only_obvious

echo $CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_toprofile_only_shared.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/toprofile_only_shared
$CUDA_BIN_PATH/nvcc -Xcompiler=-fopenmp $PWD/main/main_toprofile_only_shared.cpp $PWD/src/cuda_matmul.cu -o $PWD/bin/toprofile_only_shared

echo ""
echo "COMPILATION END"

echo ""
echo "EXECUTION TESTS AND EXPERIMENTS:"
echo $PWD/bin/tests_and_time
$PWD/bin/tests_and_time

echo ""
echo "SHARED PROFILING:"
echo $CUDA_BIN_PATH/nvprof $PWD/bin/toprofile_only_shared
$CUDA_BIN_PATH/nvprof $PWD/bin/toprofile_only_shared

echo ""
echo "OBVIOUS PROFILING:"
echo $CUDA_BIN_PATH/nvprof $PWD/bin/toprofile_only_obvious
$CUDA_BIN_PATH/nvprof $PWD/bin/toprofile_only_obvious