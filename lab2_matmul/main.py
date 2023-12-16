import os
import shutil
import re
cur_dir = os.path.abspath(os.curdir)

nvcc = '/usr/local/cuda-12.3/bin/nvcc'
main_fold_path = os.path.join(cur_dir, 'main/')
cuda_kernel_path = os.path.join(cur_dir, 'src/cuda_matmul.cu')
bin_path = os.path.join(cur_dir, 'bin')
profiler = '/usr/local/cuda-12.3/bin/nvprof'

bin_to_profile = []
bin_to_execute = []


def compile():
    print('COMPILATION')
    if os.path.exists(bin_path):
        shutil.rmtree(bin_path)
        os.makedirs(bin_path)
    else:
        os.makedirs(bin_path)

    for main_file in os.listdir(main_fold_path):
        file_path = os.path.join(main_fold_path, main_file)
        bin_name = os.path.join(bin_path, main_file[:-4])
        command = f"{nvcc} -Xcompiler=-fopenmp -lineinfo {file_path} {cuda_kernel_path} -o {bin_name}"

        if re.findall('toprofile', bin_name):
            bin_to_profile.append(bin_name)
        else:
            bin_to_execute.append(bin_name)

        print(command, "\n")
        os.system(command)


def execute_without_profile():
    print('EXECUTE:')
    for file in bin_to_execute:
        os.system(file)


def profile_files():
    print('EXECUTE:')
    for file in bin_to_profile:
        os.system(
            f"{profiler} {file}")


if __name__ == "__main__":
    compile()
    execute_without_profile()
    profile_files()
