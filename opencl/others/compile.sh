g++ -o test test.cpp -O3 -DPROJECT_ROOT="/ghome/zlai/workspace/opencl_study/opencl" -DOPENCL_PROJ -DSILENCE -Wno-write-strings \
    -I/ghome/zlai/workspace/opencl_study/opencl/inc/ \
    -I/ghome/zlai/workspace/opencl_study/opencl/common/  \
    -I/ghome/zlai/workspace/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common/inc/   \
    -L/ghome/zlai/workspace/Nvidia_OpenCL_SDK_4_2_Linux/OpenCL/common/lib/ -lOpenCL  \