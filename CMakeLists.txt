cmake_minimum_required(VERSION 3.27)
project(C++CudaTransformerX)

set(CMAKE_CXX_STANDARD 23)

add_executable(C++CudaTransformerX ../CPPCudaTransformerX/src/main.cpp
        ../CPPCudaTransformerX/include/MixedPrecisionFloat16.h
        ../CPPCudaTransformerX/src/MixedPrecisionFloat16.cpp
        ../CPPCudaTransformerX/include/Tensor.h
        ../CPPCudaTransformerX/src/Tensor.cpp
)
