cmake_minimum_required(VERSION 3.12)
project(C++CudaTransformerX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories and link libraries
include_directories(${CMAKE_BINARY_DIR}/include)
link_directories(${CMAKE_BINARY_DIR}/lib)

# Add the main executable
add_executable(C++CudaTransformerX
        src/main.cpp
        src/MixedPrecisionFloat16.cpp
        src/Tensor.tpp
        include/MixedPrecisionFloat16.h
        include/Tensor.h
        include/DenseLayer.h
        include/ActivationFunction.h
        include/Optimizer.h
        include/LossFunction.h
        src/Optimizer.cpp
        src/DenseLayer.cpp
        src/LossFunction.cpp
        src/ActivationFunction.tpp
)

# Add the test executable
add_executable(global_tests tests/test_tensor.cpp
        tests/test_activation_function.cpp)

# Link the test executable with Google Test libraries
target_link_libraries(global_tests gtest gtest_main)
