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
        src/Tensor.cpp
        include/MixedPrecisionFloat16.h
        include/Tensor.h
)

# Add the test executable
add_executable(test_tensor tests/test_tensor.cpp)

# Link the test executable with Google Test libraries
target_link_libraries(test_tensor gtest gtest_main)
