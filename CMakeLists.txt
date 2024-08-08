cmake_minimum_required(VERSION 3.12)
project(C++CudaTransformerX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories and link libraries
include_directories(${CMAKE_BINARY_DIR}/include ${CUDAToolkit_INCLUDE_DIRS})
link_directories(${CMAKE_BINARY_DIR}/lib ${CUDAToolkit_LIBRARY_DIR})

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
        src/Optimizer.tpp
        src/DenseLayer.tpp
        src/LossFunction.tpp
        src/ActivationFunction.tpp
        include/Embedding.h
        src/Embedding.tpp
        examples/utils/loadMNIST.tpp
        include/Tokenizer.h
        src/Tokenizer.tpp
        examples/utils/loadAGNews.tpp
)

# Link the main executable with CUDA
target_link_libraries(C++CudaTransformerX ${CUDAToolkit_LIBRARIES})

# Add the test executable
add_executable(global_tests
        tests/test_tensor.cpp
        tests/test_activation_function.cpp
        tests/test_loss_function.cpp
        tests/test_optimizer.cpp
        tests/test_dense_layer.cpp
        tests/test_embedding.cpp
)

# Add example of the DigitRecognizer executable
add_executable(digit_recognizer
        examples/run/DigitRecognizerRunner.cpp
        examples/DigitRecognizer.h
        examples/DigitRecognizer.tpp
        examples/utils/readUInt32.h
        examples/utils/readUInt32.cpp
        examples/utils/swapEndian.h
        examples/utils/swapEndian.cpp
        examples/utils/loadMNIST.tpp
        examples/utils/loadAGNews.h
        examples/utils/loadMNIST.h
)

add_executable(embedding_model
        examples/run/EmbeddingModelRunner.cpp
        examples/EmbeddingModel.h
        examples/EmbeddingModel.tpp
        src/Tokenizer.tpp
)

# Link the test executable with CUDA
target_link_libraries(global_tests ${CUDAToolkit_LIBRARIES})

# Link the test executable with Google Test libraries
target_link_libraries(global_tests gtest gtest_main)
