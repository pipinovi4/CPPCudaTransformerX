cmake_minimum_required(VERSION 3.15)

project(
    BackpropLab
    VERSION 0.1.0
    DESCRIPTION "Experimental C++ neural-network framework and Transformer implementation"
    LANGUAGES CXX
)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BACKPROP_LAB_BUILD_TESTS "Build the GoogleTest test suite" ON)
option(BACKPROP_LAB_BUILD_EXAMPLES "Build standalone model examples" ON)
option(BACKPROP_LAB_ENABLE_OPENMP "Enable OpenMP when available" ON)

# Eigen is supplied by Conan or by a compatible system package.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

if(BACKPROP_LAB_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
endif()

# The framework is template-based, so consumers only need its headers and
# implementation files on the include path.
add_library(backprop_lab INTERFACE)
target_include_directories(backprop_lab INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils"
)
target_link_libraries(backprop_lab INTERFACE Eigen3::Eigen)
target_compile_features(backprop_lab INTERFACE cxx_std_17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(backprop_lab INTERFACE OpenMP::OpenMP_CXX)
endif()

function(backprop_lab_add_executable target source)
    add_executable("${target}" "${source}")
    target_link_libraries("${target}" PRIVATE backprop_lab)
endfunction()

backprop_lab_add_executable(backprop_lab_cli src/main.cpp)
backprop_lab_add_executable(train src/train.cpp)
backprop_lab_add_executable(generate src/generate.cpp)

if(BACKPROP_LAB_BUILD_EXAMPLES)
    backprop_lab_add_executable(digit_recognizer examples/DigitRecognizerRunner.cpp)
    backprop_lab_add_executable(embedding_model examples/EmbeddingModelRunner.cpp)
    backprop_lab_add_executable(
        multi_head_attention_model
        examples/MultiHeadAttentionModelRunner.cpp
    )
endif()

if(BUILD_TESTING AND BACKPROP_LAB_BUILD_TESTS)
    find_package(GTest REQUIRED)
    include(GoogleTest)

    add_executable(global_tests
        tests/test_activation_function.cpp
        tests/test_bpe.cpp
        tests/test_dense_layer.cpp
        tests/test_embedding.cpp
        tests/test_layer_norm.cpp
        tests/test_loss_function.cpp
        tests/test_multi_head_attention.cpp
        tests/test_optimizer.cpp
        tests/test_positional_wise_dense_layer.cpp
        tests/test_residual_block.cpp
        tests/test_tensor.cpp
        tests/test_tokenizer.cpp
    )
    target_link_libraries(global_tests PRIVATE
        backprop_lab
        GTest::gtest
        GTest::gtest_main
    )

    gtest_discover_tests(global_tests)
endif()
