# CMakeLists.txt

# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.12)

# Set the project name
project(C++CudaTransformerX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set the path to Eigen manually
set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen-3.4.0")

# Set the path to the CUDAToolkit manually
link_directories(${CMAKE_BINARY_DIR}/lib ${CUDAToolkit_LIBRARY_DIR})

# Include the Eigen directory
include_directories(${EIGEN3_INCLUDE_DIR})

# Check if Google Test is installed
find_package(GTest REQUIRED)

# Check if the Eigen library is installed
find_package(Eigen3 REQUIRED NO_MODULE)

# Add the main executable
add_executable(C++CudaTransformerX
        src/main.cpp
        src/Float16.tpp
        src/Tensor.tpp
        include/Float16.h
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
        utils/loadMNIST.tpp
        include/Tokenizer.h
        src/Tokenizer.tpp
        utils/loadAGNews.tpp
        include/MultiHeadAttention.h
        src/MultiHeadAttention.tpp
        models/MultiHeadAttentionModel.h
        src/MultiHeadAttention.tpp
        utils/loadWikiText.tpp
        models/Transformer.h
        models/Transformer.tpp
        include/LayerNorm.h
        src/LayerNorm.tpp
        include/ResidualBlock.h
        src/ResidualBlock.tpp
        include/PositionalWiseDenseLayer.h
        src/PositionalWiseDenseLayer.tpp
        include/Layer.h
        src/LayerNorm.tpp
        include/BPE.h
        src/BPE.tpp
        utils/loadAGNews.h
        utils/loadMNIST.h
        utils/loadWikiText.tpp
        utils/swapEndian.tpp
        src/BPE.tpp
        src/BPE.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.tpp
        models/Transformer.tpp
        models/Transformer.tpp
)

# Add the test executable
add_executable(global_tests
        tests/test_tensor.cpp
        tests/test_activation_function.cpp
        tests/test_loss_function.cpp
        tests/test_optimizer.cpp
        tests/test_dense_layer.cpp
        tests/test_embedding.cpp
        tests/test_tokenizer.cpp
        tests/test_multi_head_attention.cpp
        tests/test_layer_norm.cpp
        tests/test_residual_block.cpp
        tests/test_positional_wise_dense_layer.cpp
        tests/test_bpe.cpp
)

# Link the test executable with Google Test libraries
target_link_libraries(global_tests gtest gtest_main)

# Add example of the DigitRecognizer executable
add_executable(digit_recognizer
        examples/DigitRecognizerRunner.cpp
        models/DigitRecognizer.h
        models/DigitRecognizer.tpp
        utils/readUint32.tpp
        utils/swapEndian.tpp
        utils/loadMNIST.tpp
        utils/loadWikiText.h
        utils/readUint32.h
        utils/swapEndian.h
)

# Add example of the EmbeddingModel executable
add_executable(embedding_model
        examples/EmbeddingModelRunner.cpp
        models/EmbeddingModel.h
        models/EmbeddingModel.tpp
        src/Tokenizer.tpp
)

# Add example of the MultiHeadAttentionModel executable
add_executable(multi_head_attention_model
        examples/MultiHeadAttentionModelRunner.cpp
        models/MultiHeadAttentionModel.h
        models/MultiHeadAttentionModel.tpp
)

# Add example of the train LLM executable
add_executable(train
        src/train.cpp
        src/Tensor.tpp
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
        utils/loadMNIST.tpp
        include/Tokenizer.h
        src/Tokenizer.tpp
        utils/loadAGNews.tpp
        include/MultiHeadAttention.h
        src/MultiHeadAttention.tpp
        models/MultiHeadAttentionModel.h
        src/MultiHeadAttention.tpp
        utils/loadWikiText.tpp
        models/Transformer.h
        models/Transformer.tpp
        include/LayerNorm.h
        src/LayerNorm.tpp
        include/ResidualBlock.h
        src/ResidualBlock.tpp
        include/PositionalWiseDenseLayer.h
        src/PositionalWiseDenseLayer.tpp
        include/Layer.h
        src/LayerNorm.tpp
        include/BPE.h
        src/BPE.tpp
        utils/loadAGNews.h
        utils/loadMNIST.h
        utils/loadWikiText.tpp
        utils/swapEndian.tpp
        src/BPE.tpp
        src/BPE.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.tpp
        models/Transformer.tpp
        models/Transformer.tpp
)

# Add example of the generate LLM executable
add_executable(generate
        src/generate.cpp
        src/Tensor.tpp
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
        utils/loadMNIST.tpp
        include/Tokenizer.h
        src/Tokenizer.tpp
        utils/loadAGNews.tpp
        include/MultiHeadAttention.h
        src/MultiHeadAttention.tpp
        models/MultiHeadAttentionModel.h
        src/MultiHeadAttention.tpp
        utils/loadWikiText.tpp
        models/Transformer.h
        models/Transformer.tpp
        include/LayerNorm.h
        src/LayerNorm.tpp
        include/ResidualBlock.h
        src/ResidualBlock.tpp
        include/PositionalWiseDenseLayer.h
        src/PositionalWiseDenseLayer.tpp
        include/Layer.h
        src/LayerNorm.tpp
        include/BPE.h
        src/BPE.tpp
        utils/loadAGNews.h
        utils/loadMNIST.h
        utils/loadWikiText.tpp
        utils/swapEndian.tpp
        src/BPE.tpp
        src/BPE.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.h
        utils/loadVocab.tpp
        utils/loadVocab.tpp
        models/Transformer.tpp
        models/Transformer.tpp
)
