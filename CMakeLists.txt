cmake_minimum_required(VERSION 3.12)
project(C++CudaTransformerX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

link_directories(${CMAKE_BINARY_DIR}/lib ${CUDAToolkit_LIBRARY_DIR})

find_package(GTest REQUIRED)

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


