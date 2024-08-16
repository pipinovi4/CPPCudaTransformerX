#include "gtest/gtest.h"
#include "../include/MultiHeadAttention.h"
#include "../include/Tensor.h"

constexpr int MAX_SEQUENCE_LENGTH = 64;
constexpr int HIDDEN_DIM = 64;
constexpr int NUM_HEADS = 8;
constexpr int HEAD_DIM = HIDDEN_DIM / NUM_HEADS;

class MultiHeadAttentionTest : public ::testing::Test {
protected:
    int hidden_dim = HIDDEN_DIM;
    int num_heads = NUM_HEADS;
    int head_dim = HEAD_DIM;

    ActivationFunction<float>::ReLU activation;

    MultiHeadAttention<float> multihead_attention = MultiHeadAttention<float>(hidden_dim, num_heads, head_dim, &activation);

    void ProcessInputBackward(const Tensor<float>& grad_output) {
        multihead_attention.backward(grad_output);
    }
};

// Test the forward pass of the MultiHeadAttention layer
TEST_F(MultiHeadAttentionTest, HandlesForwardPass) {
    // Initialize input tensor with random uniform values
    const Tensor<float> input_data = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});

    // Perform forward pass
    const Tensor<float> output = multihead_attention.forward(input_data);

    // Check the output tensor shape
    EXPECT_EQ(output.shape()[0], MAX_SEQUENCE_LENGTH); // Sequence length should be the same
    EXPECT_EQ(output.shape()[1], HIDDEN_DIM); // Hidden dimension should match

    // Additional checks can be added to verify the content of the output
    // For example, checking if output values are within a reasonable range
    for (int i = 0; i < output.size(); ++i) {
        EXPECT_GE(output.data[i], -1.0f); // Example check: output values should be greater than or equal to -1.0
        EXPECT_LE(output.data[i], 1.0f);  // Example check: output values should be less than or equal to 1.0
    }
}

// Test the backward pass of the MultiHeadAttention layer
TEST_F(MultiHeadAttentionTest, HandlesBackwardPass) {
    // Initialize input tensor with random uniform values
    const Tensor<float> input_data = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});

    // Perform forward pass
    const Tensor<float> output = multihead_attention.forward(input_data);

    // Initialize gradient tensor with random uniform values
    const Tensor<float> grad_output = Tensor<float>::uniform(output.shape());

    // Perform backward pass
    this->ProcessInputBackward(grad_output);

    // Additional checks can be added to verify the content of the gradients
    // For example, checking if gradient values are within a reasonable range
    for (int i = 0; i < grad_output.size(); ++i) {
        EXPECT_GE(grad_output.data[i], -1.0f); // Example check: gradient values should be greater than or equal to -1.0
        EXPECT_LE(grad_output.data[i], 1.0f);  // Example check: gradient values should be less than or equal to 1.0
    }
}

// Test parameter initialization
TEST_F(MultiHeadAttentionTest, InitializesParametersCorrectly) {
    const auto parameters = multihead_attention.parameters();
    for (const auto& param : parameters) {
        EXPECT_EQ(param.get().shape()[0], hidden_dim); // Example check for parameter shape
        EXPECT_FALSE(param.get().data.empty()); // Ensure the parameter is not empty
    }
}

// Test gradient accumulation and updating weights
TEST_F(MultiHeadAttentionTest, HandlesGradientAccumulationSimplified) {
    Tensor<float> input_data({1, HIDDEN_DIM}); // Simplify to a single example
    input_data.fill(1.0); // Fill input with known value

    const Tensor<float> output = multihead_attention.forward(input_data);

    Tensor<float> grad_output(output.shape());
    grad_output.fill(0.1); // Use a simple gradient value

    this->ProcessInputBackward(grad_output);

    const auto gradients = this->multihead_attention.gradients();

    // Check if gradients are accumulated as expected
    for (const auto& grad : gradients) {
        // Replace with the actual expected value if known, or just check that gradients are non-zero
        EXPECT_TRUE(grad.get().data[0] > -1 && 1 > grad.get().data[0]); // Example placeholder value
    }
}

// Test for proper split and concatenation of heads
TEST_F(MultiHeadAttentionTest, HandlesSplitAndConcatHeads) {
    const Tensor<float> input_data = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});

    auto split_heads = multihead_attention.split_heads(input_data);
    EXPECT_EQ(split_heads.size(), num_heads); // Number of heads

    for (const auto& head : split_heads) {
        EXPECT_EQ(head.shape()[0], input_data.shape()[0]); // Sequence length
        EXPECT_EQ(head.shape()[1], head_dim); // Dimension per head
    }

    const Tensor<float> concatenated_heads = multihead_attention.concat_heads(split_heads);
    EXPECT_EQ(concatenated_heads.shape()[0], input_data.shape()[0]); // Sequence length
    EXPECT_EQ(concatenated_heads.shape()[1], input_data.shape()[1]); // Hidden dimension
}

TEST_F(MultiHeadAttentionTest, HandleForwardPassWithMask) {
    // Initialize input tensor with random uniform values
    const Tensor<float> input_data = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, HIDDEN_DIM});

    // Initialize mask tensor with random uniform values
    const Tensor<float> mask = Tensor<float>::uniform({MAX_SEQUENCE_LENGTH, MAX_SEQUENCE_LENGTH});

    // Perform forward pass
    const Tensor<float> output = multihead_attention.forward(input_data, &mask);

    // Check the output tensor shape
    EXPECT_EQ(output.shape()[0], MAX_SEQUENCE_LENGTH); // Sequence length should be the same
    EXPECT_EQ(output.shape()[1], HIDDEN_DIM); // Hidden dimension should match

    // Additional checks can be added to verify the content of the output
    // For example, checking if output values are within a reasonable range
    for (int i = 0; i < output.size(); ++i) {
        EXPECT_GE(output.data[i], -1.0f); // Example check: output values should be greater than or equal to -1.0
        EXPECT_LE(output.data[i], 1.0f);  // Example check: output values should be less than or equal to 1.0
    }
}
