#include <gtest/gtest.h>
#include <random>
#include "../include/ResidualBlock.h"
#include "../include/MultiHeadAttention.h"
#include "../include/ActivationFunction.h"

class TestResidualBlock : public ::testing::Test {
protected:
    const int d_model;
    const int batch_size;
    const int num_heads;
    const int head_dim;
    const int max_token_size;
    MultiHeadAttention<float> mha;
    ResidualBlock<float, MultiHeadAttention<float>> residual_block;
    std::default_random_engine generator;

    // Constructor for TestResidualBlock
    TestResidualBlock()
        : d_model(64),  // Reduced dimension size
          batch_size(8),  // Set batch size
          num_heads(4),  // Reduced number of heads
          head_dim(d_model / num_heads),
          max_token_size(32),  // Set max token size
          mha(d_model, num_heads, head_dim),  // Initialize MultiHeadAttention with reduced parameters
          residual_block(d_model, 1e-6, mha) {}  // Initialize ResidualBlock

    // SetUp method for initializing the random generator
    void SetUp() override {
        std::random_device rd;
        generator.seed(rd());
    }
};

// Example test case for ResidualBlock forward pass with batch size and max token size
TEST_F(TestResidualBlock, ForwardPassWithBatchAndTokenSize) {
    // Define input tensor with batch size and max token size
    Tensor<float> input({batch_size, max_token_size, d_model});
    input.fill(1.0);  // Fill the tensor with ones

    // Define a mask tensor with batch size and max token size
    Tensor<float> mask({batch_size, max_token_size, max_token_size});
    mask.fill(0.0);  // Fill the mask tensor with zeros

    // Run forward pass through the ResidualBlock
    const Tensor<float> output = residual_block.forward(input, &mask);

    // Check output dimensions
    ASSERT_EQ(output.shape()[0], input.shape()[0]);  // Batch size should be the same
    ASSERT_EQ(output.shape()[1], input.shape()[1]);  // Max token size should match
    ASSERT_EQ(output.shape()[2], input.shape()[2]);  // Hidden dimension should match

    // Add more assertions here if necessary
}

// Example test case for backward pass with batch size and max token size
TEST_F(TestResidualBlock, BackwardPassWithBatchAndTokenSize) {
    // Define input tensor for forward pass with batch size and max token size
    Tensor<float> input({batch_size, max_token_size, d_model});
    input.fill(1.0);  // Fill the tensor with ones

    // Define a mask tensor for the forward pass with batch size and max token size
    Tensor<float> mask({batch_size, max_token_size, max_token_size});
    mask.fill(0.0);  // Fill the mask with zeros

    // Run forward pass
    Tensor<float> output = residual_block.forward(input, &mask);

    // Define output gradient tensor for backward pass (same shape as output)
    Tensor<float> dout({batch_size, max_token_size, d_model});
    dout.fill(1.0);  // Fill the tensor with ones as gradients from the next layer

    // Run backward pass
    residual_block.backward(dout);

    // Retrieve gradients from the process layer (MultiHeadAttention in this case)
    const auto gradients = residual_block.process_layer_.gradients();

    // Check that gradients are non-zero and within a reasonable range
    for (const auto& grad : gradients) {
        for (const auto& value : grad.get().data) {
            EXPECT_NE(value, 0.0f);  // Ensure that gradients are not zero
        }
    }
}
