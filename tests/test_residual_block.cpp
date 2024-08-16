#include <gtest/gtest.h>
#include <random>
#include "../include/ResidualBlock.h"
#include "../include/MultiHeadAttention.h"
#include "../include/ActivationFunction.h"

class TestResidualBlock : public ::testing::Test {
protected:
    const int d_model;
    MultiHeadAttention<float> mha;
    ResidualBlock<float, MultiHeadAttention<float>> residual_block;
    std::default_random_engine generator;

    // Constructor for TestResidualBlock
    TestResidualBlock()
        : d_model(512),
          mha(d_model, 8, 64),  // Use proper object initialization
          residual_block(d_model, 1e-6, mha) {}

    // SetUp method for initializing the random generator
    void SetUp() override {
        std::random_device rd;
        generator.seed(rd());
    }
};

// Example test case for ResidualBlock forward pass
TEST_F(TestResidualBlock, ForwardPass) {
    // Define input tensor
    Tensor<float> input({1, d_model});
    input.fill(1.0);  // Fill the tensor with ones

    // Define a mask tensor
    Tensor<float> mask({1, 1});
    mask.fill(0.0);  // Fill the mask tensor with zeros

    // Run forward pass through the ResidualBlock
    Tensor<float> output = residual_block.forward(input, &mask);

    // Check output dimensions
    ASSERT_EQ(output.shape()[0], input.shape()[0]);
    ASSERT_EQ(output.shape()[1], input.shape()[1]);

    // Add more assertions here if necessary
}

// Example test case for backward pass
TEST_F(TestResidualBlock, BackwardPass) {
    // Define input tensor for forward pass
    Tensor<float> input({64, d_model});
    input.fill(1.0);  // Fill the tensor with ones

    // Define a mask tensor for the forward pass
    Tensor<float> mask({64, 64});
    mask.fill(0.0);  // Fill the mask with zeros

    // Run forward pass
    Tensor<float> output = residual_block.forward(input, &mask);

    // Define output gradient tensor for backward pass (same shape as output)
    Tensor<float> dout({64, d_model});
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

