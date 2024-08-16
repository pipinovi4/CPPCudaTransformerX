#include "gtest/gtest.h"
#include "../include/ResidualBlock.h"
#include "../include/Tensor.h"

class TestResidualBlock : public ::testing::Test {
protected:
    const int d_model;
    ResidualBlock<float> residual_block;
    std::default_random_engine generator;

    TestResidualBlock()
        : d_model(512),
          residual_block(ResidualBlock<float>(d_model))
    {}

    void SetUp() override {
        std::random_device rd;
        generator.seed(rd());
    }
};

TEST_F(TestResidualBlock, ForwardShape) {
    // Initialize input and processed tensors
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    Tensor<float> processed(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
        processed.data[i] = distribution(generator);
    }

    // Forward pass
    const Tensor<float> y = residual_block.forward(x, processed);

    // Check the dimensions of the output tensor
    EXPECT_EQ(y.shape(), dims);
}

TEST_F(TestResidualBlock, ForwardOutputRange) {
    // Initialize input and processed tensors
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    Tensor<float> processed(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
        processed.data[i] = distribution(generator);
    }

    // Forward pass
    const Tensor<float> y = residual_block.forward(x, processed);

    // Check that output values are within a reasonable range
    constexpr float min_val = -10.0;
    constexpr float max_val = 10.0;
    for (int i = 0; i < y.size(); ++i) {
        EXPECT_GE(y.data[i], min_val);
        EXPECT_LE(y.data[i], max_val);
    }
}

TEST_F(TestResidualBlock, BackwardShape) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    Tensor<float> processed(dims);
    Tensor<float> dout(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
        processed.data[i] = distribution(generator);
        dout.data[i] = distribution(generator);
    }

    // Forward pass
    residual_block.forward(x, processed);

    // Backward pass
    const Tensor<float> grad_output = residual_block.backward(dout);

    // Check the dimensions of the output tensor
    EXPECT_EQ(grad_output.shape(), dims);
}

TEST_F(TestResidualBlock, BackwardOutputRange) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    Tensor<float> processed(dims);
    Tensor<float> dout(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
        processed.data[i] = distribution(generator);
        dout.data[i] = distribution(generator);
    }

    // Forward pass
    residual_block.forward(x, processed);

    // Backward pass
    const Tensor<float> grad_output = residual_block.backward(dout);

    // Check that output values are within a reasonable range
    constexpr float min_val = -10.0;
    constexpr float max_val = 10.0;
    for (int i = 0; i < grad_output.size(); ++i) {
        EXPECT_GE(grad_output.data[i], min_val);
        EXPECT_LE(grad_output.data[i], max_val);
    }
}

TEST_F(TestResidualBlock, BackwardInputGradient) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    Tensor<float> processed(dims);
    Tensor<float> dout(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
        processed.data[i] = distribution(generator);
        dout.data[i] = distribution(generator);
    }

    // Forward pass
    residual_block.forward(x, processed);

    // Backward pass
    const Tensor<float> grad_output = residual_block.backward(dout);

    // Check the gradient of the input tensor
    const Tensor<float>& dx = grad_output;
    for (int i = 0; i < x.size(); ++i) {
        EXPECT_EQ(dx.data[i], grad_output.data[i]);
    }
}