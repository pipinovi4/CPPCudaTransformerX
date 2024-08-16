// TestLayerNorm.h
#ifndef TESTLAYERNORM_H
#define TESTLAYERNORM_H

#include <gtest/gtest.h>
#include "../include/LayerNorm.h"
#include "../include/Tensor.h"
#include <random>

// Test class for LayerNorm
class TestLayerNorm : public ::testing::Test {
protected:
    TestLayerNorm() : layer_norm(LayerNorm<float>(d_model)), d_model(512) {}

    // LayerNorm instance to be tested
    LayerNorm<float> layer_norm;
    const int d_model;  // Example dimension size for testing

    // Random number generator with a non-predictable seed
    std::default_random_engine generator;

    void SetUp() override {
        // Seed the random number generator with the current time or random device
        std::random_device rd;
        generator = std::default_random_engine(rd());
    }
};

TEST_F(TestLayerNorm, ForwardShape) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    Tensor<float> y = layer_norm.forward(x);

    // Check the dimensions of the output tensor
    EXPECT_EQ(y.shape(), dims);
}

TEST_F(TestLayerNorm, ForwardOutputRange) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    const Tensor<float> y = layer_norm.forward(x);

    // Check that output values are within a reasonable range
    constexpr float min_val = -10.0;
    constexpr float max_val = 10.0;
    for (int i = 0; i < y.size(); ++i) {
        EXPECT_GE(y.data[i], min_val);
        EXPECT_LE(y.data[i], max_val);
        EXPECT_FALSE(std::isnan(y.data[i]));  // Ensure no NaN values
    }
}

TEST_F(TestLayerNorm, ForwardMeanOutput) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    Tensor<float> y = layer_norm.forward(x);

    // Check that the mean of the output is close to 0 for each feature
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            float sum = 0.0;
            for (int k = 0; k < dims[2]; ++k) {
                sum += y({i, j, k});
            }
            const float mean = sum / static_cast<float>(dims[2]);
            EXPECT_NEAR(mean, 0.0, 0.2);
        }
    }
}

TEST_F(TestLayerNorm, BackwardShape) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    const Tensor<float> y = layer_norm.forward(x);

    // Backward pass
    const Tensor<float> dx = layer_norm.backward(y);

    // Check the dimensions of the gradient tensor
    EXPECT_EQ(dx.shape(), dims);
}

TEST_F(TestLayerNorm, BackwardGradientRange) {
    // Initialize input tensor
    std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    Tensor<float> y = layer_norm.forward(x);

    // Create random gradient output for backward pass
    Tensor<float> dout(dims);
    for (int i = 0; i < dout.size(); ++i) {
        dout.data[i] = distribution(generator);
    }

    // Backward pass
    Tensor<float> dx = layer_norm.backward(dout);

    // Check that the gradients are within a reasonable range
    constexpr float min_grad = -10.0;
    constexpr float max_grad = 10.0;
    for (int i = 0; i < dx.size(); ++i) {
        EXPECT_GE(dx.data[i], min_grad);
        EXPECT_LE(dx.data[i], max_grad);
        EXPECT_FALSE(std::isnan(dx.data[i]));  // Ensure no NaN values
    }
}

TEST_F(TestLayerNorm, BackwardGradientMean) {
    // Initialize input tensor
    const std::vector<int> dims = {3, 4, d_model};
    Tensor<float> x(dims);
    std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < x.size(); ++i) {
        x.data[i] = distribution(generator);
    }

    // Forward pass
    Tensor<float> y = layer_norm.forward(x);

    // Create random gradient output for backward pass
    Tensor<float> dout(dims);
    for (int i = 0; i < dout.size(); ++i) {
        dout.data[i] = distribution(generator);
    }

    // Backward pass
    const Tensor<float> dx = layer_norm.backward(dout);

    // Check that the mean of the gradients is close to 0
    float sum_dx = 0.0;
    for (int i = 0; i < dx.size(); ++i) {
        sum_dx += dx.data[i];
    }
    const float mean_dx = sum_dx / static_cast<float>(dx.size());
    EXPECT_NEAR(mean_dx, 0.0, 0.01);
}

#endif // TESTLAYERNORM_H
