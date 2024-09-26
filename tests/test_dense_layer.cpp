#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "../include/DenseLayer.h"
#include "../include/Optimizer.h"

TEST(DenseLayerTest, InitializeWeights) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    for (const auto& parameter : dense_layer.parameters()) {
        for (const auto& weight : parameter.get().data) {
            EXPECT_GE(weight, -1.0f);
            EXPECT_LE(weight, 1.0f);
        }
    }
}

TEST(DenseLayerTest, ForwardPass) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({64}, 0.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector({64}));
}

TEST(DenseLayerTest, BackwardPass) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    dense_layer.backward(grad_output);

    EXPECT_EQ(grad_output.shape(), std::vector({1, 64}));
}

TEST(DenseLayerTest, UpdateParameters) {
    // Initialize activation function and dense layer
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    // Create learning rate schedule and optimizer
    auto lr_schedule = std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100);
    Optimizer<float>::SGD optimizer(dense_layer.parameters(), 0.01, *lr_schedule);

    // Create input data and perform forward pass
    Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    // Create gradient output and perform backward pass
    Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    dense_layer.backward(grad_output);

    // Update weights using optimizer
    optimizer.update(dense_layer.parameters(), dense_layer.gradients(), 1);

    // Ensure weights have been updated (not all zero)
    for (const auto& parameter : dense_layer.parameters()) {
        for (const auto& weight : parameter.get().data) {
            EXPECT_GE(weight, -1.0f);
            EXPECT_LE(weight, 1.0f);
        }
    }
}

TEST(DenseLayerTest, ForwardPassBoundary) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(1, 1, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1}, 0.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector({1}));
}

TEST(DenseLayerTest, BackwardPassBoundary) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(1, 1, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    Tensor<float> grad_output = Tensor<float>::uniform({1}, 0.0f, 1.0f);
    dense_layer.backward(grad_output);

    EXPECT_EQ(grad_output.shape(), std::vector({1}));
}

TEST(DenseLayerTest, ForwardPassEdge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({64}, -1.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector({64}));
}

TEST(DenseLayerTest, BackwardPassEdge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({64}, -1.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    Tensor<float> grad_output = Tensor<float>::uniform({64}, -1.0f, 1.0f);
    dense_layer.backward(grad_output);

    EXPECT_EQ(grad_output.shape(), std::vector({64}));
}

// Large value tests
TEST(DenseLayerTest, ForwardPassLarge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({64}, 1e6, 1e7);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector({64}));
}

TEST(DenseLayerTest, BackwardPassLarge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({64}, 1e6, 1e7);
    Tensor<float> output_data = dense_layer.forward(input_data);

    Tensor<float> grad_output = Tensor<float>::uniform({64}, 1e6, 1e7);
    dense_layer.backward(grad_output);

    EXPECT_EQ(grad_output.shape(), std::vector({64}));
}