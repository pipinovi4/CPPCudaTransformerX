#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "../include/DenseLayer.h"

TEST(DenseLayerTest, InitializeWeights) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    for (const auto& weight : dense_layer.weights.data) {
        EXPECT_NE(weight, 0); // Ensure weights are not zero
    }
}

TEST(DenseLayerTest, ForwardPass) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector<int>({1, 64}));
}

TEST(DenseLayerTest, BackwardPass) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    const Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    const Tensor<float> grad_input = dense_layer.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), std::vector<int>({1, 64}));
}

TEST(DenseLayerTest, UpdateParameters) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);
    Optimizer<float>::SGD optimizer(0.01, *new Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule(0.01, 0.95));

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    const Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, 0.0f, 1.0f);
    dense_layer.backward(grad_output);

    dense_layer.updateParameters(&optimizer, 0);

    for (const auto& weight : dense_layer.weights.data) {
        EXPECT_NE(weight, 0);
    }
}

TEST(DenseLayerTest, ForwardPassBoundary) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(1, 1, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 1}, 0.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector<int>({1, 1}));
}

TEST(DenseLayerTest, BackwardPassBoundary) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(1, 1, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 1}, 0.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    const Tensor<float> grad_output = Tensor<float>::uniform({1, 1}, 0.0f, 1.0f);
    const Tensor<float> grad_input = dense_layer.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), std::vector<int>({1, 1}));
}

TEST(DenseLayerTest, ForwardPassEdge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, -1.0f, 1.0f);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector<int>({1, 64}));
}

TEST(DenseLayerTest, BackwardPassEdge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, -1.0f, 1.0f);
    Tensor<float> output_data = dense_layer.forward(input_data);

    const Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, -1.0f, 1.0f);
    const Tensor<float> grad_input = dense_layer.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), std::vector<int>({1, 64}));
}

// Large value tests
TEST(DenseLayerTest, ForwardPassLarge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 1e6, 1e7);
    const Tensor<float> output_data = dense_layer.forward(input_data);

    EXPECT_EQ(output_data.shape(), std::vector<int>({1, 64}));
}

TEST(DenseLayerTest, BackwardPassLarge) {
    ActivationFunction<float>::ReLU activation_function;
    DenseLayer<float> dense_layer(64, 64, &activation_function);

    const Tensor<float> input_data = Tensor<float>::uniform({1, 64}, 1e6, 1e7);
    Tensor<float> output_data = dense_layer.forward(input_data);

    const Tensor<float> grad_output = Tensor<float>::uniform({1, 64}, 1e6, 1e7);
    const Tensor<float> grad_input = dense_layer.backward(grad_output);

    EXPECT_EQ(grad_input.shape(), std::vector<int>({1, 64}));
}