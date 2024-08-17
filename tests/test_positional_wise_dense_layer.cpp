#include "gtest/gtest.h"
#include "../include/PositionalWiseDenseLayer.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

class PositionalWiseDenseLayerTest : public ::testing::Test {
protected:
    // Set up the test environment
    void SetUp() override {
        d_model = 4;
        d_ff = 8;
        activation_function = new ActivationFunction<float>::ReLU();
        layer = new PositionalWiseDenseLayer<float>(d_model, d_ff, *activation_function);
    }

    // Tear down the test environment
    void TearDown() override {
        delete layer;
        delete activation_function;
    }

    int d_model{};
    int d_ff{};
    ActivationFunction<float>* activation_function{};
    PositionalWiseDenseLayer<float>* layer{};
};

// Test the initialization of weights and biases
TEST_F(PositionalWiseDenseLayerTest, InitializeWeightsAndBiases) {
    const auto params = layer->parameters();

    // Check that weights and biases have correct shapes
    EXPECT_EQ(params[0].get().shape(), std::vector<int>({d_model, d_ff}));
    EXPECT_EQ(params[1].get().shape(), std::vector<int>({d_ff}));
    EXPECT_EQ(params[2].get().shape(), std::vector<int>({d_ff, d_model}));
    EXPECT_EQ(params[3].get().shape(), std::vector<int>({d_model}));

    // Check that weights are initialized with non-zero values
    for (const auto& param : params) {
        const auto& data = param.get().data;
        for (const auto& value : data) {
            // Check that the value is either non-zero or within a specific range
            bool condition = (value != 0) || (value > -1 && value < 1);
            EXPECT_TRUE(condition) << "Value is either zero or outside the range [-1, 1]: " << value;
        }
    }
}

// Test the forward pass
TEST_F(PositionalWiseDenseLayerTest, ForwardPass) {
    const Tensor<float> input({2, d_model}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});

    const Tensor<float> output = layer->forward(input);

    // Check that the output has the correct shape
    EXPECT_EQ(output.shape(), std::vector<int>({2, d_model}));

    // Ensure that the output is not empty and that the activation function has been applied
    const auto& data = output.data;
    for (const auto& value : data) {
        EXPECT_FALSE(std::isnan(value));
    }
}

// Test the backward pass
TEST_F(PositionalWiseDenseLayerTest, BackwardPass) {
    const Tensor<float> input({2, d_model}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    Tensor<float> grad_output = Tensor<float>::uniform({2, d_model});

    // Perform forward pass first
    layer->forward(input);

    // Perform backward pass
    layer->backward(grad_output);

    // Check that the gradients have the correct shape
    const auto grads = layer->gradients();
    ASSERT_EQ(grads.size(), 4);
    EXPECT_EQ(grads[0].get().shape(), std::vector<int>({d_model, d_ff}));
    EXPECT_EQ(grads[1].get().shape(), std::vector<int>({d_ff}));
    EXPECT_EQ(grads[2].get().shape(), std::vector<int>({d_model, d_model}));
    EXPECT_EQ(grads[3].get().shape(), std::vector<int>({d_model}));

    // Ensure that the gradient with respect to the input is not empty
    for (const auto& value : grad_output.data) {
        EXPECT_FALSE(std::isnan(value));
    }
}

// Test that parameters and gradients are accessible and correctly linked
TEST_F(PositionalWiseDenseLayerTest, ParametersAndGradientsAccess) {
    const auto params = layer->parameters();
    const auto grads = layer->gradients();

    ASSERT_EQ(params.size(), 4);
    ASSERT_EQ(grads.size(), 4);

    // Modify the parameters and check if gradients get modified accordingly
    for (int i = 0; i < params.size(); ++i) {
        params[i].get().data[0] = 1.0f;
        grads[i].get().data[0] = 1.0f;

        EXPECT_EQ(params[i].get().data[0], grads[i].get().data[0]);
    }
}

