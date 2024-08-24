#include "gtest/gtest.h"
#include "../include/PositionalWiseDenseLayer.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

class PositionalWiseDenseLayerTest : public ::testing::Test {
protected:
    int d_model{};
    int d_ff{};
    int max_token_size{};
    ActivationFunction<float>* activation_function{};
    PositionalWiseDenseLayer<float>* layer{};

    // Set up the test environment
    void SetUp() override {
        d_model = 4;
        d_ff = 3;
        max_token_size = 3;  // Define max_token_size here
        activation_function = new ActivationFunction<float>::ReLU();
        layer = new PositionalWiseDenseLayer<float>(d_model, d_ff, *activation_function);
    }

    // Tear down the test environment
    void TearDown() override {
        delete layer;
        delete activation_function;
    }
};

// Test the forward pass with batch_size and max_token_size
TEST_F(PositionalWiseDenseLayerTest, ForwardPass) {
    // Define input tensor with batch_size and max_token_size
    const Tensor<float> input({max_token_size, d_model},
                        std::vector<float>{
                            1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0, 12.0
                        });  // 24 elements in total

    const Tensor<float> output = layer->forward(input);

    output.print();

    // Check that the output has the correct shape
    EXPECT_EQ(output.shape(), std::vector<int>({max_token_size, d_model}));

    // Ensure that the output is not empty and that the activation function has been applied
    const auto& data = output.data;
    for (const auto& value : data) {
        EXPECT_FALSE(std::isnan(value));
    }
}

// Test the backward pass with batch_size and max_token_size
TEST_F(PositionalWiseDenseLayerTest, BackwardPass) {
    // Define input tensor for forward pass with batch_size and max_token_size
    const Tensor<float> input({max_token_size, d_model},
                        std::vector<float>{
                            1.0, 2.0, 3.0, 4.0,
                            5.0, 6.0, 7.0, 8.0,
                            9.0, 10.0, 11.0, 12.0
                        });  // 24 elements in total
    Tensor<float> grad_output = Tensor<float>::uniform({max_token_size, d_model});

    // Perform forward pass first
    layer->forward(input);

    // Perform backward pass
    layer->backward(grad_output);

    // Check that the gradients have the correct shape
    const auto grads = layer->gradients();
    ASSERT_EQ(grads.size(), 4);
    EXPECT_EQ(grads[0].get().shape(), std::vector<int>({d_model, d_ff}));
    EXPECT_EQ(grads[1].get().shape(), std::vector<int>({d_ff}));
    EXPECT_EQ(grads[2].get().shape(), std::vector<int>({d_ff, d_model}));
    EXPECT_EQ(grads[3].get().shape(), std::vector<int>({d_model}));

    // Ensure that the gradient with respect to the input is not empty
    for (const auto& value : grad_output.data) {
        EXPECT_FALSE(std::isnan(value));
    }
}
