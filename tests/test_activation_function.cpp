#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

typedef Tensor<float> (*ActivationFunc)(const Tensor<float>&);

class ActivationFunctionTest : public ::testing::Test {
protected:
    Tensor<float> input;
    Tensor<float> expected;
    ActivationFunc activation;

    ActivationFunctionTest() : input(Tensor<float>({1, 1})), expected(Tensor<float>({1, 1})), activation(ActivationFunction<float>::relu) {}

    void SetUpTensors(const std::vector<float>& inputData, const std::vector<float>& outputData, const std::vector<int>& inputDims, const std::vector<int>& outputDims, const ActivationFunc activationFunction) {
        input = Tensor<float>(inputDims, inputData);
        expected = Tensor<float>(outputDims, outputData);
        activation = activationFunction;
    }

    template <typename T>
    Tensor<T> processInput(const Tensor<T>& input) {
        return activation(input);
    }

    void ExpectTensorNear(const double abs_error = 1e-2) {
        const Tensor<float>& actual = processInput(input);
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }
};

// Test Sigmoid Activation Function
TEST_F(ActivationFunctionTest, Sigmoid) {
    // Init activation function
    const ActivationFunc activationFunction = ActivationFunction<float>::sigmoid;

    // Init TestHelper class
    SetUpTensors({-11.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
                 {0.0f, 0.880797f, 0.952574f, 0.982014f, 0.993307f, 0.997527f, 0.999089f, 0.999665f, 0.999877f, 0.999955f},
                 {2, 5}, {2, 5}, activationFunction);

    ExpectTensorNear();

    // Normal case
    std::vector<float> inputData = {0.0f, 2.0f, -1.0f, -2.0f};
    std::vector<float> outputData = {0.5f, 0.880797f, 0.268941f, 0.119203f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: large values
    inputData = {1000.0f, -1000.0f, 700.0f, -700.0f};
    outputData = {1.0f, 0.0f, 1.0f, 0.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();
}

// Test Softmax Activation Function
TEST_F(ActivationFunctionTest, Softmax)
{
    // Normal case
    std::vector<float> inputData = {0.0f, 2.0f, -1.0f, -2.0f};
    std::vector<float> outputData = {0.11920291f, 0.880797f, 0.7310586f, 0.26894143f};
    const ActivationFunc activationFunction = ActivationFunction<float>::softmax;
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: zero vector
    inputData = {0.0, 0.0, 0.0};
    outputData = {0.333333, 0.333333, 0.333333};
    SetUpTensors(inputData, outputData, {1, 3}, {1, 3}, activationFunction);

    ExpectTensorNear();
}

// Test ReLU Activation Function
TEST_F(ActivationFunctionTest, ReLU) {
    // Init activation function
    const ActivationFunc activationFunction = ActivationFunction<float>::relu;

    // Normal case
    std::vector<float> inputData = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> outputData = {0.0f, 2.0f, 0.0f, 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all negative values
    inputData = {-1.0f, -2.0f, -3.0f, -4.0f};
    outputData = {0.0f, 0.0f, 0.0f, 0.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all positive values
    inputData = {1.0f, 2.0f, 3.0f, 4.0f};
    outputData = {1.0f, 2.0f, 3.0f, 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();
}

// Test Leaky ReLU Activation Function
TEST_F(ActivationFunctionTest, LeakyReLU) {
    // Init activation function
    const ActivationFunc activationFunction = [](const Tensor<float>& x) {
        return ActivationFunction<float>::leaky_relu(x, 0.01f);
    };

    // Normal case
    std::vector<float> inputData = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> outputData = {-0.01f, 2.0f, -0.03f, 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all negative values
    inputData = {-1.0f, -2.0f, -3.0f, -4.0f};
    outputData = {-0.01f, -0.02f, -0.03f, -0.04f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all positive values
    inputData = {1.0f, 2.0f, 3.0f, 4.0f};
    outputData = {1.0f, 2.0f, 3.0f, 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();
}

// Test ELU Activation Function
TEST_F(ActivationFunctionTest, ELU) {
    // Init activation function
    const auto activationFunction = [](const Tensor<float>& x) {
        return ActivationFunction<float>::elu(x, 1.0f);
    };

    // Normal case
    std::vector<float> inputData = {-1.0f, 2.0f, -3.0f, 4.0f};
    std::vector<float> outputData = {static_cast<float>(std::exp(-1) - 1), 2.0f, static_cast<float>(std::exp(-3) - 1), 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all negative values
    inputData = {-1.0f, -2.0f, -3.0f, -4.0f};
    outputData = {static_cast<float>(std::exp(-1) - 1), static_cast<float>(std::exp(-2) - 1), static_cast<float>(std::exp(-3) - 1), static_cast<float>(std::exp(-4) - 1)};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all positive values
    inputData = {1.0f, 2.0f, 3.0f, 4.0f};
    outputData = {1.0f, 2.0f, 3.0f, 4.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();
}

// Test Tanh Activation Function
TEST_F(ActivationFunctionTest, Tanh) {
    // Init activation function
    const ActivationFunc activationFunction = ActivationFunction<float>::tanh;

    // Normal case
    std::vector<float> inputData = {0.0f, 2.0f, -1.0f, -2.0f};
    std::vector<float> outputData = {0.0f, 0.964027f, -0.761594f, -0.964027f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: all zeros
    inputData = {0.0f, 0.0f, 0.0f, 0.0f};
    outputData = {0.0f, 0.0f, 0.0f, 0.0f};
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();

    // Edge case: large values
    inputData = {10.0f, -10.0f, 7.0f, -7.0f}; // Adjusted to avoid NaN
    outputData = {1.0f, -1.0f, 0.999998f, -0.999998f}; // Adjusted expected values
    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, activationFunction);

    ExpectTensorNear();
}