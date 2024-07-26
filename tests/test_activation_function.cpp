#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

typedef Tensor<float> (*ActivationFunc)(const Tensor<float>&);
typedef Tensor<float> (*ActivationFuncWithAlpha)(const Tensor<float>&, float);

template <typename T>
class ActivationFunctionTest : public ::testing::Test {
protected:
    Tensor<T> input;
    Tensor<T> expected;
    ActivationFunc activation;
    ActivationFuncWithAlpha activationWithAlpha{};
    float alpha;

    ActivationFunctionTest() : input(Tensor<float>({1, 1})), expected(Tensor<float>({1, 1})), activation(ActivationFunction<float>::relu), alpha(0.01f) {}

    void SetUpTensors(const std::vector<float>& inputData, const std::vector<float>& outputData, const std::vector<int>& inputDims, const std::vector<int>& outputDims, const ActivationFunc activationFunction) {
        input = Tensor<float>(inputDims, inputData);
        expected = Tensor<float>(outputDims, outputData);
        activation = activationFunction;
    }

    void SetUpTensorsWithAlpha(const std::vector<float>& inputData, const std::vector<float>& outputData, const std::vector<int>& inputDims, const std::vector<int>& outputDims, const ActivationFuncWithAlpha activationFunction, float alphaValue) {
        input = Tensor<float>(inputDims, inputData);
        expected = Tensor<float>(outputDims, outputData);
        activationWithAlpha = activationFunction;
        alpha = alphaValue;
    }

    Tensor<T> processInput(const Tensor<T>& input) {
        return activation(input);
    }

    Tensor<T> processInputWithAlpha(const Tensor<T>& input) {
        return activationWithAlpha(input, alpha);
    }

    void ExpectTensorNear(const double abs_error = 1e-2) {
        const Tensor<float>& actual = processInput(input);
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }

    void ExpectTensorNearWithAlpha(const double abs_error = 1e-2) {
        const Tensor<float>& actual = processInputWithAlpha(input);
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }
};

// Sigmoid Activation Function Tests
class SigmoidActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activation = ActivationFunction<float>::sigmoid;
    }
};

TEST_F(SigmoidActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> outputData = {0.5f, 0.7310586f, 0.880797f, 0.9525741f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::sigmoid);
    ExpectTensorNear(1e-2);
}

TEST_F(SigmoidActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1.0f};

    SetUpTensors(inputData, outputData, {1, 1}, {1, 1}, ActivationFunction<float>::sigmoid);
    ExpectTensorNear(1e-2);
}

// Softmax Activation Function Tests
class SoftMaxActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activation = ActivationFunction<float>::softmax;
    }
};

TEST_F(SoftMaxActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {1.0f, 2.0f, 3.0f};
    const std::vector<float> outputData = {0.0900306f, 0.2447285f, 0.6652409f};

    SetUpTensors(inputData, outputData, {1, 3}, {1, 3}, ActivationFunction<float>::softmax);
    ExpectTensorNear(1e-2);
}

TEST_F(SoftMaxActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, 1000.f, 1000.f};
    const std::vector<float> outputData = {0.3333333f, 0.3333333f, 0.3333333f};

    SetUpTensors(inputData, outputData, {1, 3}, {1, 3}, ActivationFunction<float>::softmax);
    ExpectTensorNear(1e-2);
}

// ReLU Activation Function Tests
class ReluActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activation = ActivationFunction<float>::relu;
    }
};

TEST_F(ReluActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {0.0f, 0.0f, 1.0f, 2.0f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::relu);
    ExpectTensorNear(1e-2);
}

TEST_F(ReluActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpTensors(inputData, outputData, {1, 1}, {1, 1}, ActivationFunction<float>::relu);
    ExpectTensorNear(1e-2);
}

// Leaky ReLU Activation Function Tests
class LeakyReluActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activationWithAlpha = ActivationFunction<float>::leakyRelu;
    }
};

TEST_F(LeakyReluActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {-0.01f, 0.0f, 1.0f, 2.0f};

    SetUpTensorsWithAlpha(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::leakyRelu, 0.01f);
    ExpectTensorNearWithAlpha(1e-2);
}

TEST_F(LeakyReluActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpTensorsWithAlpha(inputData, outputData, {1, 1}, {1, 1}, ActivationFunction<float>::leakyRelu, 0.01f);
    ExpectTensorNearWithAlpha(1e-2);
}

// ELU Activation Function Tests
class EluActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activationWithAlpha = ActivationFunction<float>::elu;
    }
};

TEST_F(EluActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {-0.6321206f, 0.0f, 1.0f, 2.0f};

    SetUpTensorsWithAlpha(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::elu, 1.0f);
    ExpectTensorNearWithAlpha(1e-2);
}

TEST_F(EluActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpTensorsWithAlpha(inputData, outputData, {1, 1}, {1, 1}, ActivationFunction<float>::elu, 1.0f);
    ExpectTensorNearWithAlpha(1e-2);
}

// Tanh Activation Function Tests
class TanhActivationFunctionTest : public ActivationFunctionTest<float> {
protected:
    void SetUp() override {
        activation = ActivationFunction<float>::tanh;
    }
};

TEST_F(TanhActivationFunctionTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {-0.7615942f, 0.0f, 0.7615942f, 0.9640276f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::tanh);
    ExpectTensorNear(1e-2);
}

TEST_F(TanhActivationFunctionTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1.0f};

    SetUpTensors(inputData, outputData, {1, 1}, {1, 1}, ActivationFunction<float>::tanh);
    ExpectTensorNear(1e-2);
}

// Derivative Tests
template <typename T>
class ActivationFunctionDerivativeTest : public ::testing::Test {
protected:
    Tensor<T> input;
    Tensor<T> expected;
    ActivationFunc activationDerivative;
    ActivationFuncWithAlpha activationDerivativeWithAlpha{};
    float alpha;

    ActivationFunctionDerivativeTest() : input(Tensor<float>({1, 1})), expected(Tensor<float>({1, 1})), activationDerivative(ActivationFunction<float>::reluDerivative), alpha(0.01f) {}

    void SetUpTensors(const std::vector<float>& inputData, const std::vector<float>& outputData, const std::vector<int>& inputDims, const std::vector<int>& outputDims, const ActivationFunc activationFunction) {
        input = Tensor<float>(inputDims, inputData);
        expected = Tensor<float>(outputDims, outputData);
        activationDerivative = activationFunction;
    }

    void SetUpTensorsWithAlpha(const std::vector<float>& inputData, const std::vector<float>& outputData, const std::vector<int>& inputDims, const std::vector<int>& outputDims, const ActivationFuncWithAlpha activationFunction, float alphaValue) {
        input = Tensor<float>(inputDims, inputData);
        expected = Tensor<float>(outputDims, outputData);
        activationDerivativeWithAlpha = activationFunction;
        alpha = alphaValue;
    }

    Tensor<T> processInput(const Tensor<T>& input) {
        return activationDerivative(input);
    }

    Tensor<T> processInputWithAlpha(const Tensor<T>& input) {
        return activationDerivativeWithAlpha(input, alpha);
    }

    void ExpectTensorNear(const double abs_error = 1e-2) {
        const Tensor<float>& actual = processInput(input);
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }

    void ExpectTensorNearWithAlpha(const double abs_error = 1e-2) {
        const Tensor<float>& actual = processInputWithAlpha(input);
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }
};

// Sigmoid Derivative Tests
class SigmoidDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivative = ActivationFunction<float>::sigmoidDerivative;
    }
};

TEST_F(SigmoidDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.0f, 1.0f, 2.0f, 3.0f};
    const std::vector<float> outputData = {0.25f, 0.1966119f, 0.1049936f, 0.0451767f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::sigmoidDerivative);
    ExpectTensorNear(1e-2);
}

TEST_F(SigmoidDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, -1000.f};
    const std::vector<float> outputData = {0.0f, 0.0f}; // Sigmoid derivative approaches 0 for large positive/negative values

    SetUpTensors(inputData, outputData, {1, 2}, {1, 2}, ActivationFunction<float>::sigmoidDerivative);
    ExpectTensorNear(1e-2);
}

// Softmax Derivative Tests
class SoftmaxDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivative = ActivationFunction<float>::softmaxDerivative;
    }
};

TEST_F(SoftmaxDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {1.0f, 2.0f, 3.0f};
    const std::vector<float> outputData = {0.0819251f, 0.1848365f, 0.2217124f}; // Example values, adjust as needed

    SetUpTensors(inputData, outputData, {1, 3}, {1, 3}, ActivationFunction<float>::softmaxDerivative);
    ExpectTensorNear(1e-2);
}

TEST_F(SoftmaxDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, 1000.f, 1000.f};
    const std::vector<float> outputData = {0.2222222f, 0.2222222f, 0.2222222f}; // Corrected values

    SetUpTensors(inputData, outputData, {1, 3}, {1, 3}, ActivationFunction<float>::softmaxDerivative);
    ExpectTensorNear(1e-2);
}

// ReLU Derivative Tests
class ReluDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivative = ActivationFunction<float>::reluDerivative;
    }
};

TEST_F(ReluDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {0.0f, 0.0f, 1.0f, 1.0f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::reluDerivative);
    ExpectTensorNear(1e-2);
}

TEST_F(ReluDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, -1000.f};
    const std::vector<float> outputData = {1.0f, 0.0f}; // ReLU derivative is 1 for positive values and 0 for negative values

    SetUpTensors(inputData, outputData, {1, 2}, {1, 2}, ActivationFunction<float>::reluDerivative);
    ExpectTensorNear(1e-2);
}

// Leaky ReLU Derivative Tests
class LeakyReluDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivativeWithAlpha = ActivationFunction<float>::leakyReluDerivative;
    }
};

TEST_F(LeakyReluDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {0.01f, 0.0f, 1.0f, 1.0f};

    SetUpTensorsWithAlpha(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::leakyReluDerivative, 0.01f);
    ExpectTensorNearWithAlpha(1e-2);
}

TEST_F(LeakyReluDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, -1000.f};
    const std::vector<float> outputData = {1.0f, 0.01f}; // Leaky ReLU derivative is 1 for positive values and alpha for negative values

    SetUpTensorsWithAlpha(inputData, outputData, {1, 2}, {1, 2}, ActivationFunction<float>::leakyReluDerivative, 0.01f);
    ExpectTensorNearWithAlpha(1e-2);
}

// ELU Derivative Tests
class EluDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivativeWithAlpha = ActivationFunction<float>::eluDerivative;
    }
};

TEST_F(EluDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {0.3678794f, 1.0f, 1.0f, 1.0f};

    SetUpTensorsWithAlpha(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::eluDerivative, 1.0f);
    ExpectTensorNearWithAlpha(1e-2);
}

TEST_F(EluDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, -1000.f};
    const std::vector<float> outputData = {1.0f, 0.0f}; // ELU derivative is 1 for positive values and alpha * exp(x) for large negative values

    SetUpTensorsWithAlpha(inputData, outputData, {1, 2}, {1, 2}, ActivationFunction<float>::eluDerivative, 1.0f);
    ExpectTensorNearWithAlpha(1e-2);
}

// Tanh Derivative Tests
class TanhDerivativeTest : public ActivationFunctionDerivativeTest<float> {
protected:
    void SetUp() override {
        activationDerivative = ActivationFunction<float>::tanhDerivative;
    }
};

TEST_F(TanhDerivativeTest, HandlesNormalCase) {
    const std::vector<float> inputData = {-1.0f, 0.0f, 1.0f, 2.0f};
    const std::vector<float> outputData = {0.4199743f, 1.0f, 0.4199743f, 0.0706508f};

    SetUpTensors(inputData, outputData, {2, 2}, {2, 2}, ActivationFunction<float>::tanhDerivative);
    ExpectTensorNear(1e-2);
}

TEST_F(TanhDerivativeTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f, -1000.f};
    const std::vector<float> outputData = {0.0f, 0.0f}; // Tanh derivative approaches 0 for large positive/negative values

    SetUpTensors(inputData, outputData, {1, 2}, {1, 2}, ActivationFunction<float>::tanhDerivative);
    ExpectTensorNear(1e-2);
}