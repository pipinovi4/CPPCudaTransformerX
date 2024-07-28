#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

template <typename T>
class ActivationFunctionTest : public ::testing::Test {
protected:
    Tensor<T> input;
    Tensor<T> expected;
    Tensor<T> processedDataForward;
    Tensor<T> processedDataBackward;

    ActivationFunctionTest() : input(Tensor<float>({1, 1})), expected(Tensor<float>({1, 1})), processedDataForward(Tensor<float>({1, 1})), processedDataBackward(Tensor<float>({1, 1})) {}

    void SetUpData(const Tensor<T>& input, const Tensor<T>& expected) {
        this->input = input;
        this->expected = expected;
    }

    template <typename F>
    void ProcessDataForward(F activationFunction) {
        processedDataForward = activationFunction.forward(input);
    }

    template <typename F>
    void ProcessDataBackward(F activationFunction) {
        processedDataBackward = activationFunction.backward(input);
    }

    void ExpectTensorNear(Tensor<T> actual, const double abs_error = 1e-2) {
        ASSERT_EQ(actual.shape(), expected.shape()) << "Tensor shapes do not match.";
        for (size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_NEAR(actual.data[i], expected.data[i], abs_error)
                << "Mismatch at index " << i << ": expected " << expected.data[i] << " but got " << actual.data[i];
        }
    }
};

class Sigmoid : public ActivationFunctionTest<float> {
public:
    ActivationFunction<float>::Sigmoid sigmoid;

    Sigmoid() {
        sigmoid = ActivationFunction<float>::Sigmoid();
   }
};

TEST_F(Sigmoid, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 0.19661193f, -0.19661193f, 0.11750186f, -0.11750186f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.5f, 0.7310586f, 0.26894143f, 0.62245935f, 0.37754068f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.0f, 0.19661193f, -0.19661193f, 0.11750186f, -0.11750186f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(sigmoid);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(sigmoid);
    ExpectTensorNear(expectedBackward, 1e-6);
}

TEST_F(Sigmoid, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, 1000.0f, -1000.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, -1000.0f, 1000.0f, -1000.0f});

    const auto expectedFoward = Tensor<float>(std::vector<float>{1.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.0f, 0.0f, -0.0f, 0.0f, -0.0f});

    SetUpData(inputForward, expectedFoward);
    ProcessDataForward(sigmoid);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataForward(sigmoid);
    ExpectTensorNear(expectedBackward, 1e-6);
}

class Softmax : public ActivationFunctionTest<float> {
protected:
    ActivationFunction<float>::Softmax softmax;

    Softmax() {
        softmax = ActivationFunction<float>::Softmax();
    }
};

TEST_F(Softmax, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.0160293f, 0.04357216f, 0.11844141f, 0.32195713f, 0.0160293f, 0.04357216f, 0.11844141f, 0.32195713f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.01577236f, 0.08334725f, 0.31323913f, 0.87320295f, 0.01577236f, 0.08334725f, 0.31323913f, 0.87320295f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(softmax);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataForward(softmax);
    ExpectTensorNear(expectedBackward, 1e-6);
}

TEST_F(Softmax, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, 1000.0f, -1000.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, -1000.0f, 1000.0f, -1000.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.33333333f, 0.33333333f, 0.0f,0.33333333f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{250.0f, 0.0f, -0.0f, 250.0f, -0.f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(softmax);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataForward(softmax);
    ExpectTensorNear(expectedBackward, 1e-6);
}

class ReLU : public ActivationFunctionTest<float> {
protected:
    ActivationFunction<float>::ReLU relu;

    ReLU() {
        relu = ActivationFunction<float>::ReLU();
    }
};

TEST_F(ReLU, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 0.5f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(relu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(relu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

TEST_F(ReLU, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, 0.0f, 1000.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(relu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(relu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

class LeakyReLU : public ActivationFunctionTest<float> {
protected:
    ActivationFunction<float>::LeakyReLU leaky_relu;

    LeakyReLU() {
        leaky_relu = ActivationFunction<float>::LeakyReLU();
    }
};

TEST_F(LeakyReLU, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});

    const auto expectedForward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, -10.0f, 1000.0f, -10.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, -10.0f, 1000.0f, -10.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(leaky_relu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(leaky_relu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

TEST_F(LeakyReLU, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, 0.0f, 1000.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(leaky_relu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(leaky_relu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

class ELU : public ActivationFunctionTest<float> {
protected:
    ActivationFunction<float>::ELU elu;

    ELU() {
        elu = ActivationFunction<float>::ELU();
    }
};

TEST_F(ELU, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 0.5f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(elu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(elu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

TEST_F(ELU, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, 0.0f, 1000.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(elu);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(elu);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

class Tanh : public ActivationFunctionTest<float> {
protected:
    ActivationFunction<float>::Tanh tanh;

    Tanh() {
        tanh = ActivationFunction<float>::Tanh();
    }
};

TEST_F(Tanh, HandlesNormalCase) {
    const auto inputForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});
    const auto inputBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, -1.0f, 0.5f, -0.5f});

    const auto expectedForward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{0.0f, 1.0f, 0.0f, 0.5f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(tanh);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(tanh);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

TEST_F(Tanh, HandlesEdgeCaseLargeValues) {
    const auto inputForward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});
    const auto inputBackward = Tensor<float>(std::vector<float>{1000.0f, -1000.0f, 0.0f, 1000.0f, -1000.0f});

    const auto expectedForward = Tensor<float>(std::vector<float>{1000.0f, 0.0f, 0.0f, 1000.0f, 0.0f});
    const auto expectedBackward = Tensor<float>(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f, 0.0f});

    SetUpData(inputForward, expectedForward);
    ProcessDataForward(tanh);
    ExpectTensorNear(processedDataForward, 1e-6);

    SetUpData(inputBackward, expectedBackward);
    ProcessDataBackward(tanh);
    ExpectTensorNear(processedDataBackward, 1e-6);
}

