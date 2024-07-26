#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include <cmath>

typedef float (*LossFunc)(const Tensor<float>& predictions, const Tensor<float>& targets, double epsilon);

template <typename T>
class LossFunctionTest : public ::testing::Test
{
protected:
    LossFunc lossFunction;
    Tensor<T> input;
    Tensor<T> targets;
    double epsilon;

    LossFunctionTest()
        : lossFunction(nullptr),
          input(Tensor<float>({2, 2})),
          targets(Tensor<float>({2, 2})),
          epsilon(1e-7) {}

    void SetUpData(const Tensor<T>& input, const Tensor<T>& targets, double epsilon = 1e-7) {
        this->input = input;
        this->targets = targets;
        this->epsilon = epsilon;
    }

    T ProcessInput(const Tensor<T>& input, const Tensor<T>& targets, double epsilon) {
        return lossFunction(input, targets, epsilon);
    }

    void ExpectLossNear(const T& expected, const double abs_error = 1e-2) {
        T actual = ProcessInput(input, targets, epsilon);
        if (std::isnan(actual)) {
            ADD_FAILURE() << "Actual value is NaN";
        } else {
            EXPECT_NEAR(actual, expected, abs_error) << "Expected loss: " << expected << " but got " << actual;
        }
    }
};

class BinaryCrossEntropyLossTest : public LossFunctionTest<float> {
protected:
    void SetUp() override {
        lossFunction = LossFunction<float>::binaryCrossEntropyLoss;
    }
};

TEST_F(BinaryCrossEntropyLossTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.423423f, 0.423423f, 0.423423f, 0.423423f};
    const std::vector<float> outputData = {0.119203f, 0.880797f, 0.7310586f, 0.26894143f};

    SetUpData(Tensor<float>({2, 2}, inputData), Tensor<float>({2, 2}, outputData));
    ExpectLossNear(0.705015f, 1e-2);
}

TEST_F(BinaryCrossEntropyLossTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(-15926.442f, 1e-2);
}

class CrossEntropyLossTest : public LossFunctionTest<float> {
protected:
    void SetUp() override {
        lossFunction = LossFunction<float>::crossEntropyLoss;
    }
};

TEST_F(CrossEntropyLossTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.1f, 0.2f, 0.7f};
    const std::vector<float> targetData = {0.0f, 0.0f, 1.0f};

    SetUpData(Tensor<float>({3}, inputData), Tensor<float>({3}, targetData));
    ExpectLossNear(0.356675f, 1e-2);
}

TEST_F(CrossEntropyLossTest, HandlesEdgeCase) {
    const std::vector<float> inputData = {1e-10f, 1e-10f, 1e-10f};
    const std::vector<float> targetData = {0.0f, 0.0f, 1.0f};

    SetUpData(Tensor<float>({3}, inputData), Tensor<float>({3}, targetData));
    ExpectLossNear(16.118095f, 1e-2);
}

class MeanAbsoluteErrorLossTest : public LossFunctionTest<float> {
protected:
    void SetUp() override {
        lossFunction = LossFunction<float>::meanAbsoluteError;
    }
};

TEST_F(MeanAbsoluteErrorLossTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.1f, 0.2f, 0.7f};
    const std::vector<float> targetData = {0.0f, 0.0f, 1.0f};

    SetUpData(Tensor<float>({3}, inputData), Tensor<float>({3}, targetData));
    ExpectLossNear(0.2f, 1e-2);
}

TEST_F(MeanAbsoluteErrorLossTest, HandlesEdgeCase) {
    const std::vector<float> inputData = {1.0f, 1.0f, 1.0f};
    const std::vector<float> targetData = {0.0f, 0.0f, 0.0f};

    SetUpData(Tensor<float>({3}, inputData), Tensor<float>({3}, targetData));
    ExpectLossNear(1.0f, 1e-2);
}