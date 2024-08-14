#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include <cmath>

template <typename T>
class LossFunctionTest : public ::testing::Test
{
protected:
    LossFunction<T>* lossFunction;
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

    T ProcessInputForward(const Tensor<T>& input, const Tensor<T>& targets, double epsilon) {
        return lossFunction->forward(input, targets);
    }

    T ProcessInputBackward(const Tensor<T>& input, const Tensor<T>& targets, double epsilon) {
        return lossFunction->backward(input, targets);
    }

    void ExpectLossNear(const T& expected, const double abs_error = 1e-2) {
        T actual = ProcessInputForward(input, targets, epsilon);
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
        lossFunction = new LossFunction<float>::binaryCrossEntropyLoss();
    }

    void TearDown() override {
        delete lossFunction;
    }
};

TEST_F(BinaryCrossEntropyLossTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.423423f, 0.423423f, 0.423423f, 0.423423f};
    const std::vector<float> outputData = {0.119203f, 0.880797f, 0.7310586f, 0.26894143f};

    SetUpData(Tensor<float>({2, 2}, inputData), Tensor<float>({2, 2}, outputData));
    ExpectLossNear(0.70501500368118286, 1e-2);
}

TEST_F(BinaryCrossEntropyLossTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(-15926.4423828125, 1e-2);
}

TEST_F(BinaryCrossEntropyLossTest, HandlesEdgeCaseSmallValues) {
    const std::vector<float> inputData = {1e-7f};
    const std::vector<float> outputData = {1e-7f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear( 1.7310188695773832e-06f, 1e-2);
}

class CrossEntropyLossTest : public LossFunctionTest<float> {
public:
    void SetUp() override {
        lossFunction = new LossFunction<float>::crossEntropyLoss();
    }

    void TearDown() override {
        delete lossFunction;
    }
};

TEST_F(CrossEntropyLossTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.423423f, 0.423423f, 0.423423f, 0.423423f};
    const std::vector<float> outputData = {0.119203f, 0.880797f, 0.7310586f, 0.26894143f};

    SetUpData(Tensor<float>({2, 2}, inputData), Tensor<float>({2, 2}, outputData));
    ExpectLossNear(0.85938370227813721, 1e-2);
}

TEST_F(CrossEntropyLossTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(0.f, 1e-2);
}

TEST_F(CrossEntropyLossTest, HandlesEdgeCaseSmallValues) {
    const std::vector<float> inputData = {1e-7f};
    const std::vector<float> outputData = {1e-7f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(1.7310188695773832e-06f, 1e-2);
}

class MeanSquaredErrorTest : public LossFunctionTest<float> {
public:
    void SetUp() override {
    lossFunction = new LossFunction<float>::meanSquaredError();
    }

    void TearDown() override {
        delete lossFunction;
    }
};

TEST_F(MeanSquaredErrorTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.423423f, 0.423423f, 0.423423f, 0.423423f};
    const std::vector<float> outputData = {0.119203f, 0.880797f, 0.7310586f, 0.26894143f};

    SetUpData(Tensor<float>({2, 2}, inputData), Tensor<float>({2, 2}, outputData));
    ExpectLossNear(0.2101224958896637f, 1e-2);
}

TEST_F(MeanSquaredErrorTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(0.f, 1e-2);
}

TEST_F(MeanSquaredErrorTest, HandlesEdgeCaseSmallValues) {
    const std::vector<float> inputData = {1e-7f};
    const std::vector<float> outputData = {1e-7f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(1.7310188695773832e-06f, 1e-2);
}

class MeanAbsoluteErrorTest : public LossFunctionTest<float> {
public:
    void SetUp() override {
        lossFunction = new LossFunction<float>::meanAbsoluteError();
    }

    void TearDown() override {
        delete lossFunction;
    }
};

TEST_F(MeanAbsoluteErrorTest, HandlesNormalCase) {
    const std::vector<float> inputData = {0.423423f, 0.423423f, 0.423423f, 0.423423f};
    const std::vector<float> outputData = {0.119203f, 0.880797f, 0.7310586f, 0.26894143f};

    SetUpData(Tensor<float>({2, 2}, inputData), Tensor<float>({2, 2}, outputData));
    ExpectLossNear(0.61185556650161743f, 1e-2);
}

TEST_F(MeanAbsoluteErrorTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> inputData = {1000.f};
    const std::vector<float> outputData = {1000.f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(0.f, 1e-2);
}

TEST_F(MeanAbsoluteErrorTest, HandlesEdgeCaseSmallValues) {
    const std::vector<float> inputData = {1e-7f};
    const std::vector<float> outputData = {1e-7f};

    SetUpData(Tensor<float>({1}, inputData), Tensor<float>({1}, outputData));
    ExpectLossNear(1.7310188695773832e-06f, 1e-2);
}
