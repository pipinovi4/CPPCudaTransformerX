#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/Optimizer.h"
#include <cmath>

TEST(StepDecayScheduleTest, GetLearningRate) {
    Optimizer<float>::LearningRateSchedule::StepDecaySchedule lr_schedule(0.01, 0.1, 100);

    EXPECT_NEAR(lr_schedule.getLearningRate(0), 0.01, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(50), 0.0031622776, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(100), 0.001, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(200), 0.0001, 1e-5);
}

TEST(ExponentialDecayScheduleTest, GetLearningRate) {
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule lr_schedule(0.01, 0.9);

    EXPECT_NEAR(lr_schedule.getLearningRate(0), 0.01, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(1), 0.009, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(2), 0.0081, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(3), 0.00729, 1e-5);
}

typedef float (*Optim)(Tensor<float>& params, const Tensor<float>& grads);

template <typename T>
class OptimizerTest : public ::testing::Test {
protected:
    Tensor<T> input_params;
    Tensor<T> input_grads;
    Tensor<T> output_params;
    Tensor<T> output_grads;

    Tensor<T> expected_params;
    Tensor<T> expected_grads;

    OptimizerTest()
        : input_params({2, 2}), input_grads({2, 2}), output_params({2, 2}), output_grads({2, 2}),
          expected_params({2, 2}), expected_grads({2, 2}) {}

    void SetUpData(const Tensor<T>& parameters, const Tensor<T>& gradients, const Tensor<T> expected_parameters, const Tensor<T> expected_gradients) {
        this->input_params = parameters;
        this->input_grads = gradients;
        this->expected_params = expected_parameters;
        this->expected_grads = expected_gradients;
    }

    void ExpectParamsNear(const float abs_error = 1e-2) {
        for (int i = 0; i < output_params.size(); i++) {
            EXPECT_NEAR(output_params.data[i], expected_params.data[i], abs_error) << "Expected parameter: " << expected_params.data[i] << " but got " << output_params.data[i];
        }
        for (int i = 0; i < output_grads.size(); i++) {
            EXPECT_NEAR(output_grads.data[i], expected_grads.data[i], abs_error) << "Expected grads: " << expected_grads.data[i] << " but got " << output_grads.data[i];
        }
    }
};

class AdamTest : public OptimizerTest<float> {
public:
    Optimizer<float>::Adam optimizer;
    AdamTest()
        : OptimizerTest<float>(), optimizer(0.9, 0.999, 1e-8, 0.01, *new Optimizer<float>::LearningRateSchedule::StepDecaySchedule(0.01, 0.1, 100)) {}
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(AdamTest, HandleNormalCase) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.099, 0.199, 0.299, 0.399}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(AdamTest, HandleZeroGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(AdamTest, HandleLargeGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.099, 0.199, 0.299, 0.399}), Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

class RMSPropTest : public OptimizerTest<float> {
public:
    Optimizer<float>::RMSprop optimizer;
    RMSPropTest()
        : OptimizerTest<float>(), optimizer(0.01, 0.9, 1e-8, *new Optimizer<float>::LearningRateSchedule::StepDecaySchedule(0.01, 0.1, 100)) {}
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(RMSPropTest, HandleNormalCase) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.06839302, 0.16838118, 0.268379, 0.36837822}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(RMSPropTest, HandleZeroGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(RMSPropTest, HandleLargeGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.068377226591110229, 0.16837722063064575, 0.26837724447250366, 0.36837723851203918}),
              Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

class SGDTest : public OptimizerTest<float> {
public:
    Optimizer<float>::SGD optimizer;
    SGDTest()
        : OptimizerTest<float>(), optimizer(0.01, *new Optimizer<float>::LearningRateSchedule::StepDecaySchedule(0.01, 0.1, 100)) {}
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(SGDTest, HandleNormalCase) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.0999, 0.1998, 0.2997, 0.3996}), Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(SGDTest, HandleZeroGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}

TEST_F(SGDTest, HandleLargeGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}), Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.09, 0.19, 0.29, 0.39}), Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}));
    output_params = input_params;
    output_grads = input_grads;
    optimizer.update(output_params, output_grads, 0);
    ExpectParamsNear();
}
