#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/Optimizer.h"
#include <cmath>

TEST(StepDecayScheduleTest, GetLearningRate) {
    Optimizer<float>::LearningRateSchedule::StepDecaySchedule lr_schedule(0.01, 0.1, 100);

    EXPECT_NEAR(lr_schedule.getLearningRate(0), 0.01, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(50), 0.01 * std::pow(0.1, 50.0 / 100.0), 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(100), 0.001, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(200), 0.0001, 1e-5);
}

TEST(ExponentialDecayScheduleTest, GetLearningRate) {
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule lr_schedule(0.01, 0.9);

    EXPECT_NEAR(lr_schedule.getLearningRate(0), 0.01, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(1), 0.01 * 0.9, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(2), 0.01 * 0.9 * 0.9, 1e-5);
    EXPECT_NEAR(lr_schedule.getLearningRate(3), 0.01 * std::pow(0.9, 3), 1e-5);
}

template <typename T>
class OptimizerTest : public ::testing::Test {
protected:
    Tensor<T> input_params;
    Tensor<T> input_grads;
    Tensor<T> expected_params;

    OptimizerTest()
        : input_params({2, 2}), input_grads({2, 2}), expected_params({2, 2}) {}

    void SetUpData(const Tensor<T>& parameters, const Tensor<T>& gradients, const Tensor<T>& expected_parameters) {
        this->input_params = parameters;
        this->input_grads = gradients;
        this->expected_params = expected_parameters;
    }

    void ExpectParamsNear(const Tensor<T>& actual_params, const float abs_error = 1e-5) {
        for (size_t i = 0; i < expected_params.size(); i++) {
            EXPECT_NEAR(actual_params.data[i], expected_params.data[i], abs_error)
                << "Expected parameter: " << expected_params.data[i] << " but got " << actual_params.data[i];
        }
    }
};

class AdamTest : public OptimizerTest<float> {
protected:
    Optimizer<float>::Adam optimizer;

    AdamTest()
        : optimizer({{2, 2}}, 0.01, *std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100)) {}

    void SetUp() override {
        optimizer.initialize_params({{2, 2}});
    }
};

TEST_F(AdamTest, HandleNormalCase) {
    // Initialize params and grads with some values
    input_params.data = {0.1, 0.2, 0.3, 0.4};
    input_grads.data = {0.01, 0.02, 0.03, 0.04};

    // Calculate expected values based on Adam optimizer calculations
    expected_params.data = {0.1, 0.2, 0.3, 0.4};  // This needs to be computed properly

    // Wrap tensors in reference wrappers and store in vectors
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 0);

    // Check the results
    ExpectParamsNear(input_params);
}

TEST_F(AdamTest, HandleZeroGradients) {
    // Initialize params and grads with zero gradients
    input_params.data = {0.1, 0.2, 0.3, 0.4};
    input_grads.data = {0.0, 0.0, 0.0, 0.0};

    // Expected params should be the same as initial since grads are zero
    expected_params.data = {0.1, 0.2, 0.3, 0.4};

    // Wrap tensors in reference wrappers and store in vectors
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 0);

    // Check the results
    ExpectParamsNear(input_params);
}

TEST_F(AdamTest, HandleLargeGradients) {
    // Initialize params and grads with large gradients
    input_params.data = {0.1, 0.2, 0.3, 0.4};
    input_grads.data = {1.0, 1.0, 1.0, 1.0};

    // Expected values after applying the Adam optimizer (need to be computed correctly)
    expected_params.data = {0.1, 0.2, 0.3, 0.4};  // Replace with correct expected values

    // Wrap tensors in reference wrappers and store in vectors
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 1);

    // Check the results
    ExpectParamsNear(input_params);
}

class SGDTest : public OptimizerTest<float> {
public:
    Optimizer<float>::SGD optimizer;

    SGDTest()
        : OptimizerTest<float>(), optimizer({{2, 2}}, 0.01, *std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100)) {}

protected:
    void SetUp() override {
        optimizer.initialize_params({{2, 2}});
    }
};

TEST_F(SGDTest, HandleNormalCase) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.0999, 0.1998, 0.2997, 0.3996}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}

TEST_F(SGDTest, HandleZeroGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}

TEST_F(SGDTest, HandleLargeGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.09, 0.19, 0.29, 0.39}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}

// Add RMSprop test cases
class RMSpropTest : public OptimizerTest<float> {
public:
    Optimizer<float>::RMSprop optimizer;

    RMSpropTest()
        : OptimizerTest<float>(), optimizer({{2, 2}}, 0.01, 0.9, 1e-8, *std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100)) {}

protected:
    void SetUp() override {
        optimizer.initialize_params({{2, 2}});
    }
};

TEST_F(RMSpropTest, HandleNormalCase) {
    // Initialize params and grads with some values
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.06837, 0.16837, 0.26837, 0.36837}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}

TEST_F(RMSpropTest, HandleZeroGradients) {
    // Initialize params and grads with zero gradients
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}

TEST_F(RMSpropTest, HandleLargeGradients) {
    // Initialize params and grads with large gradients
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.06837, 0.16837, 0.26837, 0.36837}));

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(input_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(input_grads)};

    // Perform the update
    optimizer.update(params_vec, grads_vec, 0);

    // Validate the parameters after the update
    ExpectParamsNear(input_params);
}
