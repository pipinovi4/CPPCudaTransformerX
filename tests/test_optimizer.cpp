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

    void SetUpData(const Tensor<T>& parameters, const Tensor<T>& gradients, const Tensor<T>& expected_parameters, const Tensor<T>& expected_gradients) {
        this->input_params = parameters;
        this->input_grads = gradients;
        this->expected_params = expected_parameters;
        this->expected_grads = expected_gradients;
    }

    void ExpectParamsNear(const float abs_error = 1e-2) {
        for (size_t i = 0; i < output_params.size(); i++) {
            EXPECT_NEAR(output_params.data[i], expected_params.data[i], abs_error)
                << "Expected parameter: " << expected_params.data[i] << " but got " << output_params.data[i];
        }
        for (size_t i = 0; i < output_grads.size(); i++) {
            EXPECT_NEAR(output_grads.data[i], expected_grads.data[i], abs_error)
                << "Expected grads: " << expected_grads.data[i] << " but got " << output_grads.data[i];
        }
    }
};

// class AdamTest : public ::testing::Test {
// protected:
//     Optimizer<float>::Adam optimizer;
//     Tensor<float> params;
//     Tensor<float> grads;
//     Tensor<float> expected_params;
//
//     AdamTest()
//         : optimizer({{2, 2}}, 0.01, *std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100)),
//           params({2, 2}), grads({2, 2}), expected_params({10, 19}) {}
//
//     void SetUp() override {
//         optimizer.initialize({{2, 2}});
//     }
//
//     static void ExpectParamsNear(const Tensor<float>& expected, const Tensor<float>& actual, const float abs_error = 1e-5) {
//         for (size_t i = 0; i < expected.data.size(); ++i) {
//             EXPECT_NEAR(expected.data[i], actual.data[i], abs_error) << "Mismatch at index " << i;
//         }
//     }
// };


// TEST_F(AdamTest, HandleNormalCase) {
//     // Initialize params and grads with some values
//     params.data = {0.1, 0.2, 0.3, 0.4};
//     grads.data = {0.01, 0.02, 0.03, 0.04};
//
//     // Expected values after applying the Adam optimizer
//     expected_params.data = {0.099, 0.199, 0.299, 0.399};
//
//     // Wrap tensors in reference wrappers and store in vectors
//     const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(params)};
//     const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(grads)};
//
//     // Perform the update
//     optimizer.update(params_vec, grads_vec, 1);
//
//     // Check the results
//     ExpectParamsNear(expected_params, params);
// }
//
// TEST_F(AdamTest, HandleZeroGradients) {
//     // Initialize params and grads with zero gradients
//     params.data = {0.1, 0.2, 0.3, 0.4};
//     grads.data = {0.0, 0.0, 0.0, 0.0};
//
//     // Expected params should be the same as initial since grads are zero
//     expected_params.data = {0.1, 0.2, 0.3, 0.4};
//
//     // Wrap tensors in reference wrappers and store in vectors
//     const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(params)};
//     const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(grads)};
//
//     // Perform the update
//     optimizer.update(params_vec, grads_vec, 1);
//
//     // Check the results
//     ExpectParamsNear(expected_params, params);
// }
//
// TEST_F(AdamTest, HandleLargeGradients) {
//     // Initialize params and grads with large gradients
//     params.data = {0.1, 0.2, 0.3, 0.4};
//     grads.data = {1.0, 1.0, 1.0, 1.0};
//
//     // Expected values after applying the Adam optimizer
//     // These values are hypothetical and should be calculated based on the Adam formula
//     expected_params.data = {0.099, 0.199, 0.299, 0.399};  // Replace with correct expected values
//
//     // Wrap tensors in reference wrappers and store in vectors
//     const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(params)};
//     const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(grads)};
//
//     // Perform the update
//     optimizer.update(params_vec, grads_vec, 1);
//
//     // Check the results
//     ExpectParamsNear(expected_params, params);
// }

class SGDTest : public OptimizerTest<float> {
public:
    Optimizer<float>::SGD optimizer;

    SGDTest()
        : OptimizerTest<float>(), optimizer({{2, 2}}, 0.01, *std::make_shared<Optimizer<float>::LearningRateSchedule::StepDecaySchedule>(0.01, 0.1, 100)) {}

protected:
    void SetUp() override {
        optimizer.initialize({{2, 2}});
    }
};

TEST_F(SGDTest, HandleNormalCase) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}),
              Tensor<float>({2, 2}, std::vector<float>{0.0999, 0.1998, 0.2997, 0.3996}),
              Tensor<float>({2, 2}, std::vector<float>{0.01, 0.02, 0.03, 0.04}));

    output_params = input_params;
    output_grads = input_grads;

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(output_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(output_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);
}

TEST_F(SGDTest, HandleZeroGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{0.0, 0.0, 0.0, 0.0}));

    output_params = input_params;
    output_grads = input_grads;

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(output_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(output_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);
}

TEST_F(SGDTest, HandleLargeGradients) {
    SetUpData(Tensor<float>({2, 2}, std::vector<float>{0.1, 0.2, 0.3, 0.4}),
              Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}),
              Tensor<float>({2, 2}, std::vector<float>{0.09, 0.19, 0.29, 0.39}),
              Tensor<float>({2, 2}, std::vector<float>{1.0, 1.0, 1.0, 1.0}));

    output_params = input_params;
    output_grads = input_grads;

    // Wrap in std::reference_wrapper
    const std::vector<std::reference_wrapper<Tensor<float>>> params_vec = {std::ref(output_params)};
    const std::vector<std::reference_wrapper<Tensor<float>>> grads_vec = {std::ref(output_grads)};

    // Call the update function with wrapped vectors
    optimizer.update(params_vec, grads_vec, 0);
}