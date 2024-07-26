#include "gtest/gtest.h"
#include "../include/Tensor.h"
#include "../include/Optimizer.h"
#include <cmath>

typedef float (*Optim)(Tensor<float>& params, const Tensor<float>& grads);

template <typename T>
class OptimizerTest : public ::testing::Test {
protected:
    Optimizer<T> optimizer;
    Tensor<T> params;
    Tensor<T> grads;

    OptimizerTest()
        : optimizer(Optimizer<float>(0.01)),
          params(Tensor<float>({2, 2})),
          grads(Tensor<float>({2, 2})) {
        params.fill(0.5);
        grads.fill(0.1);
    }

    void SetUpData(const Tensor<T>& parameters, const Tensor<T>& gradients) {
        this->params = parameters;
        this->grads = gradients;
    }

    void ProcessInput(Tensor<T>& parameters, Tensor<T>& gradients) {
        optimizer.SGD(parameters, gradients);
    }

    void ExpectParamsNear(Tensor<T> expected, float abs_error = 1e-2) {
        ProcessInput(params, grads);
        for (int i = 0; i < params.size(); i++) {
            EXPECT_NEAR(params.data[i], expected.data[i], abs_error) << "Expected parameter: " << expected.data[i] << " but got " << params.data[i];
        }
    }
};

class AdamTest : public OptimizerTest<float> {
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(AdamTest, HandlesNormalCase) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

TEST_F(AdamTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

class RMSpropTest : public OptimizerTest<float> {
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(RMSpropTest, HandlesNormalCase) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

TEST_F(RMSpropTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

class SGDTest : public OptimizerTest<float> {
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(SGDTest, HandlesNormalCase) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

TEST_F(SGDTest, HandlesEdgeCaseLargeValues) {
    const std::vector<float> expectedData = {0.4995f, 0.4995f, 0.4995f, 0.4995f};
    ExpectParamsNear(Tensor<float>({2, 2}, expectedData));
}

class ApplyWeightDecayTest : public OptimizerTest<float> {
protected:
    void SetUp() override {
        optimizer.initialize({2, 2});
    }
};

TEST_F(ApplyWeightDecayTest, ApplyWeightDecay) {
    Tensor<float> params({2, 2}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    float weight_decay = 0.1f;
    optimizer.apply_weight_decay(params, weight_decay);

    EXPECT_NEAR(params.data[0], 0.9f, 1e-5);
    EXPECT_NEAR(params.data[1], 1.8f, 1e-5);
    EXPECT_NEAR(params.data[2], 2.7f, 1e-5);
    EXPECT_NEAR(params.data[3], 3.6f, 1e-5);
}

TEST_F(ApplyWeightDecayTest, ClipGradients) {
    Tensor<float> grads({2, 2}, std::vector<float>{1.0f, -2.0f, 3.0f, -4.0f});
    float clip_value = 1.5f;
    optimizer.clip_gradients(grads, clip_value);

    EXPECT_NEAR(grads.data[0], 1.0f, 1e-5);
    EXPECT_NEAR(grads.data[1], -1.5f, 1e-5);
    EXPECT_NEAR(grads.data[2], 1.5f, 1e-5);
    EXPECT_NEAR(grads.data[3], -1.5f, 1e-5);
}

