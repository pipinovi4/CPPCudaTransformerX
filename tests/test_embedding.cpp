#include "gtest/gtest.h"
#include "../include/Embedding.h"
#include "../include/Tensor.h"
#include "../include/Optimizer.h"

class EmbeddingTest : public ::testing::Test {
protected:
    Embedding<float>* embedding = nullptr;
    Optimizer<float>::LearningRateSchedule* lr_schedule = nullptr;
    Optimizer<float>* optimizer = nullptr;

    void SetUpEmbedding(const int vocab_size, const int embedding_dims, const float learning_rate, const float decay_rate) {
        lr_schedule = new Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule(learning_rate, decay_rate);
        embedding = new Embedding<float>(vocab_size, embedding_dims, *lr_schedule);

        // Accamulate params shape for optimizer
        std::vector<std::vector<int>> params_shape;
        params_shape.push_back(embedding->parameters()[0].get().shape());

        // Define new optimizer
        optimizer = new Optimizer<float>::Adam(embedding->parameters(), learning_rate, *lr_schedule);
    }

    [[nodiscard]] Tensor<float> ProcessInputForward(const Tensor<float>& input_data) const {
        return embedding->forward(input_data);
    }

    void ProcessInputBackward(Tensor<float>& grad_data) const {
        embedding->backward(grad_data);
    }

    void UpdateEmbeddingWeights(const int epoch) const {
        optimizer->update(embedding->parameters(), embedding->gradients(), epoch);
    }

    static void ExpectOutputShape(const Tensor<float>& output, const std::vector<int>& expected_shape) {
        EXPECT_EQ(output.shape(), expected_shape);
    }

    static void ExpectAllFinite(const Tensor<float>& tensor) {
        for (const auto& val : tensor.data) {
            EXPECT_TRUE(std::isfinite(val));
        }
    }

    void TearDown() override {
        delete embedding;
        delete lr_schedule;
    }
};

// Test the forward pass of the Embedding layer
TEST_F(EmbeddingTest, HandlesForwardPass) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    Tensor<float> input_data({3}); // Batch size = 2, Sequence length = 3
    input_data.fill(1.0); // Filling input with a valid index

    const Tensor<float> output = ProcessInputForward(input_data);
    ExpectOutputShape(output, {3, 5}); // Expected shape: [Batch size, Sequence length, Embedding dims]
    ExpectAllFinite(output); // Ensure all values are finite
}

// Test the backward pass of the Embedding layer
TEST_F(EmbeddingTest, HandlesBackwardPass) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    Tensor<float> input_data({3});
    input_data.fill(1.0);

    const Tensor<float> output = ProcessInputForward(input_data);
    Tensor<float> grad_data(output.shape());
    grad_data.fill(0.5); // Fill gradient data with some value

    ProcessInputBackward(grad_data);

    const auto& grad = embedding->gradients();
    ASSERT_EQ(grad[0].get().shape()[0], 10); // Vocabulary size = 10
    ASSERT_EQ(grad[0].get().shape()[1], 5);  // Embedding dimensions = 5
    ExpectAllFinite(grad[0].get()); // Ensure all values are finite
}

// Test the update of weights after backpropagation
TEST_F(EmbeddingTest, HandlesWeightUpdate) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    Tensor<float> input_data({3});
    input_data.fill(1.0);

    const Tensor<float> output = ProcessInputForward(input_data);
    Tensor<float> grad_data(output.shape());
    grad_data.fill(0.5);

    ProcessInputBackward(grad_data);
    UpdateEmbeddingWeights(1);

    const auto& weights = embedding->parameters();
    ExpectAllFinite(weights[0].get()); // Ensure all values are finite
}

// Test learning rate scheduling
TEST_F(EmbeddingTest, HandlesLearningRateSchedule) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    Tensor<float> input_data({3});
    input_data.fill(1.0);

    const Tensor<float> output = ProcessInputForward(input_data);
    Tensor<float> grad_data(output.shape());
    grad_data.fill(0.5);
    ProcessInputBackward(grad_data);
    UpdateEmbeddingWeights(1);

    const auto& weights = embedding->parameters();
    ExpectAllFinite(weights[0].get()); // Ensure all values are finite

    UpdateEmbeddingWeights(2); // Update again to see if learning rate decays
    ExpectAllFinite(weights[0].get()); // Ensure all values are finite
}

// Test setting weights for the Embedding layer
TEST_F(EmbeddingTest, HandlesSetWeights) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    Tensor<float> new_weights({10, 5});
    new_weights.fill(0.1);

    embedding->setWeights(new_weights);
    const auto& weights = embedding->parameters();

    ExpectOutputShape(weights[0].get(), {10, 5});
    ExpectAllFinite(weights[0].get()); // Ensure all values are finite
}

// Test that an invalid weight shape throws an exception
TEST_F(EmbeddingTest, ThrowsOnInvalidWeightShape) {
    SetUpEmbedding(10, 5, 0.01, 0.9);
    const Tensor<float> invalid_weights({5, 10}); // Invalid shape

    EXPECT_THROW(embedding->setWeights(invalid_weights), std::invalid_argument);
}
