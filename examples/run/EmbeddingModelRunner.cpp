#include "../EmbeddingModel.h"
#include "../../include/Tokenizer.h"
#include "../../include/Optimizer.h"
#include "../../include/LossFunction.h"
#include "../../include/Tensor.h"
#include <vector>
#include <iostream>
#include <random>

/**
 * \brief Runs the EmbeddingModel application.
 *
 * This function initializes the EmbeddingModel with the specified parameters and
 * performs multiple forward and backward passes over a set number of epochs.
 * It also handles any exceptions that may occur during the process.
 *
 * \return int Returns 0 on successful execution, or a non-zero value if an error occurs.
 */

// Function to generate synthetic data
std::pair<Tensor<float>, Tensor<float>> generate_synthetic_data(int num_samples, const int vocab_size, int sequence_length) {
    // Generate random input data (sequences)
    Tensor<float> X = Tensor<float>::uniform({num_samples, sequence_length}, 0, static_cast<float>(vocab_size));

    // Generate random binary labels (0 or 1)
    Tensor<float> y = Tensor<float>::uniform({5120*4}, 0, 2).apply([](const float val) { return val > 0.5 ? 1.0 : 0.0; });

    return std::make_pair(X, y);
}

int main() {
    try {
        // Constants for the model
        constexpr int NUM_SAMPLES = 128;
        constexpr int VOCAB_SIZE = 10000;
        constexpr int SEQ_LENGTH = 10;
        constexpr int EMBEDDING_DIM = 16;
        constexpr float LEARNING_RATE = 0.01f;
        constexpr float DECAY_RATE = 0.8f;
        constexpr int NUM_EPOCHS = 10;

        // Generate synthetic data
        auto [train_data, train_labels] = generate_synthetic_data(NUM_SAMPLES, VOCAB_SIZE, SEQ_LENGTH);

        // Initialize the loss function
        LossFunction<float>::binaryCrossEntropyLoss loss_function;

        // Initialize learning rate scheduler (ExponentialDecaySchedule)
        Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_scheduler(LEARNING_RATE, DECAY_RATE);

        // Initialize the EmbeddingModel with the correct arguments
        EmbeddingModel<float> model(VOCAB_SIZE, EMBEDDING_DIM, learning_rate_scheduler, nullptr);

        // Collect model parameters for optimization
        std::vector<std::vector<int>> params_shape;
        for (const auto& ref : model.parameters()) {
            auto param_shape = ref.get().shape();
            params_shape.push_back(param_shape);
        }

        // Initialize the optimizer
        Optimizer<float>::Adam optimizer(params_shape, LEARNING_RATE, learning_rate_scheduler);

        // Perform multiple epochs of training
        for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
            // Perform forward pass
            Tensor<float> output = model.forward(train_data);

            // Compute loss
            float loss = loss_function.forward(output.reshape(static_cast<int>(output.data.size())), train_labels);

            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;

            // Perform backward pass
            Tensor<float> grad = loss_function.backward(output, train_labels);
            grad = grad.reshape(output.shape());
            model.backward(grad);

            // Update model parameters
            optimizer.update(model.parameters(), model.gradients(), epoch);
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
