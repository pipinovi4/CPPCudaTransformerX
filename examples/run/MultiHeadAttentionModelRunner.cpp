#include <iostream>
#include "../MultiHeadAttentionModel.h"
#include "../../include/Embedding.h"
#include "../utils/loadAGNews.tpp"
#include "../../include/Tensor.h"
#include "../../include/Optimizer.h"
#include "../../include/LossFunction.h"
#include "../../include/Tokenizer.h"

/**
 * @brief Main function to train and test a Multi-Head Attention model on the AG News dataset.
 *
 * This program initializes and trains a Multi-Head Attention model on the AG News dataset using
 * a sequence-to-sequence approach. It includes data preprocessing steps such as tokenization and
 * sequence padding/truncation. The program also tests the trained model on the test dataset and
 * evaluates its performance by computing the loss for each test sample.
 *
 * The training and testing process is managed in a loop where:
 * - Data is loaded and preprocessed into input and target pairs.
 * - The model is trained over several epochs using an optimizer and loss function.
 * - The trained model is then tested on unseen data to compute its generalization performance.
 *
 * Key components:
 * - Tokenization and text-to-ID conversion.
 * - Padding and truncation of sequences to a fixed length.
 * - Training loop with forward and backward passes.
 * - Testing loop to evaluate model performance on test data.
 */
int main() {
    // Constants for the test
    constexpr int MAX_SEQUENCE_LENGTH = 32;
    constexpr int HIDDEN_DIM = 64;
    constexpr int NUM_HEADS = 8;
    constexpr int HEAD_DIM = HIDDEN_DIM / NUM_HEADS;
    constexpr float LEARNING_RATE = 0.001;
    constexpr float DECAY_RATE = 0.6;
    constexpr int NUM_EPOCHS = 5;
    constexpr int BATCH_SIZE = 8; // Define the batch size

    // Define the learning rate scheduler
    Optimizer<float>::LearningRateSchedule::StepDecaySchedule learning_rate_scheduler(LEARNING_RATE, DECAY_RATE, 1);

    // Define the Multi-Head Attention model
    MultiHeadAttentionModel<float> model(MAX_SEQUENCE_LENGTH, NUM_HEADS, HEAD_DIM);

    // Collect model parameters for optimization
    std::vector<std::vector<int>> params_shape;
    for (const auto& ref : model.parameters()) {
        auto param_shape = ref.get().shape();
        params_shape.push_back(param_shape);
    }

    // Define the optimizer
    Optimizer<float>::Adam optimizer(params_shape, 0.9, learning_rate_scheduler, 1e-8);

    // Define large dummy data for training
    std::vector<Tensor<float>> input_data(1000);
    std::vector<Tensor<float>> target_data(1000);

    // Initialize input and target data with random values
    for (int i = 0; i < 1000; ++i) {
        input_data[i] = Tensor<float>::uniform({BATCH_SIZE, MAX_SEQUENCE_LENGTH, HIDDEN_DIM}, 0, 1);
        target_data[i] = Tensor<float>::uniform({BATCH_SIZE, MAX_SEQUENCE_LENGTH, HEAD_DIM}, 0, 1);
    }

    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float loss = 0;
        for (int i = 0; i < 1000; ++i) {
            // Perform forward pass
            Tensor<float> output_train = model.forward(input_data[i]);

            // Compute loss
            LossFunction<float>::crossEntropyLoss loss_function;  // Ensure your loss function is appropriate for your task
            loss += loss_function.forward(output_train, target_data[i]);

            // Perform backward pass
            Tensor<float> grad = loss_function.backward(output_train, target_data[i]);
            model.backward(grad);

            optimizer.update(model.parameters(), model.gradients(), epoch);
        }
        std::cout << "Epoch [" << epoch + 1 << "], Loss: " << loss / 1000 << std::endl;
    }

    // Test the model
    float loss = 0;
    for (int i = 0; i < 1000; ++i) {
        // Perform forward pass
        Tensor<float> output_test = model.forward(input_data[i]);

        // Compute loss
        LossFunction<float>::crossEntropyLoss loss_function; // Ensure your loss function is appropriate for your task
        loss += loss_function.forward(output_test, target_data[i]);
    }
    std::cout << "Test Loss: " << loss / 1000 << std::endl;

    return 0;
}

