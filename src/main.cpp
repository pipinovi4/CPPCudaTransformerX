#include <vector>
#include <iostream>
#include "../include/Tensor.h"
#include "../include/MultiHeadAttention.h"
#include "../include/Optimizer.h"
#include "../include/LossFunction.h"

int main() {
    // Constants for the test
    constexpr int HIDDEN_DIM = 8;
    constexpr int NUM_HEADS = 2;
    constexpr int HEAD_DIM = HIDDEN_DIM / NUM_HEADS;
    constexpr float LEARNING_RATE = 0.001;
    constexpr float DECAY_RATE = 0.6;
    constexpr int NUM_EPOCHS = 20;

    // Create a MultiHeadAttention object
    MultiHeadAttention<float> mha(HIDDEN_DIM, NUM_HEADS, HEAD_DIM);

    // Initialize parameters
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_scheduler(LEARNING_RATE, DECAY_RATE);

    // Initialize the optimizer and compute shape of the model parameters for initializing biases and weights
    std::vector<std::vector<int>> params_shape;
    for (const auto& ref : mha.parameters()) {
        auto param_shape = ref.get().shape();
        params_shape.push_back(param_shape);
    }
    Optimizer<float>::Adam optimizer(params_shape, LEARNING_RATE, learning_rate_scheduler);

    // Initialize the loss function (e.g., Mean Squared Error)
    LossFunction<float>::meanSquaredError loss_function;

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << NUM_EPOCHS << ":\n";

        // Generate some random input data and target data for each epoch
        const Tensor<float> input_data = Tensor<float>::uniform({4, HIDDEN_DIM}, 0, 1);
        const Tensor<float> target_data = Tensor<float>::uniform({4, HIDDEN_DIM}, 0, 1);  // Dummy target data

        std::cout << "Input:\n";
        input_data.print();

        // Perform the forward pass
        const Tensor<float> output = mha.forward(input_data);
        std::cout << "Output:\n";
        output.print();

        // Compute the loss
        const float loss = loss_function.forward(output, target_data);
        std::cout << "Loss: " << loss << std::endl;

        // Perform the backward pass
        Tensor<float> grad_output = loss_function.backward(output, target_data);
        mha.backward(grad_output);
        std::cout << "Backward pass completed.\n";

        // Update the model parameters
        optimizer.update(mha.parameters(), mha.gradients(), epoch);

        std::cout << "Parameters updated.\n";
        std::cout << "--------------------------------------\n";
    }

    // Display predictions after training loop
    std::cout << "Predictions after training:\n";
    const Tensor<float> test_input_data = Tensor<float>::uniform({4, HIDDEN_DIM}, 0, 1);
    test_input_data.print();

    const Tensor<float> predictions = mha.forward(test_input_data);
    predictions.print();

    return 0;
}
