#include <iostream>
#include <vector>
#include <cassert>
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "../include/DenseLayer.h"


int main() {
    try {
        // Define input and output sizes
        constexpr int input_units = 64;
        constexpr int output_units = 64;
        constexpr size_t num_samples = 2;
        constexpr size_t num_epochs = 10000;

        // Create activation function and optimizer
        ActivationFunction<float>::LeakyReLU activation_function;
        LossFunction<float>::meanSquaredError loss_function;
        Optimizer<float>::SGD optimizer(0.01, *new Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule(0.01, 0.95));

        // Create DenseLayer instance
        DenseLayer<float> dense_layer(input_units, output_units, &activation_function);

        // Generate uniform data and labels
        const Tensor<float> input_data = Tensor<float>::uniform({num_samples, input_units}, 0.0f, 1.0f);
        const Tensor<float> labels = input_data * std::sqrt(2.0f) + 1.0f;

        // Training loop
        for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
            // Forward pass
            Tensor<float> output_data = dense_layer.forward(input_data);

            // Compute loss (assuming Mean Squared Error)
            const float loss = loss_function.forward(output_data, labels);

            // Backward pass
            Tensor<float> grad_output = loss_function.backward(output_data, labels);
            Tensor<float> grad_input = dense_layer.backward(grad_output);

            // Update parameters
            dense_layer.updateParameters(&optimizer, epoch);

            // Print loss every 100 epochs
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            }
        }

        // Print labels
        std::cout << "Labels: ";
        labels.print();

        Tensor<float> preds = dense_layer.forward(input_data);

        std::cout << "Preds: ";
        preds.print();

        std::cout << "Loss preds: ";
        std::cout << loss_function.forward(preds, labels) << std::endl;

        std::cout << "Training completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
    }

    return 0;
}