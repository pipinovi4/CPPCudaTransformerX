#include "../models/DigitRecognizer.h"
#include "../include/Optimizer.h"
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include "../utils/loadMNIST.tpp"
#include <vector>
#include <iostream>

/**
 * \brief Runs the DigitRecognizer application.
 *
 * This function loads the MNIST dataset, converts the images to tensors, and initializes
 * the DigitRecognizer model with the specified parameters. It also handles any exceptions
 * that may occur during the process.
 *
 * \return int Returns 0 on successful execution, or a non-zero value if an error occurs.
 */
int main() {
    try {
        // CONSTATNTS
        constexpr float LEARNING_RATE = 0.001;
        constexpr float DECAY_RATE = 0.6;
        constexpr int NUM_EPOCHS = 5;
        constexpr int BATCH_SIZE = 64;

        // Load train data of the MNIST
        const std::vector<std::vector<std::uint8_t>> train_images = loadMNISTImages("../data/mnist/train-images-idx3-ubyte.txt");
        const std::vector<std::uint8_t> train_targets = loadMNISTLabels("../data/mnist/train-labels-idx1-ubyte.txt");

        // Load test data of the MNIST
        const std::vector<std::vector<std::uint8_t>> test_images = loadMNISTImages("../data/mnist/t10k-images-idx3-ubyte.txt");
        const std::vector<std::uint8_t> test_targets = loadMNISTLabels("../data/mnist/t10k-labels-idx1-ubyte.txt");

        // Convert normalized train data to floated Tensor with extra dimension for channel
        Tensor<float> train_data = Tensor<float>(train_images).reshape({60000, 784});
        auto train_labels = Tensor<float>(train_targets);

        // Convert normalized test data to floated Tensor with extra dimension for channel
        auto test_data = Tensor<float>(test_images).reshape({10000, 784});
        auto test_labels = Tensor<float>(test_targets);

        // Initialize the model, loss function, and optimizer
        auto loss_function = new LossFunction<float>::crossEntropyLoss();
        DigitRecognizer<float> model(784, 128, 10, loss_function);
        Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_scheduler(LEARNING_RATE, DECAY_RATE);

        // Initialize the optimizer and computate shape of the model parameters for initializing biases and wights
        std::vector<std::vector<int>> params_shape;
        for (const auto& ref : model.parameters()) {
            auto param_shape = ref.get().shape();
            params_shape.push_back(param_shape);
        }
        const auto optimizer = new Optimizer<float>::Adam(model.parameters(), LEARNING_RATE, learning_rate_scheduler);

        // Train the model
        model.train(train_data, train_labels, NUM_EPOCHS, optimizer, BATCH_SIZE);

        // Test the model
        int correct_predictions = 0;
        for (int i = 0; i < test_data.shape()[0]; ++i) {
            // Assuming test_data[{i}] returns a tensor representing the i-th sample
            Tensor<float> output = model.forward(test_data[{i}]);

            // Find the index of the maximum value in the output tensor and check if it's correct
            if (auto predicted_label = std::distance(output.data.begin(), std::max_element(output.data.begin(), output.data.end()));
                predicted_label == static_cast<int>(test_labels.data[i])) {
                correct_predictions++;
            }
        }

        // Calculate accuracy
        float accuracy = static_cast<float>(correct_predictions) / static_cast<float>(test_data.shape()[0]);
        std::cout << "Accuracy: " << accuracy * 100.0f << "%" << std::endl;

        // Cleanup
        delete loss_function;
        delete optimizer;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
