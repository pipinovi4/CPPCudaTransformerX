#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <regex>
#include <string>
#include "../include/Tensor.h"
#include "../include/DenseLayer.h"
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "ActivationFunction.tpp"
#include "../include/Embedding.h"
#include "../examples/EmbeddingModel.h"
#include "../include/MultiHeadAttention.h"

int main() {
    try {
        // Load the datasets
        const std::vector<std::vector<std::vector<std::string>>> dataset = EmbeddingModel<float>::loadDataset();

        // Separate the train and test data
        const std::vector<std::vector<std::string>>& train_data = dataset[0];
        const std::vector<std::vector<std::string>>& test_data = dataset[1];

        // Example usage: Print the first 5 lines of the training data
        std::cout << "\nFirst 5 lines of the training data:\n";
        for (size_t i = 0; i < 5 && i < train_data.size(); ++i) {
            for (const auto& word : train_data[i]) {
                std::cout << word << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\n\nFirst 5 lines of the testing data:\n";
        for (size_t i = 0; i < 5 && i < test_data.size(); ++i) {
            for (const auto& word : test_data[i]) {
                std::cout << word << " ";
            }
            std::cout << std::endl;
        }

        // Similarly, you can access and work with test_data
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
