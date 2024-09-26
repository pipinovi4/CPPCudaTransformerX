#include <iostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <utility>
#include <functional>
#include "../include/Tokenizer.h"
#include "../utils/loadAGNews.tpp"
#include <deque>
#include "../include/Embedding.h"
#include "../include/Tensor.h"
#include "../models/Transformer.h"
#include "../utils/loadWikiText.tpp"
#include "../utils/loadVocab.h"
#include <chrono>  // Include the chrono library

int main() {
    // Load the Wikitext test data
    std::cout << "Loading the WikiText dataset..." << std::endl;
    const std::vector<std::vector<std::string>> test_data = loadWikiText("../data/wikitext")[2];
    std::cout << "WikiText dataset loaded successfully!" << std::endl;

    // Load the vocabulary
    std::cout << "Loading the vocabulary..." << std::endl;
    const std::vector<std::string> vocab = load_vocab("../data/vocab/vocab_30000_words.txt");
    std::cout << "Vocabulary loaded successfully!" << std::endl;

    // Parameters
    constexpr int max_tokens = 8;
    constexpr int d_model = 32;
    constexpr int n_heads = 8;
    constexpr int d_ff = 128;
    const int vocab_size = static_cast<int>(vocab.size());
    constexpr float learning_rate = 0.001;
    constexpr float decay_rate = 0.9;
    constexpr float weight_decay = 0.001;
    constexpr float b1 = 0.9;
    constexpr float b2 = 0.999;
    constexpr float epsilon = 1e-8;
    constexpr float label_smoothing = 0.1;
    constexpr float dropout = 0.1;
    const std::string weights_path = "../data/weights/transformer_weights.txt";

    // Define learning rate scheduler
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_schedule(learning_rate, decay_rate);

    // Define loss function
    LossFunction<float>::crossEntropyLoss loss_fn;

    // Define optimizer
    Optimizer<float>::Adam optimizer(learning_rate, learning_rate_schedule, weight_decay, b1, b2, epsilon);

    // Define the Transformer model
    std::cout << "Defining the Transformer model..." << std::endl;
    Transformer<float> transformer(&loss_fn, &optimizer, vocab, learning_rate_schedule, vocab_size, d_model, n_heads, d_ff, max_tokens, dropout, label_smoothing);

    // Load the weights
    transformer.load_weights("../data/weights/transformer_weights.txt");

    // Print test data
    std::cout << "Test data:" << std::endl;
    for (auto& sentence : test_data) {
        std::cout << "Sentence: " << std::endl;
        for (auto& word : sentence) {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }

    // Generate text
    std::cout << "Generating text..." << std::endl;
    const std::vector<std::vector<std::string>>  generated_text = transformer.generate(test_data);
    for (auto& sentence : generated_text) {
        for (auto& word : sentence) {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}