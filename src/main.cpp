#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "../models/Transformer.h"
#include "../utils/loadVocab.h"

int main() {
    int initial_token_count = 0;
    std::string input_text;
    std::vector<std::string> tokens;

    std::cout << "Enter context for text generation (max 4 tokens): ";
    std::getline(std::cin, input_text);

    // Split input into individual tokens
    std::istringstream stream(input_text);
    std::string word;
    while (stream >> word) {
        tokens.push_back(word);
    }

    initial_token_count = tokens.size();

    // Adjust token size if greater than 4
    if (tokens.size() > 4) {
        std::cout << "Maximum context size is 4 tokens." << std::endl;
        tokens.resize(4);
        input_text = "";
        for (const auto& token : tokens) {
            input_text += token + " ";
        }

        std::cout << "Reduced to 4 tokens: " << input_text << std::endl;
        std::cout << "Continue? (yes/no): ";

        std::string response;
        std::getline(std::cin, response);
        std::transform(response.begin(), response.end(), response.begin(), ::tolower);

        if (response == "no" || response == "n") {
            std::cout << "Exiting..." << std::endl;
            return 0;
        }
    } else if (tokens.size() < 4) {
        std::cout << "Context has fewer than 4 tokens." << std::endl;
        int missing_tokens = 4 - tokens.size();
        tokens.insert(tokens.end(), missing_tokens, "<PAD>");
        input_text = "";
        for (const auto& token : tokens) {
            input_text += token + " ";
        }

        std::cout << "Filled with <PAD>: " << input_text << std::endl;
        std::cout << "Continue? (yes/no): ";

        std::string response;
        std::getline(std::cin, response);
        std::transform(response.begin(), response.end(), response.begin(), ::tolower);
        if (response == "no" || response == "n") {
            std::cout << "Exiting..." << std::endl;
            return 0;
        }
    }

    // Load vocabulary
    std::cout << "Loading vocabulary..." << std::endl;
    const std::vector<std::string> vocab = load_vocab("../data/vocab/vocab_30000_words.txt");
    std::cout << "Vocabulary loaded." << std::endl;

    // Model parameters
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

    // Initialize learning rate scheduler, loss function, and optimizer
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule lr_schedule(learning_rate, decay_rate);
    LossFunction<float>::crossEntropyLoss loss_fn;
    Optimizer<float>::Adam optimizer(learning_rate, lr_schedule, weight_decay, b1, b2, epsilon);

    // Initialize and load the Transformer model
    Transformer<float> transformer(&loss_fn, &optimizer, vocab, lr_schedule, vocab_size, d_model, n_heads, d_ff, max_tokens, dropout, label_smoothing);
    transformer.load_weights(weights_path);

    std::cout << "Do you want to see the model's parameters? (yes/no): ";
    std::string response;
    std::getline(std::cin, response);
    std::transform(response.begin(), response.end(), response.begin(), ::tolower);

    if (response == "yes" || response == "y") {
        std::cout << "\nModel Parameters:\n";
        std::cout << "Vocab size: " << vocab_size << "\n"
                  << "Max tokens: " << max_tokens << "\n"
                  << "Learning rate: " << learning_rate << "\n"
                  << "d_model: " << d_model << "\n"
                  << "n_heads: " << n_heads << "\n"
                  << "d_ff: " << d_ff << "\n"
                  << "Weight decay: " << weight_decay << "\n"
                  << "b1: " << b1 << "\n"
                  << "b2: " << b2 << "\n"
                  << "Epsilon: " << epsilon << "\n"
                  << "Dropout: " << dropout << "\n"
                  << "Label smoothing: " << label_smoothing << "\n";
    }

    // Generate text based on input
    const std::vector<std::vector<std::string>> generated_text = transformer.generate({transformer.positional_encoder_->tokenize(input_text)}, initial_token_count);
    std::cout << "Text generated successfully!" << std::endl;

    // Output the generated text
    std::cout << "\nGenerated text: ";
    for (int i = initial_token_count + 1; i < generated_text[0].size(); ++i) {
        std::cout << generated_text[0][i] << " ";
    }

    std::cout << "\nFull generated text: ";
    for (const auto& sentence : generated_text) {
        for (const auto& word : sentence) {
            std::cout << word << " ";
            if (word == "<EOS>") break;
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Goodbye!" << std::endl;
    return 0;
}
