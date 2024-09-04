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
    // Load the AG News dataset
    const std::vector<std::vector<std::vector<std::string>>> dataset = loadWikiText("data/wikitext");

    // Get the train data
    std::vector<std::vector<std::string>> train_data = dataset[0];
    train_data.resize(1000);

    // Get the validation data
    const std::vector<std::vector<std::string>> val_data = dataset[1];

    // Get the test data
    const std::vector<std::vector<std::string>> test_data = dataset[2];

    // Load the vocabulary
    const std::vector<std::string> vocab = load_vocab("data/vocab/vocab_30000_words.txt");

    // Parameters
    constexpr int max_tokens = 8;
    constexpr int d_model = 16;
    constexpr int num_heads = 2;
    constexpr int d_ff = 32;
    const int vocab_size = static_cast<int>(vocab.size());
    constexpr float learning_rate = 0.001;
    constexpr float decay_rate = 0.5;

    // Define learning rate scheduler
    Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_schedule(learning_rate, decay_rate);

    // Define loss function
    LossFunction<float>::crossEntropyLoss loss_fn;

    // Define optimizer
    Optimizer<float>::Adam optimizer(learning_rate_schedule);

    // Define the Transformer model
    Transformer<float> transformer(vocab_size, d_model, num_heads, d_ff, max_tokens, decay_rate, decay_rate, 2, learning_rate_schedule, &loss_fn, &optimizer, vocab);

    // Convert to tensor
    auto train_data_tensor = transformer.convert_to_tensor(val_data);

    auto src = train_data_tensor[0];
    auto tgt = train_data_tensor[1];
    auto true_labels = train_data_tensor[2];

    transformer.load_weights("data/transformer_models/transformer_weights.txt");

    // Train the model
    auto start = std::chrono::high_resolution_clock::now();
    transformer.train(train_data, 1, 128);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " s\n";


    // Print parameters
    for (auto param : transformer.parameters()) {
        param.get().print();
    }

    // Predict
    transformer.predict({{"How", "are", "you", "?"}}, 8);

    return 0;
}


// // Activation function relu
// ActivationFunction<float>::ReLU relu;
// // Learning rate scheduler
// Optimizer<float>::LearningRateSchedule::ExponentialDecaySchedule learning_rate_schedule(0.001, 0.5);
// // Embeddding layer
// Embedding<float> embedding(30000, 512, learning_rate_schedule);
//
// // Dense layer
// DenseLayer<float> dense(128, 30000, &relu, 0.0);
//
// // Position-wise dense layer
// PositionalWiseDenseLayer<float> pos_dense(512, 128, relu);
//
// // Multi-head attention
// MultiHeadAttention<float> multi_head_attention(128, 8, 64, &relu);
//
// // Residual block
// ResidualBlock<float, PositionalWiseDenseLayer<float>*> residual_block(128, .95, &pos_dense);
//
// // Forward pass layers for testing in place operations
// Tensor<float> input = Tensor<float>::uniform({128}, 0, 1);
//
// input = embedding.forward(input);
// auto input2 = input;
//
// // // Pass Dense layer
// // // Take a time
// auto start = std::chrono::high_resolution_clock::now();
// pos_dense.forward(input);
// auto end = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double> diff = end - start;
// std::cout << "Time eigen: " << diff.count() << " s\n";
