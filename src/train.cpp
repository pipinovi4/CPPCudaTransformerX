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
#include <chrono>

int train() {
    // Load the Wikitext train data
    std::cout << "Loading the WikiText dataset..." << std::endl;
    const std::vector<std::vector<std::string>> train_data = loadWikiText("data/wikitext")[0];
    std::cout << "WikiText dataset loaded successfully!" << std::endl;

    // Load the vocabulary
    std::cout << "Loading the vocabulary..." << std::endl;
    const std::vector<std::string> vocab = load_vocab("data/vocab/vocab_30000_words.txt");
    std::cout << "Vocabulary loaded successfully!" << std::endl;

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
    std::cout << "Defining the Transformer model..." << std::endl;
    Transformer<float> transformer(vocab_size, d_model, num_heads, d_ff, max_tokens, decay_rate, decay_rate, 2, learning_rate_schedule, &loss_fn, &optimizer, vocab);

    transformer.save_weights("data/weights/transformer_weights.txt");

    // Train the model
    std::cout << "Training the model..." << std::endl;
    transformer.train(train_data, 10, 128);

    // Save the weights
    transformer.save_weights("data/weights/transformer_weights.txt");

    // Print success message
    std::cout << "Model trained and saved successfully!" << std::endl;
    return 0;
}
