#include <iostream>
#include <vector>
#include <string>
#include "../models/Transformer.h"
#include "../utils/loadWikiText.tpp"
#include "../utils/loadVocab.h"
#include <filesystem>

int main() {
    // Load the Wikitext train data
    std::cout << "Loading the WikiText dataset..." << std::endl;
    const std::vector<std::vector<std::string>> train_data = loadWikiText("../data/wikitext")[0];
    std::cout << "WikiText dataset loaded successfully!" << std::endl;

    // Load the vocabulary
    std::cout << "Loading the vocabulary..." << std::endl;
    const std::vector<std::string> vocab = load_vocab("../data/vocab/vocab_30000_words.txt");
    std::cout << "Vocabulary loaded successfully!" << std::endl;

    // Parameters
    constexpr int max_tokens = 16;
    constexpr int d_model = 128;
    constexpr int n_heads = 8;
    constexpr int d_ff = 512;
    const int vocab_size = static_cast<int>(vocab.size());
    constexpr float learning_rate = 0.001;
    constexpr float decay_rate = 0.9;
    constexpr float weight_decay = 0.001;
    constexpr float b1 = 0.9;
    constexpr float b2 = 0.999;
    constexpr float epsilon = 1e-8;
    constexpr float label_smoothing = 0.1;
    constexpr float dropout = 0.1;
    constexpr float num_epochs = 1;
    constexpr float batch_size = 128;
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
    if (std::filesystem::exists(weights_path)) {
        std::cout << "Pretrained weights is exist, loading the weights..." << std::endl;
        transformer.load_weights("../data/weights/transformer_weights.txt");
    } else {
        std::cout << "Pretrained weights is not exist, was initialized initial weights..." << std::endl;
    }

    // Train the model
    std::cout << "Training the model..." << std::endl;
    transformer.train(train_data, num_epochs, batch_size);

    // Save the weights
    transformer.save_weights("../data/weights/transformer_weights.txt");

    // Print success message
    std::cout << "Model trained and saved successfully!" << std::endl;
    return 0;
}
