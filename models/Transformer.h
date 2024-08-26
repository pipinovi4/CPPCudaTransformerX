#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#pragma once
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "../include/Embedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/ResidualBlock.h"
#include "../include/PositionalWiseDenseLayer.h"
#include "../include/Tokenizer.h"
#include <memory>

#include "../include/DenseLayer.h"


template <typename T>
class Transformer {
public:
    // Constructor
    Transformer(int vocab_size, int d_model, int n_heads, int d_ff, int max_len,
        float dropout, float label_smoothing, int warmup_steps,
        typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule,
        LossFunction<T>* loss_function, Optimizer<T>* optimizer, std::vector<std::string> vocab);

    // Defaulted Destructor
    ~Transformer() = default;

    // Move Constructor
    Transformer(Transformer&& other) noexcept = default;

    // Move Assignment Operator
    Transformer& operator=(Transformer&& other) noexcept = default;

    // Deleted Copy Constructor and Assignment Operator
    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    // Forward method to pass data through the model
    Tensor<T> forward(const Tensor<T>& src, const Tensor<T>& tgt);

    // Backward method to compute gradients
    void backward(Tensor<T>& grad);

    // Method to update the weights of the model
    void update(int epoch);

    // Utility functions
    void load_weights(const std::string& filepath);
    void save_weights(const std::string& filepath);

    // Training method
    void train(const std::vector<std::vector<std::string>>& train_data, int n_epochs, int batch_size = 32);

    // Evaluation method
    float evaluate(const std::vector<std::vector<std::string>>& val_data, int batch_size);

    // Prediction method
    Tensor<T> predict(const std::vector<std::vector<std::string>>& src, int max_len);

    // Getters model parameters
    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

    // Getters model parameters shape
    std::vector<std::vector<int>> parameters_shape();

private:
    int vocab_size_;
    int d_model_;
    int n_heads_;
    int d_ff_;
    int max_len_;
    float dropout_;
    float label_smoothing_;
    int warmup_steps_;
    typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule_;
    LossFunction<T>* loss_function_;
    Optimizer<T>* optimizer_;

    // Embedding layers
    std::unique_ptr<Embedding<T>> embedding_; // Embedding layer

    // Positional encoding
    std::unique_ptr<Tokenizer<T>> positional_encoder_; // Positional encoder

    // Encoder and decoder layers
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> encoder_layers_; // Encoder layers
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> decoder_layers_; // Decoder layers
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> output_encoder_layers_; // Output encoder layers

    // Output layer
    std::unique_ptr<DenseLayer<T>> output_layer_softmax_; // Softmax layer

    // Convert sequence to tokenized tensor
    Tensor<T> convert_to_tensor(const std::vector<std::string>& sequence);

    // Label smoothing function
    Tensor<T> apply_label_smoothing(const Tensor<T>& logits, const Tensor<T>& labels);
};

#include "Transformer.tpp"

#endif //TRANSFORMER_H
