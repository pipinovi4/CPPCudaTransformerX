#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#pragma once
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include "../include/Optimizer.h"
#include "../include/Embedding.h"
#include "../include/MultiHeadAttention.h"
#include "../include/ResidualBlock.h"
#include "../include/Tokenizer.h"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include "../include/DenseLayer.h"

/**
 * @brief Transformer model class template for sequence-to-sequence tasks.
 *
 * @tparam T Data type (e.g., float or double) for tensor calculations.
 */
template <typename T>
class Transformer {
public:
    /**
     * @brief Constructs a Transformer model.
     *
     * @param loss_function Pointer to the loss function.
     * @param optimizer Pointer to the optimizer.
     * @param vocab Vocabulary for the model.
     * @param learning_rate_schedule Reference to the learning rate scheduler.
     * @param vocab_size Vocabulary size.
     * @param d_model Dimensionality of the model.
     * @param n_heads Number of attention heads.
     * @param d_ff Dimensionality of the feed-forward network.
     * @param max_len Maximum sequence length.
     * @param dropout Dropout rate.
     * @param label_smoothing Label smoothing rate for training.
     */
    Transformer(LossFunction<T>* loss_function, Optimizer<T>* optimizer, std::vector<std::string> vocab,
        typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule, int vocab_size,
        int d_model, int n_heads, int d_ff, int max_len, float dropout, float label_smoothing);

    /**
     * @brief Defaulted Destructor.
     */
    ~Transformer() = default;

    /**
     * @brief Move Constructor.
     */
    Transformer(Transformer&& other) noexcept = default;

    /**
     * @brief Move Assignment Operator.
     */
    Transformer& operator=(Transformer&& other) noexcept = default;

    // Deleted Copy Constructor and Assignment Operator to prevent copying
    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

    /**
     * @brief Forward pass through the model.
     *
     * @param src Source tensor input.
     * @param tgt Target tensor input.
     * @return Tensor<T> Output tensor after passing through the model.
     */
    Tensor<T> forward(const Tensor<T>& src, const Tensor<T>& tgt);

    /**
     * @brief Backward pass to compute gradients.
     *
     * @param grad Gradient tensor to backpropagate.
     */
    void backward(Tensor<T>& grad);

    /**
     * @brief Updates the model parameters based on gradients.
     *
     * @param epoch Current epoch number, used for adjusting learning rate.
     */
    void update(int epoch);

    /**
     * @brief Load model weights from a file.
     *
     * @param filepath Path to the file containing weights.
     */
    void load_weights(const std::string& filepath);

    /**
     * @brief Save model weights to a file.
     *
     * @param filepath Path to the file to save weights.
     */
    void save_weights(const std::string& filepath);

    /**
     * @brief Train the model on the provided data.
     *
     * @param data Training data.
     * @param n_epochs Number of training epochs.
     * @param batch_size Size of each training batch (default is 32).
     */
    void train(const std::vector<std::vector<std::string>>& data, int n_epochs, int batch_size = 32);

    /**
     * @brief Generate text using the model.
     *
     * @param input Input context for generation.
     * @param context_tokens_size Number of tokens to use from the input context.
     * @return std::vector<std::vector<std::string>> Generated text sequences.
     */
    std::vector<std::vector<std::string>> generate(const std::vector<std::vector<std::string>>& input, int context_tokens_size);

    /**
     * @brief Retrieve all model parameters (weights and biases).
     *
     * @return std::vector<std::reference_wrapper<Tensor<T>>> Vector of references to the model parameters.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters();

    /**
     * @brief Retrieve all model gradients.
     *
     * @return std::vector<std::reference_wrapper<Tensor<T>>> Vector of references to the model gradients.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

    /**
     * @brief Get the shapes of all model parameters.
     *
     * @return std::vector<std::vector<int>> Vector containing shapes of model parameters.
     */
    std::vector<std::vector<int>> parameters_shape();

    // Public attribute for the positional encoder
    std::unique_ptr<Tokenizer<T>> positional_encoder_; // Positional encoder

private:
    // Parameters of the model
    int vocab_size_;    ///< Size of the vocabulary
    int d_model_;       ///< Dimensionality of the model
    int n_heads_;       ///< Number of attention heads
    int d_ff_;          ///< Dimensionality of feed-forward network
    int max_len_;       ///< Maximum sequence length
    float dropout_;     ///< Dropout rate for regularization
    float label_smoothing_; ///< Label smoothing rate

    // Learning rate schedule
    typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule_; ///< Reference to learning rate schedule

    // Loss function and optimizer
    LossFunction<T>* loss_function_; ///< Pointer to the loss function
    Optimizer<T>* optimizer_;        ///< Pointer to the optimizer

    // Embedding layer
    std::unique_ptr<Embedding<T>> embedding_; ///< Embedding layer for input tokens

    // Encoder and decoder layers
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> encoder_layers_; ///< Encoder layers for processing input
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> decoder_layers_; ///< Decoder layers for generating output
    std::vector<std::unique_ptr<ResidualBlock<T, Layer<T>*>>> output_encoder_layers_; ///< Output encoder layers for masked multi-head attention

    // Output layer
    std::unique_ptr<DenseLayer<T>> output_layer_softmax_; ///< Final softmax layer for generating probabilities

    /**
     * @brief Sets all gradients to zero.
     */
    void zero_grad();

    /**
     * @brief Converts a sequence of strings to a tensor of tokenized data.
     *
     * @param data Input data as a vector of sentences (string vectors).
     * @return std::vector<std::vector<Tensor<T>>> Tokenized tensor representation of the data.
     */
    std::vector<std::vector<Tensor<T>>> convert_to_tensor(const std::vector<std::vector<std::string>>& data);

    /**
     * @brief Converts float tokens to integer tokens.
     *
     * @param tokens Tensor containing float tokens.
     * @return std::vector<int> Vector containing integer tokens.
     */
    std::vector<int> convert_to_int_tokens(const Tensor<T>& tokens);
};

#include "Transformer.tpp"

#endif //TRANSFORMER_H
