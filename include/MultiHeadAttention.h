#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#pragma once
#include "Tensor.h"
#include "ActivationFunction.h"
#include "Layer.h"
#include <vector>

/**
 * @brief MultiHeadAttention layer class.
 *
 * This class implements the multi-head attention mechanism, a key component in the Transformer model architecture.
 * It allows the model to focus on different parts of the input sequence simultaneously, capturing various types of relationships.
 *
 * @tparam T Data type for tensor elements (e.g., float, double).
 */
template <typename T>
class MultiHeadAttention final : public Layer<T> {
public:
    /**
     * @brief Constructor for MultiHeadAttention.
     *
     * @param hidden_dim Total hidden dimension size.
     * @param num_heads Number of attention heads.
     * @param head_dim Dimension size for each attention head.
     * @param activation Pointer to an activation function. Defaults to a linear activation.
     */
    MultiHeadAttention(const int& hidden_dim, const int& num_heads, const int& head_dim, ActivationFunction<T>* activation = new typename ActivationFunction<T>::Linear());

    /**
     * @brief Destructor for MultiHeadAttention.
     */
    ~MultiHeadAttention() override = default;

    /**
     * @brief Forward pass for the MultiHeadAttention layer.
     *
     * This method processes the input tensor through the multi-head attention mechanism.
     * Optionally, a mask can be provided to ignore certain positions (useful for preventing attention to future positions in a sequence).
     *
     * @param input Input tensor of shape (batch_size, seq_length, hidden_dim).
     * @param mask Optional tensor to mask certain positions (shape must be compatible with input tensor).
     * @return Tensor<T> Output tensor after applying multi-head attention.
     */
    Tensor<T> forward(const Tensor<T>& input, const Tensor<T>* mask = nullptr);

    /**
     * @brief Overloaded forward method without a mask.
     *
     * This version of the forward method is required to override the pure virtual method in the Layer base class.
     *
     * @param input Input tensor of shape (batch_size, seq_length, hidden_dim).
     * @return Tensor<T> Output tensor after applying multi-head attention.
     */
    Tensor<T> forward(const Tensor<T>& input) override;

    /**
     * @brief Backward pass for MultiHeadAttention layer.
     *
     * Computes the gradients of the loss with respect to the input and the layer's parameters.
     *
     * @param grad Gradient of the loss with respect to the output of this layer.
     */
    void backward(Tensor<T>& grad) override;

    /**
     * @brief Split the input tensor into multiple heads.
     *
     * This method reshapes and splits the input tensor into multiple heads for parallel attention computation.
     *
     * @param x Input tensor of shape (batch_size, seq_length, hidden_dim).
     * @return std::vector<Tensor<T>> Vector of tensors, each corresponding to one attention head.
     */
    std::vector<Tensor<T>> split_heads(const Tensor<T>& x) const;

    /**
     * @brief Concatenate the output from multiple heads.
     *
     * After processing the input through each attention head, this method concatenates the results back into a single tensor.
     *
     * @param heads Vector of tensors, each corresponding to one attention head.
     * @return Tensor<T> Concatenated tensor of shape (batch_size, seq_length, hidden_dim).
     */
    Tensor<T> concat_heads(const std::vector<Tensor<T>>& heads) const;

    /**
    * @brief Get the parameters of the MultiHeadAttention layer.
    *
    * @return std::vector<std::reference_wrapper<Tensor<T>>> Vector of references to the layer's parameters.
    */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters() override;

    /**
     * @brief Get the gradients of the MultiHeadAttention layer.
     *
     * @return std::vector<std::reference_wrapper<Tensor<T>>> Vector of references to the gradients of the layer's parameters.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> gradients() override;

private:
    int hidden_dim;  ///< Total hidden dimension size.
    int num_heads;   ///< Number of attention heads.
    int head_dim;    ///< Dimension size for each attention head.

    // Tensors for queries, keys, values, and output projections
    Tensor<T> W_q, W_k, W_v, W_o;      ///< Weights for queries, keys, values, and output projection.
    Tensor<T> b_q, b_k, b_v, b_o;      ///< Biases for queries, keys, values, and output projection.
    Tensor<T> grad_W_q, grad_W_k, grad_W_v, grad_W_o; ///< Gradients for weights.
    Tensor<T> grad_b_q, grad_b_k, grad_b_v, grad_b_o; ///< Gradients for biases.

    // Tensors for attention heads
    std::vector<Tensor<T>> queries_heads; ///< Query projections split into heads.
    std::vector<Tensor<T>> keys_heads;    ///< Key projections split into heads.
    std::vector<Tensor<T>> values_heads;  ///< Value projections split into heads.

    std::vector<Tensor<T>> attention_heads; ///< Outputs of each attention head.

    Tensor<T> input_cache; ///< Cache of the input tensor for backpropagation.

    ActivationFunction<T>* activation; ///< Activation function applied to the output.

    /**
     * @brief Initialize the layer's parameters.
     *
     * This method initializes the weights and biases for the layer using a suitable initialization strategy.
     *
     * @param weights Tensor to be initialized.
     */
    static void initializeParameter(Tensor<T>& weights);
};

#include "../src/MultiHeadAttention.tpp"

#endif // MULTIHEADATTENTION_H
