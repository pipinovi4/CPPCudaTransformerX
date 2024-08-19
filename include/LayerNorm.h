#ifndef LAYERNORM_H
#define LAYERNORM_H

#pragma once
#include "Layer.h"
#include "Tensor.h"

/**
 * @class LayerNorm
 * @brief Implements Layer Normalization for neural networks.
 *
 * Layer normalization normalizes the input tensor across the features (last dimension)
 * to stabilize and accelerate training. This class supports both forward and backward
 * passes, making it suitable for training deep learning models.
 *
 * @tparam T The data type used for the computations, typically `float` or `double`.
 */
template <typename T>
class LayerNorm final : public Layer<T> {
public:
    /**
     * @brief Constructor for the LayerNorm layer.
     *
     * Initializes the LayerNorm layer with the specified model dimension and
     * a small epsilon value to prevent division by zero during normalization.
     *
     * @param d_model The dimension of the model, which corresponds to the size
     *                of the last dimension of the input tensor.
     * @param epsilon A small value added to the variance to prevent division
     *                by zero during normalization. Default is 1e-6.
     */
    explicit LayerNorm(int d_model, float epsilon = 1e-6);

    /**
     * @brief Forward pass of the LayerNorm layer.
     *
     * Applies layer normalization to the input tensor. The normalization
     * is performed across the last dimension of the input tensor, and
     * the result is scaled and shifted using learnable parameters `gamma_` and `beta_`.
     *
     * @param input The input tensor to be normalized.
     * @return The normalized output tensor after applying the scale and shift.
     */
    Tensor<T> forward(const Tensor<T>& input) override;

    /**
     * @brief Backward pass of the LayerNorm layer.
     *
     * Computes the gradients of the loss with respect to the input tensor,
     * which are used to update the parameters during backpropagation.
     *
     * @param grad The gradient of the loss with respect to the output of the layer.
     */
    void backward(Tensor<T>& grad) override;

    /**
     * @brief Get the parameters of the layer.
     *
     * This method returns a vector of references to the layer's parameters.
     * This allows the optimizer to update the parameters during training.
     *
     * @return A vector of references to the layer's parameters.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters() override {
        // Return empty vector since the parameters doesn't exist
        return {};
    };

    /**
     * @brief Get the gradients of the layer.
     *
     * This method returns a vector of references to the layer's gradients.
     * This allows the optimizer to update the parameters during training.
     *
     * @return A vector of references to the layer's gradients.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> gradients() override {
        // Return empty vector since the gradients doesn't exist
        return {};
    }

private:
    int d_model_;  // Dimension of the model (size of the last dimension of the input tensor)
    float epsilon_;  // Small constant to prevent division by zero during normalization
    Tensor<T> gamma_;  // Learnable scale parameter (applied after normalization)
    Tensor<T> beta_;  // Learnable shift parameter (applied after normalization)

    // Intermediate results needed for backpropagation
    Tensor<T> normalized_;  // Normalized input tensor
    Tensor<T> mean_;  // Mean of the input tensor across the last dimension
    Tensor<T> variance_;  // Variance of the input tensor across the last dimension

    Tensor<T> input_;  // Stores the input tensor for use during backpropagation
};

#include "../src/LayerNorm.tpp"

#endif //LAYERNORM_H
