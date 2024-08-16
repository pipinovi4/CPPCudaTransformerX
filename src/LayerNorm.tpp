#ifndef LAYERNORM_TPP
#define LAYERNORM_TPP

#include "../include/LayerNorm.h"

/**
 * @brief Construct a new LayerNorm object.
 * 
 * @tparam T Data type of the tensor elements (e.g., float, double).
 * @param d_model The dimension of the model (size of the last dimension).
 * @param epsilon A small value added to the variance to prevent division by zero during normalization.
 */
template <typename T>
LayerNorm<T>::LayerNorm(const int d_model, const float epsilon) 
    : d_model_(d_model), epsilon_(epsilon), gamma_(Tensor<T>::ones({d_model})), beta_(Tensor<T>::zeros({d_model})) {}

/**
 * @brief Forward pass of the LayerNorm layer.
 * 
 * This function applies layer normalization to the input tensor `x`. 
 * Layer normalization normalizes the input across the last dimension by subtracting the mean 
 * and dividing by the standard deviation. After normalization, it applies a learned scale (`gamma_`)
 * and shift (`beta_`).
 * 
 * @tparam T Data type of the tensor elements (e.g., float, double).
 * @param x Input tensor with shape (batch_size, seq_length, d_model).
 * @return Tensor<T> The normalized and scaled tensor with the same shape as the input.
 */
template <typename T>
Tensor<T> LayerNorm<T>::forward(const Tensor<T>& x) {
    // Calculate mean across the last dimension and expand the dimensions to match the input tensor
    Tensor<T> mean = x.mean(-1).expandDimsAs(x.shape());

    // Calculate variance across the last dimension
    Tensor<T> variance = ((x - mean) * (x - mean)).mean(-1);

    // Normalize the input
    Tensor<T> normalized = (x - mean) / (variance.expandDimsAs(x.shape()) + epsilon_).sqrt();

    // Scale and shift using learned parameters gamma and beta
    return gamma_.expandDimsAs(x.shape()) * normalized + beta_.expandDimsAs(x.shape());
}

#endif //LAYERNORM_TPP
