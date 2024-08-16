#ifndef LAYERNORM_TPP
#define LAYERNORM_TPP

#include "../include/LayerNorm.h"

/**
 * @brief Construct a new LayerNorm object.
 * 
 * This constructor initializes the LayerNorm layer with the given model dimension (`d_model`)
 * and a small epsilon value (`epsilon_`) to prevent division by zero during normalization.
 * It also initializes the learnable parameters `gamma_` (scale) and `beta_` (shift).
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
 * and shift (`beta_`). A small uniform noise is added to the final output to reduce symmetry.
 *
 * @tparam T Data type of the tensor elements (e.g., float, double).
 * @param x Input tensor with shape (batch_size, seq_length, d_model).
 * @return Tensor<T> The normalized and scaled tensor with the same shape as the input.
 */
template <typename T>
Tensor<T> LayerNorm<T>::forward(const Tensor<T>& x) {
    // Store the input tensor for backpropagation
    input_ = x;

    // Calculate mean across the last dimension and expand the dimensions to match the input tensor
    mean_ = x.mean(-1).expandDimsAs(x.shape());

    // Calculate variance across the last dimension
    variance_ = ((x - mean_) * (x - mean_)).mean(-1);

    // Normalize the input
    normalized_ = (x - mean_) / (variance_.expandDimsAs(x.shape()) + epsilon_).sqrt();

    // Add slight noise to normalized values if needed
    Tensor<T> noise = Tensor<T>::uniform(normalized_.shape(), -1e-5, 1e-5);
    normalized_ = normalized_ + noise;

    // Scale and shift using learned parameters gamma and beta (with slight noise)
    return (gamma_.expandDimsAs(x.shape()) * normalized_ + beta_.expandDimsAs(x.shape()));
}

/**
 * @brief Backward pass of the LayerNorm layer.
 *
 * This function calculates the gradient of the loss with respect to the input tensor `x`,
 * the scale (`gamma_`), and the shift (`beta_`), based on the gradient of the loss
 * with respect to the output (`dout`). A small uniform noise is added to the gradients
 * to reduce symmetry.
 *
 * @tparam T Data type of the tensor elements (e.g., float, double).
 * @param dout Gradient of the loss with respect to the output of the LayerNorm layer.
 * @return Tensor<T> The gradient with respect to the input tensor `x`.
 */
template <typename T>
Tensor<T> LayerNorm<T>::backward(const Tensor<T>& dout) {
    const int N = dout.shape()[dout.shape().size() - 1];  // Assuming the last dimension is the one normalized

    // Expand gamma to match the shape of dout
    Tensor<T> gamma_expanded = gamma_.expandDimsAs(dout.shape());

    // Gradient of the normalized output
    Tensor<T> dnorm = dout * gamma_expanded;

    // Gradient of the variance
    Tensor<T> diff = input_ - mean_;  // Shape: same as input_
    Tensor<T> dvar = (dnorm * diff * -0.5 * (variance_ + epsilon_).pow(-1.5).expandDimsAs(dout.shape())).sum(-1);  // Sum across the last dimension
    dvar = dvar.expandDimsAs(diff.shape());  // Ensure correct broadcasting to the shape of input_

    // Gradient of the mean
    Tensor<T> dmean = (dnorm * -1 / (variance_ + epsilon_).sqrt().expandDimsAs(dout.shape())).sum(-1);  // Sum across the last dimension
    dmean = dmean.expandDimsAs(diff.shape());  // Ensure correct broadcasting
    dmean = dmean + dvar * diff * -2.0 / N;

    // Gradient with respect to the input
    Tensor<T> dx = dnorm / (variance_ + epsilon_).sqrt().expandDimsAs(dout.shape());
    dx = dx + dvar * 2.0 * diff / N + dmean / N;

    // Optional: Add slight noise to the gradients to reduce symmetry
    Tensor<T> noise = Tensor<T>::uniform(dx.shape(), -1e-5, 1e-5);
    dx = dx + noise;

    return dx;
}

#endif // LAYERNORM_TPP
