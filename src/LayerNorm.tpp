#ifndef LAYERNORM_TPP
#define LAYERNORM_TPP

#include "../include/LayerNorm.h"

// Constructor initializes the LayerNorm layer with model dimension (`d_model`),
// epsilon value (`epsilon_`), and learnable parameters `gamma_` (scale) and `beta_` (shift).
template <typename T>
LayerNorm<T>::LayerNorm(const int d_model, const float epsilon)
    : d_model_(d_model), epsilon_(epsilon),
      gamma_(Tensor<T>::ones({d_model})),  // Initialize gamma to ones
      beta_(Tensor<T>::zeros({d_model})) { }  // Initialize beta to zeros

// Forward pass of the LayerNorm layer.
template <typename T>
Tensor<T> LayerNorm<T>::forward(const Tensor<T>& input) {
    input_ = input;  // Store input tensor for backpropagation

    // Calculate the mean across the last dimension
    mean_ = input.mean(-1).expandDimsAs(input.shape());

    // Calculate the variance across the last dimension
    variance_ = ((input - mean_) * (input - mean_)).mean(-1);

    // Normalize the input using the calculated mean and variance
    normalized_ = (input - mean_) / (variance_.expandDimsAs(input.shape()) + epsilon_).sqrt();

    // Optional: Add slight noise to normalized values to reduce symmetry
    Tensor<T> noise = Tensor<T>::uniform(normalized_.shape(), -1e-5, 1e-5);
    normalized_ = normalized_ + noise;

    // Scale and shift the normalized output using gamma and beta
    return (gamma_.expandDimsAs(input.shape()) * normalized_ + beta_.expandDimsAs(input.shape()));
}

// Backward pass of the LayerNorm layer.
template <typename T>
void LayerNorm<T>::backward(Tensor<T>& grad) {
    const int N = grad.shape().back();  // Number of elements in the last dimension

    // Expand gamma to match the shape of the gradient tensor
    Tensor<T> gamma_expanded = gamma_.expandDimsAs(grad.shape());

    // Gradient of the normalized output
    Tensor<T> dnorm = grad * gamma_expanded;

    // Calculate the difference between input and mean
    Tensor<T> diff = input_ - mean_;

    // Compute the gradient of the variance
    Tensor<T> dvar = (dnorm * diff * -0.5 * (variance_ + epsilon_).pow(-1.5).expandDimsAs(grad.shape())).sum(-1);
    dvar = dvar.expandDimsAs(diff.shape());  // Ensure broadcasting is correct

    // Compute the gradient of the mean
    Tensor<T> dmean = (dnorm * -1 / (variance_ + epsilon_).sqrt().expandDimsAs(grad.shape())).sum(-1);
    dmean = dmean.expandDimsAs(diff.shape());  // Ensure broadcasting is correct
    dmean = dmean + dvar * diff * -2.0 / N;

    // Compute the gradient with respect to the input
    grad = dnorm / (variance_ + epsilon_).sqrt().expandDimsAs(grad.shape());
    grad = grad + dvar * 2.0 * diff / N + dmean / N;

    // Optional: Add slight noise to the gradients to reduce symmetry
    Tensor<T> noise = Tensor<T>::uniform(grad.shape(), -1e-5, 1e-5);
    grad = grad + noise;
}

#endif // LAYERNORM_TPP
