#ifndef LAYERNORM_H
#define LAYERNORM_H

#pragma once
#include "Tensor.h"

template <typename T>
class LayerNorm {
public:
    // Constructor initializes the LayerNorm layer with the model dimension and epsilon value
    explicit LayerNorm(int d_model, float epsilon = 1e-6);

    // Forward pass: applies layer normalization to the input tensor
    Tensor<T> forward(const Tensor<T>& x);

    // Backward pass: computes gradients with respect to the input tensor
    Tensor<T> backward(const Tensor<T>& dout);

private:
    int d_model_;  // Dimension of the model (size of the last dimension)
    float epsilon_;  // Small value to prevent division by zero
    Tensor<T> gamma_;  // Learnable scale parameter
    Tensor<T> beta_;  // Learnable shift parameter

    // Intermediate results needed for backpropagation
    Tensor<T> normalized_;  // Normalized input tensor
    Tensor<T> mean_;  // Mean across the last dimension
    Tensor<T> variance_;  // Variance across the last dimension

    // Store the input tensor for use during backpropagation
    Tensor<T> input_;
};

#include "../src/LayerNorm.tpp"

#endif //LAYERNORM_H
