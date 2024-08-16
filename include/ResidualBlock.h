#ifndef RESIDUALBLOCK_H
#define RESIDUALBLOCK_H

#pragma once
#include "Tensor.h"
#include "LayerNorm.h"

template <typename T>
class ResidualBlock {
public:
    // Constructor that initializes the ResidualBlock
    explicit ResidualBlock(int d_model, float epsilon = 1e-6);

    // Forward pass: takes input tensor and returns the output after residual connection and normalization
    Tensor<T> forward(const Tensor<T>& input, const Tensor<T>& processed);

    // Backward pass: calculates the gradients with respect to the input and processed tensors
    Tensor<T> backward(const Tensor<T>& dout);

private:
    // Layer normalization applied after adding the residual
    LayerNorm<T> layer_norm_;

    // Intermediate tensors to store results needed for backpropagation
    Tensor<T> input_;       // The original input tensor
    Tensor<T> processed_;   // The processed tensor before adding the residual
    Tensor<T> output_;      // The output tensor after forward pass
};

#include "../src/ResidualBlock.tpp"

#endif // RESIDUALBLOCK_H
