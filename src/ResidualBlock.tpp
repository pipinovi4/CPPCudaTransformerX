#ifndef RESIDUALBLOCK_TPP
#define RESIDUALBLOCK_TPP

#include "../include/ResidualBlock.h"
#include "../include/Tensor.h"

/**
 * @brief ResidualBlock constructor
 *
 * Initializes the ResidualBlock with a given model dimension and epsilon value
 * for layer normalization.
 *
 * @tparam T The data type of the tensor elements (e.g., float, double).
 * @param d_model The dimension of the model (size of the feature vector).
 * @param epsilon A small value added to the variance for numerical stability in layer normalization.
 */
template <typename T>
ResidualBlock<T>::ResidualBlock(int d_model, float epsilon)
    : layer_norm_(d_model, epsilon) {}

/**
 * @brief Forward pass through the Residual Block.
 *
 * This function performs the forward pass of the Residual Block. It adds the
 * residual connection (input + processed), applies layer normalization, and
 * returns the output.
 *
 * @param input The original input tensor to the block.
 * @param processed The processed tensor after passing through some layers.
 * @return Tensor<T> The output tensor after applying the residual connection and layer normalization.
 */
template <typename T>
Tensor<T> ResidualBlock<T>::forward(const Tensor<T>& input, const Tensor<T>& processed) {
    // Store the input and processed tensors for use during backpropagation
    input_ = input;
    processed_ = processed;

    // Add the residual connection: output = input + processed
    output_ = input + processed;

    // Apply layer normalization to the result of the residual connection
    return layer_norm_.forward(output_);
}

/**
 * @brief Backward pass through the Residual Block.
 *
 * This function performs the backward pass of the Residual Block. It computes
 * the gradient of the loss with respect to the output of the layer normalization
 * and then calculates the gradients with respect to the input and processed tensors.
 *
 * @param dout The gradient of the loss with respect to the output tensor from the forward pass.
 * @return Tensor<T> The gradient with respect to the input tensor.
 */
template <typename T>
Tensor<T> ResidualBlock<T>::backward(const Tensor<T>& dout) {
    // Calculate the gradient of the loss with respect to the output of the layer normalization
    Tensor<T> dnorm = layer_norm_.backward(dout);

    // The gradient of the output tensor is used to compute the gradient of the input tensors
    Tensor<T> doutput = dnorm;

    // Return the gradient with respect to the input tensor
    return doutput;
}

#endif // RESIDUALBLOCK_TPP
