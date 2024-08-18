#ifndef POSITIONALWISEDENSELAYER_TPP
#define POSITIONALWISEDENSELAYER_TPP

#include "../include/PositionalWiseDenseLayer.h"

// Constructor for the PositionalWiseDenseLayer class.
template <typename T>
PositionalWiseDenseLayer<T>::PositionalWiseDenseLayer(const int dimensions_model, const int dimensions_feed_forward, ActivationFunction<T>& activation_function, T bias_init_value)
    : activation_function_(activation_function), dimensions_model_(dimensions_model), dimensions_feed_forward_(dimensions_feed_forward) {
    // Initialize the first set of weights with dimensions (dimensions_model_, dimensions_feed_forward_) (without data only reserve memory).
    weights_1_ = Tensor<T>({dimensions_model_, dimensions_feed_forward_}, dimensions_model_ * dimensions_feed_forward_);
    // Initialize the second set of weights with dimensions (dimensions_feed_forward_, dimensions_model_).
    weights_2_ = Tensor<T>({dimensions_feed_forward_, dimensions_model_}, dimensions_feed_forward_ * dimensions_model_);
    // Initialize the weights using the initializeWeights function.
    initializeWeights(weights_1_);
    initializeWeights(weights_2_);

    // Initialize biases with dimensions (dimensions_feed_forward_) and (dimensions_model_) (without data only reserve memory).
    biases_1_ = Tensor<T>({dimensions_feed_forward_}, dimensions_feed_forward_);
    biases_2_ = Tensor<T>({dimensions_model_}, dimensions_model_);

    // Data access for better performance.
    T* biases_1_data = biases_1_.data.data();
    T* biases_2_data = biases_2_.data.data();

    // Set biases to the given initialization value.
    for (int i = 0; i < dimensions_feed_forward_; ++i) {
        biases_1_data[i] = bias_init_value;
    }
    for (int i = 0; i < dimensions_model_; ++i) {
        biases_2_data[i] = bias_init_value;
    }

    // Initialize gradients for weights as zero tensors.
    grad_weights_1_ = Tensor<T>({dimensions_model_, dimensions_feed_forward_});
    grad_weights_2_ = Tensor<T>({dimensions_feed_forward_, dimensions_model_});

    // Initialize gradients for biases as zero tensors.
    grad_biases_1_ = Tensor<T>({dimensions_feed_forward_});
    grad_biases_2_ = Tensor<T>({dimensions_model_});
}

// Initialize the weights using the Xavier (Glorot) initialization.
template <typename T>
void PositionalWiseDenseLayer<T>::initializeWeights(Tensor<T>& weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Calculate the limit for uniform distribution based on the dimensions.
    T limit = std::sqrt(6.0 / (dimensions_model_ + dimensions_feed_forward_));
    std::uniform_real_distribution<T> dist(-limit, limit);

    // Pointer to the weights data for efficient access.
    T* weights_data = weights.data.data();

    // Initialize each weight value using the distribution.
    for (int i = 0; i < weights.size(); ++i) {
        weights_data[i] = dist(gen);
    }
}

// Forward pass of the PositionalWiseDenseLayer.
template <typename T>
Tensor<T> PositionalWiseDenseLayer<T>::forward(const Tensor<T>& input) {
    // Check input dimensions.
    if (input.shape().size() != 2 || input.shape()[1] != dimensions_model_) {
        throw std::invalid_argument("Forward pass error: Input tensor has incorrect dimensions. Expected shape: [batch_size, " + std::to_string(dimensions_model_) + "], received shape: [" + std::to_string(input.shape()[0]) + ", " + std::to_string(input.shape()[1]) + "].");
    }

    // Cache the input tensor for use in the backward pass.
    input_cache_ = input;

    // Perform the first matrix multiplication and add the first bias.
    Tensor<T> projection_1 = input.dot(weights_1_);
    if (projection_1.shape()[1] != dimensions_feed_forward_) {
        throw std::runtime_error("Forward pass error: Projection 1 has incorrect dimensions after dot product with weights_1_. Expected shape: [batch_size, " + std::to_string(dimensions_feed_forward_) + "], received shape: [" + std::to_string(projection_1.shape()[0]) + ", " + std::to_string(projection_1.shape()[1]) + "].");
    }
    projection_1 += biases_1_;

    // Apply the activation function to the result of the first projection.
    activation_function_.forward(projection_1);

    // Perform the second matrix multiplication and add the second bias.
    Tensor<T> projection_2 = projection_1.dot(weights_2_);
    if (projection_2.shape()[1] != dimensions_model_) {
        throw std::runtime_error("Forward pass error: Projection 2 has incorrect dimensions after dot product with weights_2_. Expected shape: [batch_size, " + std::to_string(dimensions_model_) + "], received shape: [" + std::to_string(projection_2.shape()[0]) + ", " + std::to_string(projection_2.shape()[1]) + "].");
    }
    projection_2 += biases_2_;

    // Return the final output tensor after the second projection.
    return projection_2;
}

// Backward pass of the PositionalWiseDenseLayer.
template <typename T>
void PositionalWiseDenseLayer<T>::backward(Tensor<T>& grad) {
    // Check gradient dimensions.
    if (grad.shape().size() != 2 || grad.shape()[1] != dimensions_model_) {
        throw std::invalid_argument("Backward pass error: Gradient tensor has incorrect dimensions. Expected shape: [batch_size, " + std::to_string(dimensions_model_) + "], received shape: [" + std::to_string(grad.shape()[0]) + ", " + std::to_string(grad.shape()[1]) + "].");
    }

    // Compute the gradients with respect to the second bias by summing along the first dimension.
    grad_biases_2_ = grad.sum(0);

    // Compute the gradients with respect to the second weights by matrix multiplication of the transposed input and the gradient.
    grad_weights_2_ = input_cache_.transpose().dot(grad);

    // Compute the gradients with respect to the first projection by matrix multiplication of the gradient and the transposed second weights.
    Tensor<T> grad_projection_1 = grad.dot(weights_2_.transpose());

    // Apply the backward pass of the activation function to the gradient of the first projection.
    activation_function_.backward(grad_projection_1);

    // Compute the gradients with respect to the first bias by summing along the first dimension.
    grad_biases_1_ = grad_projection_1.sum(0);

    // Compute the gradients with respect to the first weights by matrix multiplication of the transposed input and the gradient of the first projection.
    grad_weights_1_ = input_cache_.transpose().dot(grad_projection_1);

    // Compute the gradients with respect to the input by matrix multiplication of the gradient of the first projection and the transposed first weights.
    grad = grad_projection_1.dot(weights_1_.transpose());
}

// Update the weights of the PositionalWiseDenseLayer using the optimizer.
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::parameters() {
    // Return a vector containing references to all the parameters (weights and biases).
    return {weights_1_, biases_1_, weights_2_, biases_2_};
}

// Getter for the gradients of the PositionalWiseDenseLayer.
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::gradients() {
    // Return a vector containing references to all the gradients of the parameters.
    return {grad_weights_1_, grad_biases_1_, grad_weights_2_, grad_biases_2_};
}

#endif //POSITIONALWISEDENSELAYER_TPP