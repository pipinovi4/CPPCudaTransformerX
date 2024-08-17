#ifndef POSITIONALWISEDENSELAYER_TPP
#define POSITIONALWISEDENSELAYER_TPP

#include "../include/PositionalWiseDenseLayer.h"

template <typename T>
PositionalWiseDenseLayer<T>::PositionalWiseDenseLayer(const int d_model, const int d_ff, ActivationFunction<T>& activation_function, T bias_init_value)
    : activation_function_(activation_function), d_model_(d_model), d_ff_(d_ff) {
    // Initialize the first set of weights with dimensions (d_model, d_ff)
    weights_1_ = Tensor<T>({d_model, d_ff});
    // Initialize the second set of weights with dimensions (d_ff, d_model)
    weights_2_ = Tensor<T>({d_ff, d_model});
    // Initialize the weights using the initializeWeights function
    initializeWeights(weights_1_);
    initializeWeights(weights_2_);

    // Initialize biases as zero tensors
    biases_1_ = Tensor<T>({d_ff});
    biases_2_ = Tensor<T>({d_model});
    // Set biases to the given initialization value
    biases_1_.fill(bias_init_value);
    biases_2_.fill(bias_init_value);

    // Initialize gradients for weights as zero tensors
    grad_weights_1_ = Tensor<T>({d_model, d_ff});
    grad_weights_2_ = Tensor<T>({d_ff, d_model});

    // Initialize gradients for biases as zero tensors
    grad_biases_1_ = Tensor<T>({d_ff});
    grad_biases_2_ = Tensor<T>({d_model});
}

template <typename T>
void PositionalWiseDenseLayer<T>::initializeWeights(Tensor<T>& weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    // Calculate the limit for uniform distribution based on the dimensions
    T limit = std::sqrt(6.0 / (d_model_ + d_ff_));
    std::uniform_real_distribution<T> dist(-limit, limit);

    // Pointer to the weights data for efficient access
    auto weights_data = weights.data.data();

    // Initialize each weight value using the distribution
    for (int i = 0; i < weights.size(); ++i) {
        weights_data[i] = (dist(gen));
    }
}

template <typename T>
Tensor<T> PositionalWiseDenseLayer<T>::forward(const Tensor<T>& input) {
    // Cache the input tensor for use in the backward pass
    input_cache_ = input;

    // Perform the first matrix multiplication and add the first bias
    Tensor<T> projection_1 = input.dot(weights_1_);
    projection_1 += biases_1_;

    // Apply the activation function to the result of the first projection
    activation_function_.forward(projection_1);

    // Perform the second matrix multiplication and add the second bias
    Tensor<T> projection_2 = projection_1.dot(weights_2_);
    projection_2 += biases_2_;

    // Return the final output tensor after the second projection
    return projection_2;
}

template <typename T>
void PositionalWiseDenseLayer<T>::backward(Tensor<T>& grad) {
    // Compute the gradients with respect to the second bias by summing along the first dimension
    grad_biases_2_ = grad.sum(0);
    // Compute the gradients with respect to the second weights by matrix multiplication of the transposed input and the gradient
    grad_weights_2_ = input_cache_.transpose().dot(grad);

    // Compute the gradients with respect to the first projection by matrix multiplication of the gradient and the transposed second weights
    Tensor<T> grad_projection_1 = grad.dot(weights_2_.transpose());

    // Apply the backward pass of the activation function to the gradient of the first projection
    activation_function_.backward(grad_projection_1);

    // Compute the gradients with respect to the first bias by summing along the first dimension
    grad_biases_1_ = grad_projection_1.sum(0);
    // Compute the gradients with respect to the first weights by matrix multiplication of the transposed input and the gradient of the first projection
    grad_weights_1_ = input_cache_.transpose().dot(grad_projection_1);

    // Compute the gradients with respect to the input by matrix multiplication of the gradient of the first projection and the transposed first weights
    grad = grad_projection_1.dot(weights_1_.transpose());
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::parameters() {
    // Return a vector containing references to all the parameters (weights and biases)
    return {weights_1_, biases_1_, weights_2_, biases_2_};
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::gradients() {
    // Return a vector containing references to all the gradients of the parameters
    return {grad_weights_1_, grad_biases_1_, grad_weights_2_, grad_biases_2_};
}

#endif //POSITIONALWISEDENSELAYER_TPP
