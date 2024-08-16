#ifndef POSITIONALWISEDENSELAYER_TPP
#define POSITIONALWISEDENSELAYER_TPP

#include "../include/PositionalWiseDenseLayer.h"

template <typename T>
PositionalWiseDenseLayer<T>::PositionalWiseDenseLayer(const int d_model, const int d_ff, ActivationFunction<T>& activation_function, T bias_init_value)
    : activation_function_(activation_function), d_model_(d_model), d_ff_(d_ff) {
    // Define weights and resize them
    weights_1_ = Tensor<T>({d_model, d_ff});
    weights_2_ = Tensor<T>({d_ff, d_model});
    initializeWeights(weights_1_);
    initializeWeights(weights_2_);

    // Define biases as zeros
    biases_1_ = Tensor<T>({d_ff});
    biases_2_ = Tensor<T>({d_model});
    biases_1_.fill(bias_init_value);
    biases_2_.fill(bias_init_value);

    // Initialize parameter gradients and resize them
    grad_weights_1_ = Tensor<T>({d_model, d_ff}, d_model * d_ff);
    grad_weights_2_ = Tensor<T>({d_ff, d_model}, d_ff * d_model);
    grad_weights_1_.fill(0.0);
    grad_weights_2_.fill(0.0);

    // Initialize biases gradients as zeros
    grad_biases_1_ = Tensor<T>({d_ff});
    grad_biases_2_ = Tensor<T>({d_model});
    grad_biases_1_.fill(0.0);
    grad_biases_2_.fill(0.0);
}

template <typename T>
void PositionalWiseDenseLayer<T>::initializeWeights(Tensor<T>& weights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    T limit = std::sqrt(6.0 / (d_model_ + d_ff_));
    std::uniform_real_distribution<T> dist(-limit, limit);

    // Data pointer for better performance
    auto weights_data = weights.data.data();

    for (int i = 0; i < weights.size(); ++i) {
        weights_data[i] = (dist(gen));
    }
}

template <typename T>
Tensor<T> PositionalWiseDenseLayer<T>::forward(const Tensor<T>& input) {
    // Cache the input for the backward pass
    input_cache_ = input;

    // Perform the first projection
    Tensor<T> projection_1 = input.dot(weights_1_);
    projection_1 += biases_1_;

    // Apply the activation function
    activation_function_.forward(projection_1);

    // Perform the second projection
    Tensor<T> projection_2 = projection_1.dot(weights_2_);
    projection_2 += biases_2_;

    return projection_2;
}

template <typename T>
Tensor<T> PositionalWiseDenseLayer<T>::backward(const Tensor<T>& grad_output) {
    // Compute the gradients with respect to the second projection
    grad_biases_2_ = grad_output.sum(0);
    grad_weights_2_ = input_cache_.transpose().dot(grad_output);

    // Compute the gradients with respect to the first projection
    Tensor<T> grad_projection_1 = grad_output.dot(weights_2_.transpose());

    // Apply the activation function's backward to modify grad_projection_1
    activation_function_.backward(grad_projection_1);

    // Compute the gradients with respect to the first projection weights and biases
    grad_biases_1_ = grad_projection_1.sum(0);
    grad_weights_1_ = input_cache_.transpose().dot(grad_projection_1);

    // Compute the gradients with respect to the input
    return grad_projection_1.dot(weights_1_.transpose());
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::parameters() {
    return {weights_1_, biases_1_, weights_2_, biases_2_};
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> PositionalWiseDenseLayer<T>::gradients() {
    return {grad_weights_1_, grad_biases_1_, grad_weights_2_, grad_biases_2_};
}

#endif //POSITIONALWISEDENSELAYER_TPP
