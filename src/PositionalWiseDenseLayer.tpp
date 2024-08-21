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
    auto& biases_1_data = biases_1_.data;
    auto& biases_2_data = biases_2_.data;

    // Set biases to the given initialization value.
    for (int i = 0; i < dimensions_feed_forward_; ++i) {
        biases_1_data.emplace_back(bias_init_value);
    }
    for (int i = 0; i < dimensions_model_; ++i) {
        biases_2_data.emplace_back(bias_init_value);
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
    T limit = std::sqrt(6.0 / (weights.shape()[0] + weights.shape()[1]));
    std::normal_distribution<T> dist(-limit, limit);  // Xavier/Glorot initialization

    size_t weights_size = 1;
    for (auto dim : weights.dimensions) {
        weights_size *= dim;  // Calculate the total number of elements in the tensor
    }

    weights.data.clear();  // Clear any existing data in the tensor
    for (size_t i = 0; i < weights_size; i++) {
        weights.data.push_back(dist(gen));  // Initialize the weights with values drawn from a normal distribution
    }
}

// Forward pass of the PositionalWiseDenseLayer.
template <typename T>
Tensor<T> PositionalWiseDenseLayer<T>::forward(const Tensor<T>& input) {
    // Check input dimensions.
    if (input.shape().size() != 3 || input.shape()[2] != dimensions_model_) {
        throw std::invalid_argument("Forward pass error: Input tensor has incorrect dimensions. Expected shape: [batch_size, max_token_size, " + std::to_string(dimensions_model_) + "], received shape: [" + std::to_string(input.shape()[0]) + ", " + std::to_string(input.shape()[1]) + ", " + std::to_string(input.shape()[2]) + "].");
    }

    input_cache_ = input;

    // Perform the first matrix multiplication and add the first bias.
    cache_projection_1_ = input.dot(weights_1_) + biases_1_;

    // Apply the activation function to the result of the first projection.
    // activation_function_.forward(projection_1);

    // Perform the second matrix multiplication and add the second bias.
    Tensor<T> projection_2 = cache_projection_1_.dot(weights_2_) + biases_2_;

    // Reshape the output back to [batch_size, max_token_size, d_model]
    Tensor<T> output = projection_2.reshape({input.shape()[0], input.shape()[1], dimensions_model_});

    // Return the final output tensor after the second projection.
    return output;
}

// Backward pass of the PositionalWiseDenseLayer.
template <typename T>
void PositionalWiseDenseLayer<T>::backward(Tensor<T>& grad) {
    // Check gradient dimensions.
    if (grad.shape().size() != 3 || grad.shape()[2] != dimensions_model_) {
        throw std::invalid_argument("Backward pass error: Gradient tensor has incorrect dimensions. Expected shape: [batch_size, max_token_size, " + std::to_string(dimensions_model_) + "], received shape: [" + std::to_string(grad.shape()[0]) + ", " + std::to_string(grad.shape()[1]) + ", " + std::to_string(grad.shape()[2]) + "].");
    }

    // Step 1: Calculate the gradient w.r.t. the second set of weights and biases.
    grad_weights_2_ = cache_projection_1_.dot(grad).sum(2).sum(0);
    grad_biases_2_ = grad.sum(0).sum(0);

    // Step 2: Backpropagate the gradient through the second layer to the first layer.
    Tensor<T> grad_intermediate = grad.dot(weights_2_.transpose({1, 0}));

    // Step 3: Apply the activation function gradient.
    // If you're using an activation function like ReLU, you'll need to apply its gradient here.
    // activation_function_.backward(grad_intermediate);

    // Step 4: Calculate the gradient w.r.t. the first set of weights and biases.
    grad_weights_1_ = input_cache_.transpose({0, 2, 1}).dot(grad_intermediate).sum(2).sum(0);
    grad_biases_1_ = grad_intermediate.sum(0).sum(0);

    // Step 5: Backpropagate the gradient to the input.
    grad = grad_intermediate.dot(weights_1_.transpose({1, 0})).reshape(input_cache_.shape());
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