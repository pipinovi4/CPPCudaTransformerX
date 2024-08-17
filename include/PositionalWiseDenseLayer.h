#ifndef POSITIONALWISEDENSELAYER_H
#define POSITIONALWISEDENSELAYER_H

#pragma once
#include "ActivationFunction.h"
#include "Layer.h"
#include "Tensor.h"

/**
 * @class PositionalWiseDenseLayer
 * @brief Implements a position-wise feedforward layer, commonly used in transformer models.
 *
 * This layer applies two dense (fully connected) transformations to each position of the input independently.
 * The input is first transformed to a higher-dimensional space using the first set of weights and biases.
 * Then, an activation function is applied, and the output is transformed back to the original input dimension
 * using the second set of weights and biases. The same set of transformations is applied to each position
 * independently.
 *
 * @tparam T The data type of the elements in the tensors (e.g., float, double).
 */
template <typename T>
class PositionalWiseDenseLayer final : public Layer<T> {
public:
    /**
     * @brief Constructor for the PositionalWiseDenseLayer class.
     *
     * @param d_model The input dimension size.
     * @param d_ff The output dimension size (intermediate size, usually larger than d_model).
     * @param activation_function The activation function to apply between the two dense layers.
     * @param bias_init_value The initial value for the biases (default is 0.0).
     */
    PositionalWiseDenseLayer(int d_model, int d_ff, ActivationFunction<T>& activation_function, T bias_init_value = 0.0);

    /**
     * @brief Forward pass through the PositionalWiseDenseLayer.
     *
     * This function applies the first dense layer, then the activation function,
     * and finally the second dense layer. The result is returned as the output.
     *
     * @param input The input tensor to the layer.
     * @return The output tensor after the position-wise transformations.
     */
    Tensor<T> forward(const Tensor<T>& input) override;

    /**
     * @brief Backward pass through the PositionalWiseDenseLayer to compute gradients.
     *
     * This function computes the gradients for the weights and biases based on the
     * gradient of the loss with respect to the output of this layer. It also computes
     * the gradient of the loss with respect to the input, which is returned.
     *
     * @param grad The gradient of the loss with respect to the output of this layer.
     * @return The gradient of the loss with respect to the input of this layer.
     */
    void backward(Tensor<T>& grad) override;

    /**
     * @brief Returns the parameters (weights and biases) of the layer.
     *
     * This function is used to access the parameters of the layer, which can then be
     * passed to an optimizer for updating.
     *
     * @return A vector of references to the weights and biases tensors.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters();

    /**
     * @brief Returns the gradients of the layer's parameters.
     *
     * This function is used to access the gradients of the parameters, which can then
     * be used by an optimizer to update the parameters.
     *
     * @return A vector of references to the gradient tensors for the weights and biases.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    Tensor<T> input_cache_; // Cache the input for the backward pass

    ActivationFunction<T>& activation_function_; // Activation function

    int d_model_; // Input dimension
    int d_ff_;    // Hidden/output dimension

    // Weights and biases for the dense layer
    Tensor<T> weights_1_; // Weights for the first projection (d_model -> d_ff)
    Tensor<T> biases_1_;  // Biases for the first projection

    Tensor<T> weights_2_; // Weights for the second projection (d_ff -> d_model)
    Tensor<T> biases_2_;  // Biases for the second projection

    // Gradients
    Tensor<T> grad_weights_1_; // Gradients for the first projection
    Tensor<T> grad_biases_1_; // Gradients for the first projection

    Tensor<T> grad_weights_2_; // Gradients for the second projection
    Tensor<T> grad_biases_2_; // Gradients for the second projection

    /**
     * @brief Helper function to initialize the weights using Xavier initialization.
     *
     * This function initializes the weights with values drawn from a uniform distribution
     * with limits based on the input and output dimensions of the layer.
     *
     * @param weights The tensor representing the weights to be initialized.
     */
    void initializeWeights(Tensor<T>& weights);
};

#include "../src/PositionalWiseDenseLayer.tpp"

#endif // POSITIONALWISEDENSELAYER_H
