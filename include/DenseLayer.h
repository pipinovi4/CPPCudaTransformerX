/**
 * @file DenseLayer.h
 * @brief Defines the DenseLayer class, a fully connected neural network layer with customizable activation.
 */

#ifndef DENSELAYER_H
#define DENSELAYER_H

#pragma once

#include "Layer.h"
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "eigen3/Eigen/Dense"

/**
 * @class DenseLayer
 * @brief A fully connected layer (dense layer) in a neural network.
 * 
 * This class implements a fully connected layer where every input is connected to every output.
 * It also includes an activation function to introduce non-linearity into the network.
 * 
 * @tparam T The data type used for computations (e.g., float, double).
 */
template <typename T>
class DenseLayer final : public Layer<T> {
public:
    /**
     * @brief Constructs a DenseLayer object with specified input and output units.
     * 
     * @param input_units The number of input units (size of the input vector).
     * @param output_units The number of output units (size of the output vector).
     * @param activation The activation function applied to the layer's output (default is Linear).
     * @param biasInitValue The initial value for the bias terms (default is 0.0).
     */
    DenseLayer(int input_units, int output_units, ActivationFunction<T>* activation = nullptr, T biasInitValue = 0.0);

    /**
     * @brief Initializes the weights of the layer.
     * 
     * This method sets the weights of the layer to the provided tensor.
     * 
     * @param inputWeights A tensor containing the initial weights.
     */
    void initializeWeights(Tensor<T>& inputWeights);

    /**
     * @brief Performs the forward pass of the layer.
     * 
     * @param input The input tensor to the layer.
     * @return The output tensor after applying the weights, bias, and activation function.
     */
    Tensor<T> forward(const Tensor<T>& input) override;

    /**
     * @brief Performs the backward pass of the layer, computing gradients.
     * 
     * @param grad The gradient of the loss with respect to the output of this layer.
     */
    void backward(Tensor<T>& grad) override;

    /**
     * @brief Get the parameters of the layer.
     *
     * This method returns a vector of references to the layer's parameters.
     *
     * @return A vector of references to the layer's parameters.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters() override;

    /**
     * @brief Get the gradients of the layer.
     *
     * This method returns a vector of references to the layer's gradients.
     *
     * @return A vector of references to the layer's gradients.
     */
    std::vector<std::reference_wrapper<Tensor<T>>> gradients() override;

private:
    int inputUnits;  ///< The number of input units.
    int outputUnits; ///< The number of output units.

    ActivationFunction<T>* activation; ///< The activation function for the layer.

    Tensor<T> weights; ///< The weight matrix of the layer.
    Tensor<T> bias;    ///< The bias vector of the layer.

    Tensor<T> input_cache; ///< Cache of the input tensor for use in the backward pass.

    Tensor<T> weightGradients; ///< Gradient of the loss with respect to the weights.
    Tensor<T> biasGradients;   ///< Gradient of the loss with respect to the biases.
    Tensor<T> inputGradients;  ///< Gradient of the loss with respect to the input.
};

#include "../src/DenseLayer.tpp"

#endif //DENSELAYER_H
