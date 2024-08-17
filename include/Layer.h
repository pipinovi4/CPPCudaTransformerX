#ifndef LAYER_H
#define LAYER_H

#pragma once
#include "Tensor.h"

/**
 * @class Layer
 * @brief Abstract base class for neural network layers.
 * 
 * This class defines the interface for a neural network layer. All specific 
 * layer types (such as DenseLayer, ConvolutionalLayer, etc.) should inherit 
 * from this class and implement the `forward` and `backward` methods.
 * 
 * @tparam T The data type used for the computations, typically `float` or `double`.
 */
template <typename T>
class Layer {
public:
    /**
     * @brief Perform the forward pass of the layer.
     * 
     * This pure virtual function must be implemented by any derived class. 
     * It takes an input tensor and returns the output tensor after applying 
     * the layer's computation (e.g., a dense layer would perform a matrix 
     * multiplication and apply an activation function).
     * 
     * @param input The input tensor to the layer.
     * @return The output tensor after applying the layer's computation.
     */
    virtual Tensor<float> forward(const Tensor<T>& input) = 0;

    /**
     * @brief Perform the backward pass of the layer.
     * 
     * This pure virtual function must be implemented by any derived class. 
     * It takes a gradient tensor and computes the gradients with respect to 
     * the layer's parameters and input. The gradients are typically used to 
     * update the layer's parameters during training.
     * 
     * @param grad The gradient tensor from the next layer (or loss function).
     */
    virtual void backward(Tensor<T>& grad) = 0;

    /**
     * @brief Virtual destructor.
     * 
     * Ensures that derived classes can clean up resources properly when they are 
     * destroyed. Declaring this as `virtual` is essential to allow the correct 
     * destructor to be called for derived classes.
     */
    virtual ~Layer() = default;
};

#endif //LAYER_H
