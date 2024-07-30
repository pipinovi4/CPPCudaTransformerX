#ifndef DENSELAYER_TPP
#define DENSELAYER_TPP

#include "../include/Tensor.h"
#include "../include/DenseLayer.h"
#include "../include/Optimizer.h"

template <typename T>
void DenseLayer<T>::initializeWeights(Tensor<T>& inputWeights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<T> dist(0, std::sqrt(2.0 / inputUnits));

    for (int i = 0; i < inputWeights.size(); ++i) {
        inputWeights.data[i] = dist(gen);
    }
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T>& input) {
    // Cache the input for use in the backward pass
    input_cache = input;

    // Initialize the output tensor with the appropriate shape
    Tensor<T> output({input.shape()[0], outputUnits});

    // Compute the weighted sum and add the bias
    for (size_t i = 0; i < input.shape()[0]; ++i) {
        for (size_t j = 0; j < outputUnits; ++j) {
            T sum = 0;
            for (size_t k = 0; k < inputUnits; ++k) {
                sum += input.data[i * inputUnits + k] * weights.data[k * outputUnits + j];
            }
            output.data[i * outputUnits + j] = sum + bias.data[j];
        }
    }

    // Apply the activation function
    output = activation->forward(output);

    return output;
}

template <typename T>
Tensor<T> DenseLayer<T>::backward(const Tensor<T>& grad_output) {
    std::vector<int> input_shape = input_cache.shape();
    input_shape.pop_back();
    input_shape.push_back(inputUnits);
    Tensor<T> input_gradient(input_shape);

    // Initialize bias gradients to zero
    std::fill(biasGradients.data.begin(), biasGradients.data.end(), T(0));

    for (size_t i = 0; i < grad_output.shape()[0]; ++i) {
        for (size_t j = 0; j < outputUnits; ++j) {
            T grad = grad_output.data[i * outputUnits + j];
            biasGradients.data[j] += grad; // Update bias gradients
            for (size_t k = 0; k < inputUnits; ++k) {
                weightGradients.data[k * outputUnits + j] += grad * input_cache.data[i * inputUnits + k];
                input_gradient.data[i * inputUnits + k] += grad * weights.data[k * outputUnits + j];
            }
        }
    }
    input_gradient = activation->backward(input_gradient);
    return input_gradient;
}

template <typename T>
void DenseLayer<T>::updateParameters(Optimizer<T>* optimizer, size_t epoch) {
    optimizer->update(weights, weightGradients, epoch);
    optimizer->update(bias, biasGradients, epoch);
}

#endif // DENSELAYER_TPP