#ifndef DENSELAYER_TPP
#define DENSELAYER_TPP

#include "../include/Tensor.h"
#include "../include/DenseLayer.h"

template <typename T>
DenseLayer<T>::DenseLayer(int input_units, int output_units, ActivationFunction<T>* activation, T biasInitValue)
    : inputUnits(input_units), outputUnits(output_units), activation(activation) {
    weights = Tensor<T>({input_units, output_units});
    weightGradients = Tensor<T>({input_units, output_units});

    bias = Tensor<T>({output_units});
    biasGradients = Tensor<T>({output_units});

    initializeWeights(weights);
    bias.fill(biasInitValue);
}


template <typename T>
void DenseLayer<T>::initializeWeights(Tensor<T>& inputWeights) {
    std::random_device rd;
    std::mt19937 gen(rd());
    T limit = std::sqrt(6.0 / (inputUnits + outputUnits));
    std::uniform_real_distribution<T> dist(-limit, limit);

    for (int i = 0; i < inputWeights.size(); ++i) {
        inputWeights.data[i] = dist(gen);
    }
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(Tensor<T>& input) {
    input_cache = input;
    Tensor<T> output({outputUnits});

    // Direct access to data pointers for better performance
    const T* __restrict__ input_data = input_cache.data.data();
    const T* __restrict__ weights_data = weights.data.data();
    T* __restrict__ output_data = output.data.data();
    const T* __restrict__ bias_data = bias.data.data();

    // Prefetch data to minimize latency (if necessary)
    // _mm_prefetch((const char*)(weights_data), _MM_HINT_T0);

    #pragma omp parallel for simd
    for (size_t j = 0; j < outputUnits; ++j) {
        T sum = bias_data[j];  // Initialize with bias
        
        // Vectorization-friendly loop
        #pragma omp simd
        for (size_t k = 0; k < inputUnits; ++k) {
            sum += input_data[k] * weights_data[k * outputUnits + j];
        }
        output_data[j] = sum;
    }

    // Apply the activation function
    activation->forward(output);

    return output;
}

template <typename T>
void DenseLayer<T>::backward(Tensor<T>& grad_output) {
    // Initialize gradients to zero
    biasGradients.fill(0);
    weightGradients.fill(0);
    
    // Apply the activation function's backward to modify grad_output
    activation->backward(grad_output);

    // Initialize input gradients tensor
    Tensor<T> input_gradient(input_cache.shape());

    // Compute gradients with respect to weights, biases, and input
    for (size_t j = 0; j < outputUnits; ++j) {
        T grad = grad_output.data[j];
        biasGradients.data[j] += grad;
        for (size_t k = 0; k < inputUnits; ++k) {
            input_gradient.data[k] += grad * weights.data[k * outputUnits + j];
            weightGradients.data[k * outputUnits + j] += grad * input_cache.data[k];
        }
    }

    // Update grad_output to be the gradient with respect to the input
    grad_output = input_gradient;
}

#endif // DENSELAYER_TPP