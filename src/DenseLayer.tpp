#ifndef DENSELAYER_TPP
#define DENSELAYER_TPP

#include "../include/Tensor.h"
#include "../include/DenseLayer.h"

template <typename T>
DenseLayer<T>::DenseLayer(int input_units, int output_units, ActivationFunction<T>* activation, T biasInitValue)
    : inputUnits(input_units), outputUnits(output_units), activation(activation) {

    // Initialize weights and gradients tensors with appropriate shapes
    weights = Tensor<T>({input_units, output_units});
    weightGradients = Tensor<T>({input_units, output_units});

    // Initialize bias and its gradients with appropriate shapes
    bias = Tensor<T>({output_units});
    biasGradients = Tensor<T>({output_units});

    // Initialize weights and set bias to the specified initial value
    initializeWeights(weights);
    bias.fill(biasInitValue);
}

template <typename T>
void DenseLayer<T>::initializeWeights(Tensor<T>& inputWeights) {
    // Set up random number generation for weight initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    T limit = std::sqrt(6.0 / (inputUnits + outputUnits)); // Xavier/Glorot initialization limit
    std::uniform_real_distribution<T> dist(-limit, limit);

    // Populate the weights tensor with random values
    for (int i = 0; i < inputWeights.size(); ++i) {
        inputWeights.data[i] = dist(gen);
    }
}

template <typename T>
Tensor<T> DenseLayer<T>::forward(const Tensor<T>& input) {
    // Cache the input for use in the backward pass
    input_cache = input;

    // Create an output tensor initialized with the correct shape
    Tensor<T> output({outputUnits});

    // Direct access to data pointers for better performance
    const T* __restrict__ input_data = input_cache.data.data();
    const T* __restrict__ weights_data = weights.data.data();
    T* __restrict__ output_data = output.data.data();
    const T* __restrict__ bias_data = bias.data.data();

    // Prefetch data to minimize latency (uncomment if necessary)
    // _mm_prefetch((const char*)(weights_data), _MM_HINT_T0);

    // Parallelized and vectorized loop for computing the output
    #pragma omp parallel for simd
    for (size_t j = 0; j < outputUnits; ++j) {
        T sum = bias_data[j];  // Start with bias for each output unit

        // Multiply input by weights and accumulate the results
        #pragma omp simd
        for (size_t k = 0; k < inputUnits; ++k) {
            sum += input_data[k] * weights_data[k * outputUnits + j];
        }
        output_data[j] = sum;  // Store the result in the output tensor
    }

    // Apply the activation function to the output
    activation->forward(output);

    return output;
}

template <typename T>
void DenseLayer<T>::backward(Tensor<T>& grad) {
    // Initialize gradients to zero for accumulation
    biasGradients.fill(0);
    weightGradients.fill(0);

    // Apply the backward pass of the activation function
    activation->backward(grad);

    // Initialize input gradients tensor with the shape of the input
    Tensor<T> input_gradient(input_cache.shape());

    // Compute gradients with respect to weights, biases, and input
    for (size_t j = 0; j < outputUnits; ++j) {
        T grad_data = grad.data[j];  // Gradient from the next layer
        biasGradients.data[j] += grad_data;  // Accumulate bias gradients
        for (size_t k = 0; k < inputUnits; ++k) {
            // Accumulate gradients with respect to input and weights
            input_gradient.data[k] += grad_data * weights.data[k * outputUnits + j];
            weightGradients.data[k * outputUnits + j] += grad_data * input_cache.data[k];
        }
    }

    // Set the gradient to be used for the previous layer
    grad = input_gradient;
}

// Getter for the layer parameters
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> DenseLayer<T>::parameters() {
    // Return references to the weight and bias tensors
    return {std::ref(weights), std::ref(bias)};
}

// Getter for the layer gradients
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> DenseLayer<T>::gradients() {
    // Return references to the weight and bias gradients
    return {std::ref(weightGradients), std::ref(biasGradients)};
}

#endif // DENSELAYER_TPP
