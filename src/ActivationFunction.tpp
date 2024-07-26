#ifndef ACTIVATIONFUNCTION_TPP
#define ACTIVATIONFUNCTION_TPP

#include <cmath>
#include "../include/Tensor.h"

template <typename T>
Tensor<T> ActivationFunction<T>::sigmoid(const Tensor<T>& x) {
    Tensor<T> result(x.shape());

    // Iterate over all elements in the tensor
    for (size_t i = 0; i < x.data.size(); ++i) {
        // Compute sigmoid function
        const auto value = static_cast<double>(x.data[i]);
        result.data[i] = static_cast<T>(1.0 / (1.0 + std::exp(-value)));
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::sigmoidDerivative(const Tensor<T>& x) {
    Tensor<T> sigmoid_x = sigmoid(x);
    Tensor<T> result(x.shape());

    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = sigmoid_x.data[i] * (1 - sigmoid_x.data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::softmax(const Tensor<T>& x) {
    Tensor<T> result(x.shape());
    const auto& shape = x.shape();
    size_t outerSize = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        outerSize *= shape[i];
    }
    size_t innerSize = shape.back();

    for (size_t i = 0; i < outerSize; ++i) {
        // Find the maximum value in the current slice for numerical stability
        double maxVal = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < innerSize; ++j) {
            maxVal = std::max(maxVal, static_cast<double>(x.data[i * innerSize + j]));
        }

        // Compute the exponentials and their sum
        double sum = 0.0;
        for (size_t j = 0; j < innerSize; ++j) {
            result.data[i * innerSize + j] = std::exp(static_cast<double>(x.data[i * innerSize + j]) - maxVal);
            sum += result.data[i * innerSize + j];
        }

        // Normalize the values to get the softmax probabilities
        for (size_t j = 0; j < innerSize; ++j) {
            result.data[i * innerSize + j] /= sum;
        }
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::softmaxDerivative(const Tensor<T>& x) {
    Tensor<T> softmax_x = softmax(x);
    Tensor<T> result(x.shape());

    const auto& shape = x.shape();
    size_t outerSize = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
        outerSize *= shape[i];
    }
    size_t innerSize = shape.back();

    for (size_t i = 0; i < outerSize; ++i) {
        for (size_t j = 0; j < innerSize; ++j) {
            T softmax_val = softmax_x.data[i * innerSize + j];
            result.data[i * innerSize + j] = softmax_val * (1 - softmax_val);
        }
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::relu(const Tensor<T>& x) {
    Tensor<T> result(x.shape());

    // Iterate over all elements in the tensor
    for (size_t i = 0; i < x.data.size(); ++i) {
        // Compute ReLU function
        result.data[i] = std::max(static_cast<T>(0), x.data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::reluDerivative(const Tensor<T>& x) {
    Tensor<T> result(x.shape());

    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = (x.data[i] > 0) ? 1 : 0;
    }

    return result;
}


template <typename T>
Tensor<T> ActivationFunction<T>::leakyRelu(const Tensor<T>& x, const T alpha) {
    Tensor<T> result(x.shape());

    // Iterate over all elements in the tensor
    for (size_t i = 0; i < x.data.size(); ++i) {
        // Compute Leaky ReLU function
        result.data[i] = (x.data[i] > 0) ? x.data[i] : alpha * x.data[i];
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::leakyReluDerivative(const Tensor<T>& x, const T alpha) {
    Tensor<T> result(x.shape());

    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = (x.data[i] > 0) ? 1 : alpha;
    }

    return result;
}


template <typename T>
Tensor<T> ActivationFunction<T>::elu(const Tensor<T>& x, const T alpha) {
    Tensor<T> result(x.shape());

    // Iterate over all elements in the tensor
    for (size_t i = 0; i < x.data.size(); ++i) {
        // Compute ELU function
        result.data[i] = (x.data[i] > 0) ? x.data[i] : alpha * (std::exp(x.data[i]) - 1);
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::eluDerivative(const Tensor<T>& x, const T alpha) {
    Tensor<T> result(x.shape());

    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = (x.data[i] > 0) ? 1 : alpha * std::exp(x.data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::tanh(const Tensor<T>& x) {
    Tensor<T> result(x.shape());

    // Iterate over all elements in the tensor
    for (size_t i = 0; i < x.data.size(); ++i) {
        // Compute tanh function
        result.data[i] = std::tanh(x.data[i]);
    }

    return result;
}

template <typename T>
Tensor<T> ActivationFunction<T>::tanhDerivative(const Tensor<T>& x) {
    Tensor<T> tanh_x = tanh(x);
    Tensor<T> result(x.shape());

    for (size_t i = 0; i < x.data.size(); ++i) {
        result.data[i] = 1 - tanh_x.data[i] * tanh_x.data[i];
    }

    return result;
}


#endif // ACTIVATIONFUNCTION_TPP
