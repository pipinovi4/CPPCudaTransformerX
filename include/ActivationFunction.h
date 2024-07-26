#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include "Tensor.h"

template <typename T>
class ActivationFunction {
public:
    // Sigmoid activation function
    static Tensor<T> sigmoid(const Tensor<T>& x);

    // Softmax activation function (optional)
    static Tensor<T> softmax(const Tensor<T>& x);

    // ReLU activation function
    static Tensor<T> relu(const Tensor<T>& x);

    static Tensor<T> leaky_relu(const Tensor<T>& x, const T alpha);

    static Tensor<T> elu(const Tensor<T>& x, const T alpha);

    // Tanh activation function (optional)
    static Tensor<T> tanh(const Tensor<T>& x);
};

template class ActivationFunction<float>;
template class ActivationFunction<double>;

#include "../src/ActivationFunction.tpp" // Include the template implementation file

#endif // ACTIVATIONFUNCTION_H