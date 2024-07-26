#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include "Tensor.h"

template <typename T>
class ActivationFunction {
public:
    // Sigmoid activation function
    static Tensor<T> sigmoid(const Tensor<T>& x);
    static Tensor<T> sigmoidDerivative(const Tensor<T>& x);

    // Softmax activation function (optional)
    static Tensor<T> softmax(const Tensor<T>& x);
    static Tensor<T> softmaxDerivative(const Tensor<T>& x);

    // ReLU activation function
    static Tensor<T> relu(const Tensor<T>& x);
    static Tensor<T> reluDerivative(const Tensor<T>& x);

    static Tensor<T> leakyRelu(const Tensor<T>& x, T alpha);
    static Tensor<T> leakyReluDerivative(const Tensor<T>& x, T alpha);

    static Tensor<T> elu(const Tensor<T>& x, T alpha);
    static Tensor<T> eluDerivative(const Tensor<T>& x, T alpha);

    // Tanh activation function (optional)
    static Tensor<T> tanh(const Tensor<T>& x);
    static Tensor<T> tanhDerivative(const Tensor<T>& x);
};

template class ActivationFunction<float>;
template class ActivationFunction<double>;

#include "../src/ActivationFunction.tpp"

#endif // ACTIVATIONFUNCTION_H