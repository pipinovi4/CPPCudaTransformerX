#ifndef ACTIVATIONFUNCTION_TPP
#define ACTIVATIONFUNCTION_TPP

#include <cmath>
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

template <typename T>
class ActivationFunction<T>::Sigmoid : public ActivationFunction<T> {
public:
    // Forward method of the Sigmoid activation function (1 / (1 + exp(-x))
    Tensor<float> forward(const Tensor<T>& x) override {
        Tensor<float> output = x;
        for (auto& val : output.data) {
            val = 1.0 / (1.0 + std::exp(-val));
        }
        return output;
    }

    // Backward method for computation of the gradient (derivative of the sigmoid function - sigmoid(x) * (1 - sigmoid(x))
    Tensor<float> backward(const Tensor<T>& gradOutput) override {
        Tensor<float> sigmoidOutput = forward(gradOutput);
        Tensor<float> gradInput = gradOutput;
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradInput.data[i] = gradOutput.data[i] * sigmoidOutput.data[i] * (1 - sigmoidOutput.data[i]);
        }
        return gradInput;
    }
};

template <typename T>
class ActivationFunction<T>::Softmax final : public ActivationFunction<T> {
public:
    // Forward method of the Softmax activation function (exp(x_i) / sum(exp(x_j)))
    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> result(x.shape());
        const auto& shape = x.shape();
        size_t outerSize = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            outerSize *= shape[i];
        }
        const size_t innerSize = shape.back();

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

    // Backward method for computation of the gradient (derivative of the softmax function - softmax(x) * (1 - softmax(x))
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> softmaxOutput = forward(gradOutput);
        Tensor<T> gradInput = gradOutput;
        const auto& shape = gradOutput.shape();
        size_t outerSize = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            outerSize *= shape[i];
        }
        const size_t innerSize = shape.back();

        for (size_t i = 0; i < outerSize; ++i) {
            for (size_t j = 0; j < innerSize; ++j) {
                T softmax_val = softmaxOutput.data[i * innerSize + j];
                gradInput.data[i * innerSize + j] = softmax_val * (1 - softmax_val);
            }
        }

        return gradInput;
    }
};

template <typename T>
class ActivationFunction<T>::ReLU final : public ActivationFunction<T> {
public:
    // Forward method of the ReLU activation function (max(0, x))
    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = std::max(static_cast<T>(0), x.data[i]);
        }
        return result;
    }

    // Backward method for computation of the gradient (derivative of the ReLU function - 1 if x > 0, 0 otherwise)
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> gradInput = gradOutput;
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradInput.data[i] = (gradOutput.data[i] > 0) ? 1 : 0;
        }
        return gradInput;
    }
};

template <typename T>
class ActivationFunction<T>::LeakyReLU final : public ActivationFunction<T> {
public:
    // Forward method of the Leaky ReLU activation function (x if x > 0, alpha * x otherwise)
    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = (x.data[i] > 0) ? x.data[i] : 0.01 * x.data[i];
        }
        return result;
    }

    // Backward method for computation of the gradient (derivative of the Leaky ReLU function - 1 if x > 0, alpha otherwise)
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> gradInput = gradOutput;
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradInput.data[i] = (gradOutput.data[i] > 0) ? 1 : 0.01;
        }
        return gradInput;
    }
};

template <typename T>
class ActivationFunction<T>::ELU final : public ActivationFunction<T> {
public:
    // Forward method of the ELU activation function (x if x > 0, alpha * (exp(x) - 1) otherwise)
    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = (x.data[i] > 0) ? x.data[i] : 0.01 * (std::exp(x.data[i]) - 1);
        }
        return result;
    }

    // Backward method for computation of the gradient (derivative of the ELU function - 1 if x > 0, alpha * exp(x) otherwise)
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> gradInput = gradOutput;
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradInput.data[i] = (gradOutput.data[i] > 0) ? 1 : 0.01 * std::exp(gradOutput.data[i]);
        }
        return gradInput;
    }
};

template <typename T>
class ActivationFunction<T>::Tanh final : public ActivationFunction<T> {
public:
    // Forward method of the Tanh activation function (tanh(x))
    Tensor<T> forward(const Tensor<T>& x) override {
        Tensor<T> result(x.shape());
        for (size_t i = 0; i < x.data.size(); ++i) {
            result.data[i] = std::tanh(x.data[i]);
        }
        return result;
    }

    // Backward method for computation of the gradient (derivative of the Tanh function - 1 - tanh(x)^2)
    Tensor<T> backward(const Tensor<T>& gradOutput) override {
        Tensor<T> tanhOutput = forward(gradOutput);
        Tensor<T> gradInput = gradOutput;
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradInput.data[i] = 1 - tanhOutput.data[i] * tanhOutput.data[i];
        }
        return gradInput;
    }
};

#endif // ACTIVATIONFUNCTION_TPP
