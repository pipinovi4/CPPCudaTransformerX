#ifndef ACTIVATIONFUNCTION_TPP
#define ACTIVATIONFUNCTION_TPP

#include <cmath>
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"

template <typename T>
class ActivationFunction<T>::Sigmoid final : public ActivationFunction<T> {
public:
    // Forward method of the Sigmoid activation function (1 / (1 + exp(-x))
    void forward(Tensor<T>& x) override {
        for (auto& val : x.data) {
            val = 1.0 / (1.0 + std::exp(-val));
        }
    }

    // Backward method for computation of the gradient (derivative of the sigmoid function - sigmoid(x) * (1 - sigmoid(x))
    void backward(Tensor<T>& gradOutput) override {
        Tensor<float> sigmoidOutput = gradOutput;
        forward(sigmoidOutput);
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradOutput.data[i] = gradOutput.data[i] * sigmoidOutput.data[i] * (1 - sigmoidOutput.data[i]);
        }
    }
};

template <typename T>
class ActivationFunction<T>::Softmax final : public ActivationFunction<T> {
public:
    // Forward method of the Softmax activation function (exp(x_i) / sum(exp(x_j)))
    void forward(Tensor<T>& x) override {
        const auto& shape = x.shape();
        size_t outerSize = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            outerSize *= shape[i];
        }
        const size_t innerSize = shape.back();

        for (size_t i = 0; i < outerSize; ++i) {
            T* x_slice = &x.data[i * innerSize];
            T maxVal = *std::max_element(x_slice, x_slice + innerSize);
            T sum = 0.0;
            for (size_t j = 0; j < innerSize; ++j) {
                x_slice[j] = std::exp(x_slice[j] - maxVal);
                sum += x_slice[j];
            }
            for (size_t j = 0; j < innerSize; ++j) {
                x_slice[j] /= sum;
            }
        }
    }

    // Backward method for computation of the gradient
    void backward(Tensor<T>& gradOutput) override {
        Tensor<T> softmax_output = gradOutput; // Assume softmax_output has the same structure
        forward(softmax_output);
        const auto& shape = gradOutput.shape();
        size_t outerSize = 1;
        for (size_t i = 0; i < shape.size() - 1; ++i) {
            outerSize *= shape[i];
        }
        const size_t innerSize = shape.back();

        for (size_t i = 0; i < outerSize; ++i) {
            T* gradOutput_i = &gradOutput.data[i * innerSize];
            T* softmaxOutput_i = &softmax_output.data[i * innerSize];
            T dot_product = 0;
            for (size_t j = 0; j < innerSize; ++j) {
                dot_product += softmaxOutput_i[j] * gradOutput_i[j];
            }
            for (size_t j = 0; j < innerSize; ++j) {
                gradOutput_i[j] = softmaxOutput_i[j] * (gradOutput_i[j] - dot_product);
            }
        }
    }
};

template <typename T>
class ActivationFunction<T>::ReLU final : public ActivationFunction<T> {
public:
    // Forward method of the ReLU activation function (max(0, x))
    void forward(Tensor<T>& x) override {
        input_cache = x;
        for (size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] = std::max(static_cast<T>(0), x.data[i]);
        }
    }

    // Backward method for computation of the gradient (derivative of the ReLU function)
    void backward(Tensor<T>& gradOutput) override {
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradOutput.data[i];
            gradOutput.data[i] *= (input_cache.data[i] > 0) ? 1 : 0;
        }
    }

private:
    Tensor<T> input_cache;
};


template <typename T>
class ActivationFunction<T>::LeakyReLU final : public ActivationFunction<T> {
public:
    // Forward method of the Leaky ReLU activation function (x if x > 0, alpha * x otherwise)
    void forward(Tensor<T>& x) override {
        for (size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] = (x.data[i] > 0) ? x.data[i] : 0.01 * x.data[i];
        }
    }

    // Backward method for computation of the gradient (derivative of the Leaky ReLU function - 1 if x > 0, alpha otherwise)
    void backward(Tensor<T>& gradOutput) override {
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradOutput.data[i] = (gradOutput.data[i] > 0) ? 1 : 0.01;
        }
    }
};

template <typename T>
class ActivationFunction<T>::ELU final : public ActivationFunction<T> {
public:
    explicit ELU(T alpha = 1.0) : alpha(alpha) {}

    void forward(Tensor<T>& x) override {
        for (size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] = (x.data[i] > 0) ? x.data[i] : alpha * (std::exp(x.data[i]) - 1);
        }
    }

    void backward(Tensor<T>& gradOutput) override {
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradOutput.data[i] = (gradOutput.data[i] > 0) ? 1 : alpha * std::exp(gradOutput.data[i]);
        }
    }

private:
    T alpha;
};

template <typename T>
class ActivationFunction<T>::Tanh final : public ActivationFunction<T> {
public:
    // Forward method of the Tanh activation function (tanh(x))
    void forward(Tensor<T>& x) override {
        for (size_t i = 0; i < x.data.size(); ++i) {
            x.data[i] = std::tanh(x.data[i]);
        }
    }

    // Backward method for computation of the gradient (derivative of the Tanh function - 1 - tanh(x)^2)
    void backward(Tensor<T>& gradOutput) override {
        Tensor<T> tanhOutput = gradOutput;
        forward(tanhOutput);
        for (size_t i = 0; i < gradOutput.data.size(); ++i) {
            gradOutput.data[i] = 1 - tanhOutput.data[i] * tanhOutput.data[i];
        }
    }
};

template <typename T>
class ActivationFunction<T>::Linear final : public ActivationFunction<T> {
public:
    // The forward pass just returns the input unchanged
    void forward(Tensor<T>& input) override {
        // Just pass the input through unchanged
    }

    // The backward pass returns the gradient unchanged
    void backward(Tensor<T>& grad_output) override {
        // Just pass the gradient through unchanged
    }
};



#endif // ACTIVATIONFUNCTION_TPP
