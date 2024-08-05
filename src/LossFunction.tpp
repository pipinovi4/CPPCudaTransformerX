#ifndef LOSSFUNCTION_TPP
#define LOSSFUNCTION_TPP

#include <cmath>
#include "../include/Tensor.h"
#include "../include/LossFunction.h"

template<typename T>
constexpr T clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

template<typename T>
T LossFunction<T>::binaryCrossEntropyLoss::forward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    T loss = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss += -targets.data[i] * std::log(clamp(predictions.data[i], static_cast<T>(1e-15), static_cast<T>(1.0 - 1e-15)))
            - (1 - targets.data[i]) * std::log(clamp(1 - predictions.data[i], static_cast<T>(1e-15), static_cast<T>(1.0 - 1e-15)));
    }
    return loss / predictions.shape()[0];
}

template<typename T>
Tensor<T> LossFunction<T>::binaryCrossEntropyLoss::backward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    Tensor<T> grad_output(predictions.shape());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad_output.data[i] = (predictions.data[i] - targets.data[i]) / (predictions.data[i] * (1 - predictions.data[i]));
    }
    return grad_output;
}

template<typename T>
T LossFunction<T>::crossEntropyLoss::forward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    T loss = 0;
    const T epsilon = static_cast<T>(1e-15);

    for (size_t i = 0; i < predictions.size(); ++i) {
        // Clamp the prediction value to avoid log(0) or log(negative)
        T pred_value = std::max(epsilon, std::min(predictions.data[i], static_cast<T>(1.0) - epsilon));
        
        // Calculate the cross-entropy loss
        loss += -targets.data[i] * std::log(pred_value);
    }

    // Normalize loss by the number of samples
    size_t num_samples = predictions.shape()[0];
    return loss / static_cast<T>(num_samples);
}

template<typename T>
Tensor<T> LossFunction<T>::crossEntropyLoss::backward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    Tensor<T> grad_output(predictions.shape());
    const T epsilon = static_cast<T>(1e-15);
    const T one_minus_epsilon = static_cast<T>(1.0) - epsilon;

    for (size_t i = 0; i < predictions.size(); ++i) {
        T clamped_pred = clamp(predictions.data[i], epsilon, one_minus_epsilon);
        grad_output.data[i] = (clamped_pred - targets.data[i]) / predictions.shape()[0];
    }

    return grad_output;
}

template<typename T>
T LossFunction<T>::meanSquaredError::forward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    T loss = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss += std::pow(predictions.data[i] - targets.data[i], 2);
    }
    return loss / predictions.shape()[0];
}

template<typename T>
Tensor<T> LossFunction<T>::meanSquaredError::backward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    Tensor<T> grad_output(predictions.shape());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad_output.data[i] = 2 * (predictions.data[i] - targets.data[i]);
    }
    return grad_output;
}

template<typename T>
T LossFunction<T>::meanAbsoluteError::forward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    T loss = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        loss += std::abs(predictions.data[i] - targets.data[i]);
    }
    return loss / predictions.shape()[0];
}

template<typename T>
Tensor<T> LossFunction<T>::meanAbsoluteError::backward(const Tensor<T>& predictions, const Tensor<T>& targets) {
    Tensor<T> grad_output(predictions.shape());
    for (size_t i = 0; i < predictions.size(); ++i) {
        grad_output.data[i] = (predictions.data[i] > targets.data[i]) ? 1 : -1;
    }
    return grad_output;
}

#endif // LOSSFUNCTION_TPP