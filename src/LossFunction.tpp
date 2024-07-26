#ifndef LOSSFUNCTION_TPP
#define LOSSFUNCTION_TPP

#include <cmath>
#include "../include/Tensor.h"

template<typename T>
constexpr T clamp(const T& value, const T& low, const T& high) {
    return (value < low) ? low : ((value > high) ? high : value);
}

template <typename T>
T LossFunction<T>::binaryCrossEntropyLoss(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon) {
    if (targets.size() != predictions.size()) {
        throw std::invalid_argument("Size of y_true and y_pred must be the same.");
    }

    T loss = 0.0;

    for (size_t i = 0; i < targets.size(); ++i) {
        const T clipped_pred = clamp(predictions.data[i], static_cast<T>(epsilon), static_cast<T>(1 - epsilon));
        const T logits = std::log(clipped_pred / (1 - clipped_pred));
        const T max_logit = std::max(logits, static_cast<T>(0));
        loss += logits - logits * targets.data[i] + max_logit + std::log(std::exp(-max_logit) + std::exp(-logits - max_logit));
    }

    return loss / targets.size();
}

template <typename T>
T LossFunction<T>::crossEntropyLoss(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon) {
    // Check if the shapes of predictions and targets are the same
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Shapes of predictions and targets are not the same!");
    }

    T loss = 0.0;
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        const T clipped_pred = clamp(predictions.data[i], static_cast<T>(epsilon), static_cast<T>(1 - epsilon));
        loss -= targets.data[i] * std::log(clipped_pred);
    }

    return loss;
}

template <typename T>
T LossFunction<T>::meanSquaredError(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon) {
    // Check if the shapes of predictions and targets are the same
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Shapes of predictions and targets are not the same!");
    }

    T loss = 0.0;
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        const T clipped_pred = clamp(predictions.data[i], static_cast<T>(epsilon), static_cast<T>(1 - epsilon));
        loss += std::pow(clipped_pred - targets.data[i], 2);
    }

    return loss / static_cast<T>(predictions.data.size());
}

template <typename T>
T LossFunction<T>::meanAbsoluteError(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon) {
    // Check if the shapes of predictions and targets are the same
    if (predictions.shape() != targets.shape()) {
        throw std::runtime_error("Shapes of predictions and targets are not the same!");
    }

    T loss = 0.0;
    for (size_t i = 0; i < predictions.data.size(); ++i) {
        loss += std::abs(predictions.data[i] - targets.data[i]);
    }

    return loss / static_cast<T>(predictions.data.size());
}

#endif // LOSSFUNCTION_TPP
