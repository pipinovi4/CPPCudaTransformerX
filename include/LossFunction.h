#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#pragma once
#include "../include/Tensor.h"

template <typename T>
class LossFunction {
public:
    static T binaryCrossEntropyLoss(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon = 1e-7);

    static T crossEntropyLoss(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon = 1e-7);

    static T meanSquaredError(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon = 1e-7);

    static T meanAbsoluteError(const Tensor<T>& predictions, const Tensor<T>& targets, double epsilon = 1e-7);
};

#include "../src/LossFunction.tpp"

template class LossFunction<float>;
template class LossFunction<double>;

#endif //LOSSFUNCTION_H
