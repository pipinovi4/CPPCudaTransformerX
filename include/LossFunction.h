#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#pragma once
#include "../include/Tensor.h"

template <typename T>
class LossFunction {
public:
    virtual ~LossFunction() = default;
    virtual T forward(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;
    virtual Tensor<T> backward(const Tensor<T>& predictions, const Tensor<T>& targets) = 0;

    class binaryCrossEntropyLoss;
    class crossEntropyLoss;
    class meanSquaredError;
    class meanAbsoluteError;
};

template<typename T>
class LossFunction<T>::binaryCrossEntropyLoss final : public LossFunction<T> {
public:
    T forward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
    Tensor<T> backward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

template<typename T>
class LossFunction<T>::crossEntropyLoss final : public LossFunction<T> {
public:
    T forward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
    Tensor<T> backward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

template<typename T>
class LossFunction<T>::meanSquaredError final : public LossFunction<T> {
public:
    T forward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
    Tensor<T> backward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

template<typename T>
class LossFunction<T>::meanAbsoluteError final : public LossFunction<T> {
public:
    T forward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
    Tensor<T> backward(const Tensor<T>& predictions, const Tensor<T>& targets) override;
};

#include "../src/LossFunction.tpp"

template class LossFunction<float>;
template class LossFunction<double>;

#endif //LOSSFUNCTION_H
