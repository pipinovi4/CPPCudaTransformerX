#ifndef LAYERNORM_H
#define LAYERNORM_H

#pragma once
#include "Tensor.h"

template <typename T>
class LayerNorm {
public:
    explicit LayerNorm(int d_model, float epsilon = 1e-6);

    Tensor<T> forward(const Tensor<T>& x);
private:
    int d_model_;
    float epsilon_;
    Tensor<T> gamma_;
    Tensor<T> beta_;
};

#include "../src/LayerNorm.tpp"

#endif //LAYERNORM_H
