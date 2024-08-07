#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#pragma once
#include "../include/Tensor.h"

class MultiHeadAttention {
public:
    MultiHeadAttention(const int& hidden_dim, const int& num_heads, const int& head_dim);

    Tensor<float> forward(const Tensor<float>& input_data);
    void backward(const Tensor<float>& grad_data);
};

#include "../src/MultiHeadAttention.tpp"

#endif // MULTIHEADATTENTION_H