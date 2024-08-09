#ifndef MULTIHEADATTENTIONMODEL_H
#define MULTIHEADATTENTIONMODEL_H

#pragma once
#include "../include/DenseLayer.h"
#include "../include/MultiHeadAttention.h"

template <typename T>
class MultiHeadAttentionModel {
public:
    MultiHeadAttentionModel(const int& hidden_dim, const int& num_heads, const int& head_dim);

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(Tensor<T>& grad_output);

    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    int hidden_dim{};
    int num_heads{};
    int head_dim{};

    std::vector<DenseLayer<T>> dense_layers;
    std::vector<MultiHeadAttention<T>> attention_layers;
};

#include "MultiHeadAttentionModel.tpp"

#endif //MULTIHEADATTENTIONMODEL_H
