#ifndef MULTIHEADATTENTIONMODEL_H
#define MULTIHEADATTENTIONMODEL_H

#pragma once
#include "../include/DenseLayer.h"
#include "../include/Embedding.h"
#include "../include/MultiHeadAttention.h"

template <typename T>
class MultiHeadAttentionModel {
public:
    MultiHeadAttentionModel(const int& max_sequence_length, const int& num_heads, const int& head_dim);

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(Tensor<T>& grad_output);

    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    int max_sequence_length{};
    int num_heads{};
    int head_dim{};

    MultiHeadAttention<T> multi_head_attention;
    DenseLayer<T> input_layer;
    DenseLayer<T> output_layer;
};

#include "MultiHeadAttentionModel.tpp"

#endif //MULTIHEADATTENTIONMODEL_H
