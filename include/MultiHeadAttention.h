#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#pragma once
#include "../include/Tensor.h"
#include <vector>
#include "ActivationFunction.h"

template <typename T>
class MultiHeadAttention {
public:
    // Constructor with default activation function
    MultiHeadAttention(const int& hidden_dim, const int& num_heads, const int& head_dim, ActivationFunction<T>* activation = nullptr);

    ~MultiHeadAttention() {
        delete activation;
    }

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(const Tensor<T>& grad_output);

    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

    std::vector<Tensor<T>> split_heads(const Tensor<T>& x) const;
    Tensor<T> concat_heads(const std::vector<Tensor<T>>& heads) const;

private:
    int hidden_dim;
    int num_heads;
    int head_dim;
    
    std::vector<Tensor<T>> queries_heads;
    std::vector<Tensor<T>> keys_heads;
    std::vector<Tensor<T>> values_heads;

    Tensor<T> W_q, W_k, W_v, W_o;
    Tensor<T> b_q, b_k, b_v, b_o;
    Tensor<T> grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor<T> grad_b_q, grad_b_k, grad_b_v, grad_b_o;

    std::vector<Tensor<T>> attention_heads;
    Tensor<T> input_cache;

    ActivationFunction<T>* activation;

    static void intitializeParameter(Tensor<T>& weights);
};

#include "../src/MultiHeadAttention.tpp"

#endif // MULTIHEADATTENTION_H
