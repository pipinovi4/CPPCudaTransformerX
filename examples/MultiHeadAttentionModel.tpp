#ifndef MULTIHEADATTENTIONMODEL_TPP
#define MULTIHEADATTENTIONMODEL_TPP

#include "MultiHeadAttentionModel.h"

template <typename T>
MultiHeadAttentionModel<T>::MultiHeadAttentionModel(const int& hidden_dim, const int& num_heads, const int& head_dim)
    : hidden_dim(hidden_dim), num_heads(num_heads), head_dim(head_dim) {
    for (int i = 0; i < num_heads; i++) {
        dense_layers.push_back(DenseLayer<T>(hidden_dim, hidden_dim, new typename ActivationFunction<T>::ReLU()));
    }
    for (int i = 0; i < num_heads; i++) {
        attention_layers.push_back(MultiHeadAttention<T>(hidden_dim, num_heads, head_dim));
    }
}

template <typename T>
Tensor<T> MultiHeadAttentionModel<T>::forward(const Tensor<T>& input_data) {
    Tensor<T> x = input_data;
    for (int i = 0; i < num_heads; i++) {
        x = attention_layers[i].forward(x);
    }
    for (int i = 0; i < num_heads; i++) {
        x = dense_layers[i].forward(x);
    }
    return x;
}

template <typename T>
void MultiHeadAttentionModel<T>::backward(Tensor<T>& grad_output) {
    for (int i = num_heads - 1; i >= 0; i--) {
        dense_layers[i].backward(grad_output);
        attention_layers[i].backward(grad_output);
    }
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttentionModel<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> params;
    for (int i = 0; i < num_heads; i++) {
        params.push_back(dense_layers[i].weights);
        params.push_back(dense_layers[i].bias);
        auto attention_params = attention_layers[i].parameters();
        params.insert(params.end(), attention_params.begin(), attention_params.end());
    }
    return params;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttentionModel<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grads;
    for (int i = 0; i < num_heads; i++) {
        grads.push_back(dense_layers[i].weightGradients);
        grads.push_back(dense_layers[i].biasGradients);
        auto attention_grads = attention_layers[i].gradients();
        grads.insert(grads.end(), attention_grads.begin(), attention_grads.end());
    }
    return grads;
}

#endif //MULTIHEADATTENTIONMODEL_TPP
