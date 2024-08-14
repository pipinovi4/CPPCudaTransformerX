#ifndef MULTIHEADATTENTIONMODEL_TPP
#define MULTIHEADATTENTIONMODEL_TPP

#include "MultiHeadAttentionModel.h"

template <typename T>
MultiHeadAttentionModel<T>::MultiHeadAttentionModel(const int& max_sequence_length, const int& num_heads, const int& head_dim)
    : max_sequence_length(max_sequence_length), num_heads(num_heads), head_dim(head_dim), 
      multi_head_attention(max_sequence_length, num_heads, head_dim),
      input_layer(max_sequence_length, head_dim * head_dim, new typename ActivationFunction<T>::ReLU()),
      output_layer(max_sequence_length, head_dim * head_dim, new typename ActivationFunction<T>::ReLU()) {}

template <typename T>
Tensor<T> MultiHeadAttentionModel<T>::forward(const Tensor<T>& input_data) {
    Tensor<T> x = multi_head_attention.forward(input_data);
    x = input_layer.forward(x);
    return output_layer.forward(x);
}

template <typename T>
void MultiHeadAttentionModel<T>::backward(Tensor<T>& grad_output) {
    multi_head_attention.backward(grad_output);
    output_layer.backward(grad_output);
    input_layer.backward(grad_output);
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttentionModel<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> params;
    params.push_back(input_layer.weights);
    params.push_back(input_layer.bias);
    params.push_back(output_layer.weights);
    params.push_back(output_layer.bias);
    for (int i = 0; i < num_heads; i++) {
        auto attention_params = multi_head_attention.parameters();
        params.insert(params.end(), attention_params.begin(), attention_params.end());
    }
    return params;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttentionModel<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> params;
    params.push_back(input_layer.weightGradients);
    params.push_back(input_layer.biasGradients);
    params.push_back(output_layer.weightGradients);
    params.push_back(output_layer.biasGradients);
    for (int i = 0; i < num_heads; i++) {
        auto attention_params = multi_head_attention.gradients();
        params.insert(params.end(), attention_params.begin(), attention_params.end());
    }
    return params;
}

#endif //MULTIHEADATTENTIONMODEL_TPP
