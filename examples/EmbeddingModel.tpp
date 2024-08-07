#ifndef EMBEDDINGMODEL_TPP
#define EMBEDDINGMODEL_TPP

#include "EmbeddingModel.h"

template <typename T>
EmbeddingModel<T>::EmbeddingModel(const int& hidden_dim, const int& output_dim, const size_t& vocab_size,
    const int& embedding_dim, Optimizer<float>::LearningRateSchedule& lr_schedule,
    std::function<void(Tensor<T>&)> init_func, const int& num_layers)
    : hidden_dim(hidden_dim), output_dim(output_dim),
      vocab_size(vocab_size), embedding_dim(embedding_dim),
      embedding(vocab_size, embedding_dim, init_func, lr_schedule) {
    // Initialize the input layer
    layers.push_back(DenseLayer<T>(embedding_dim, hidden_dim, new typename ActivationFunction<T>::ReLU() , 0.0));

    // Initialize the hidden layers
    for (int i = 0; i < num_layers - 2; ++i) {
        layers.push_back(DenseLayer<T>(hidden_dim, hidden_dim, new typename ActivationFunction<T>::ReLU(), 0.0));
    }

    // Initialize the output layer
    layers.push_back(DenseLayer<T>(hidden_dim, output_dim, new typename ActivationFunction<T>::Softmax(), 0.0));
}

template <typename T>
Tensor<T> EmbeddingModel<T>::forward(Tensor<T>& input_data) {
    Tensor<T> embedding_output = embedding.forward(input_data);
    Tensor<T> output = layers[0].forward(embedding_output);

    for (int i = 1; i < layers.size(); ++i) {
        output = layers[i].forward(output);
    }

    return output;
}

template <typename T>
void EmbeddingModel<T>::backward(Tensor<T>& grad_data) {
    layers[layers.size() - 1].backward(grad_data);

    for (int i = layers.size() - 2; i >= 0; --i) {
       layers[i].backward(grad_data);
    }

    embedding.backward(grad_data);
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> EmbeddingModel<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> param_refs;
    for (auto& layer : layers) {
        param_refs.push_back(std::ref(layer.weights));
        param_refs.push_back(std::ref(layer.bias));
    }
    return param_refs;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> EmbeddingModel<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grad_refs;
    for (auto& layer : layers) {
        grad_refs.push_back(std::ref(layer.weightGradients));
        grad_refs.push_back(std::ref(layer.biasGradients));
    }
    return grad_refs;
}

#endif // EMBEDDINGMODEL_TPP