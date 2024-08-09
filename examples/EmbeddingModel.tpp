#ifndef EMBEDDINGMODEL_TPP
#define EMBEDDINGMODEL_TPP

#include "EmbeddingModel.h"

/**
 * @brief Constructs an EmbeddingModel with specified parameters.
 *
 * @param vocab_size Size of the vocabulary.
 * @param embedding_dim Dimension of the embedding layer.
 * @param lr_schedule Learning rate scheduler for the model.
 * @param init_func Initialization function for the embeddings.
 */
template <typename T>
EmbeddingModel<T>::EmbeddingModel(const size_t& vocab_size, const int& embedding_dim, Optimizer<float>::LearningRateSchedule& lr_schedule,
    std::function<void(Tensor<T>&)> init_func)
    : vocab_size(vocab_size), embedding_dim(embedding_dim),
      embedding(vocab_size, embedding_dim, init_func, lr_schedule) {}

/**
 * @brief Performs the forward pass of the EmbeddingModel.
 *
 * @param input_data The input data tensor to be passed through the model.
 * @return Tensor<T> The output tensor after passing through the model layers.
 */
template <typename T>
Tensor<T> EmbeddingModel<T>::forward(Tensor<T>& input_data) {
    Tensor<T> output = embedding.forward(input_data);
    typename ActivationFunction<T>::Sigmoid().forward(output);  // Added typename
    return output;
}

/**
 * @brief Performs the backward pass of the EmbeddingModel, computing gradients.
 *
 * @param grad_data The gradient tensor from the loss function to be backpropagated.
 */
template <typename T>
void EmbeddingModel<T>::backward(Tensor<T>& grad_data) {
    embedding.backward(grad_data);
    typename ActivationFunction<T>::Sigmoid().backward(grad_data);  // Added typename
}

/**
 * @brief Retrieves the model parameters for optimization.
 *
 * @return std::vector<std::reference_wrapper<Tensor<T>>> A vector of references to the model's parameters.
 */
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> EmbeddingModel<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> param_refs;
    auto& weights = embedding.getWeights();
    param_refs.push_back(std::ref(weights));
    return param_refs;
}

/**
 * @brief Retrieves the model gradients for optimization.
 *
 * @return std::vector<std::reference_wrapper<Tensor<T>>> A vector of references to the model's gradients.
 */
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> EmbeddingModel<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grad_refs;
    auto& grads = embedding.getGrad();
    grad_refs.push_back(std::ref(grads));
    return grad_refs;
}

#endif // EMBEDDINGMODEL_TPP
