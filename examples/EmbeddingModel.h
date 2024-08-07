//
// Created by root on 8/6/24.
//

#ifndef EMBEDDINGMODEL_H
#define EMBEDDINGMODEL_H

#include "../include/Embedding.h"
#include "../include/DenseLayer.h"
#include <vector>

/**
 * @brief Constructs an Embedding layer with the specified dimensions and learning rate parameters.
 *
 * @param input_dim The size of the input dimension (vocabulary size).
 * @param output_dim The size of the output dimension (embedding vector size).
 * @param learning_rate The initial learning rate for updating the weights.
 * @param decay_rate The rate at which the learning rate decays.
 * @param decay_step The number of steps after which the learning rate decays.
 * @param init_func Optional custom initialization function for weights.
 */
template <typename T>
class EmbeddingModel {
public:
    EmbeddingModel(const int& hidden_dim, const int& output_dim, const size_t& vocab_size,
                   const int& embedding_dim, Optimizer<float>::LearningRateSchedule& lr_schedule,
                   std::function<void(Tensor<T>&)> init_func, const int& num_layers);

    Tensor<T> forward(Tensor<T>& input_data);
    void backward(Tensor<T>& grad_data);

    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    int hidden_dim;
    int output_dim;

    size_t vocab_size;
    int embedding_dim;

    Embedding<T> embedding;
    std::vector<DenseLayer<T>> layers;
};

#include "EmbeddingModel.tpp"

#endif //EMBEDDINGMODEL_H
