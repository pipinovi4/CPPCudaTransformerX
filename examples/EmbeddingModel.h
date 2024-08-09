//
// Created by root on 8/6/24.
//

#ifndef EMBEDDINGMODEL_H
#define EMBEDDINGMODEL_H

#include "../include/Embedding.h"
#include "../include/DenseLayer.h"
#include <vector>

/**
 * @brief A class representing an EmbeddingModel.
 *
 * @tparam T Data type for the model.
 */
template <typename T>
class EmbeddingModel {
public:
    EmbeddingModel(const size_t& vocab_size, const int& embedding_dim, Optimizer<float>::LearningRateSchedule& lr_schedule,
                   std::function<void(Tensor<T>&)> init_func);

    Tensor<T> forward(Tensor<T>& input_data);
    void backward(Tensor<T>& grad_data);

    std::vector<std::reference_wrapper<Tensor<T>>> parameters();
    std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    size_t vocab_size;
    int embedding_dim;

    Embedding<T> embedding;
};

#include "EmbeddingModel.tpp"

#endif //EMBEDDINGMODEL_H
