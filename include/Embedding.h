#ifndef EMBEDDING_H
#define EMBEDDING_H

#pragma once
#include "../include/Tensor.h"
#include "../include/Optimizer.h"

template <typename T>
class Embedding {
public:
    Embedding(const int& vocab_size, const int& embedding_dims, std::function<void(Tensor<T>&)> init_func,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule);

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(const Tensor<T>& grad_data);

    void initializeWeights();

    void update(const int& epoch);
    void zero_grad();
    void setWeights(const Tensor<T>& new_weights);

    Tensor<T>& getWeights();
    Tensor<T>& getGrad();

private:
    Optimizer<float>::LearningRateSchedule& lr_schedule;

    Tensor<T> weights;
    Tensor<T> grad;

    Tensor<T> input;
    Tensor<T> output;

    int vocab_size{};
    int embedding_dims{};

    Tensor<T> input_cache;
};

#include "../src/Embedding.tpp"

#endif // EMBEDDING_H