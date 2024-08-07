#ifndef EMBEDDING_H
#define EMBEDDING_H

#pragma once
#include "../include/Tensor.h"
#include "../include/Optimizer.h"

template <typename T>
class Embedding {
public:
    Embedding(const int& input_dim, const int& output_dim, std::function<void(Tensor<T>&)> init_func,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule);

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(const Tensor<T>& grad_data);

    void initializeWeights();

    void update(const int& epoch);
    void zero_grad();
    void setWeights(const Tensor<T>& new_weights);

    Tensor<T> getWeights();
    Tensor<T> getGrad();

private:
    Optimizer<float>::LearningRateSchedule& lr_schedule;

    Tensor<T> weights;
    Tensor<T> grad;

    Tensor<T> input;
    Tensor<T> output;

    int input_dim;
    int output_dim;
};

#include "../src/Embedding.tpp"

#endif // EMBEDDING_H