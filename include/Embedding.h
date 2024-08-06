#ifndef EMBEDDING_H
#define EMBEDDING_H

#pragma once
#include "../include/Tensor.h"
#include "../include/Optimizer.h"

template <typename T>
class Embedding {
public:
    Embedding(const int& input_dim, const int& output_dim, const T& learning_rate, const T& decay_rate, const size_t& decay_step, std::function<void(Tensor<T>&)> init_func = nullptr);

    Tensor<T> forward(const Tensor<T>& input_data);
    void backward(const Tensor<T>& grad_data);

    void initializeWeights();

    void update(const int& epoch);
    void zero_grad();
    void setWeights(const Tensor<T>& new_weights);

    Tensor<T> getWeights() const;
    Tensor<T> getGrad() const;

private:
    Optimizer<float>::LearningRateSchedule::StepDecaySchedule learning_rate_scheduler;

    Tensor<T> weights;
    Tensor<T> grad;

    Tensor<T> input;
    Tensor<T> output;

    int input_dim;
    int output_dim;
};

#include "../src/Embedding.tpp"

#endif // EMBEDDING_H