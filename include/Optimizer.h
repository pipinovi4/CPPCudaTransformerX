#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#pragma once
#include "../include/Tensor.h"
#include "../include/LossFunction.h"
#include "../include/ActivationFunction.h"
#include <cmath>
#include <vector>
#include <fstream>

template <typename T>
class Optimizer {
public:
    explicit Optimizer(T learning_rate, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), first_moment_vector(Tensor<T>({1})), second_moment_vector(Tensor<T>({1})), time_step(0) {}

    void initialize(std::vector<int> param_shape);

    void Adam(Tensor<T>& params, const Tensor<T>& grads);

    void RMSprop(Tensor<T>& params, const Tensor<T>& grads);

    void SGD(Tensor<T>& params, const Tensor<T>& grads);

    void reset();

    void save_state(const std::string& filename) const;

    void deserialize(std::istream& is);

    void load_state(const std::string& filename);

    void apply_weight_decay(Tensor<T>& params, T weight_decay);

    void clip_gradients(Tensor<T>& grads, T clip_value);

private:
    T learning_rate;
    T beta1;
    T beta2;
    T epsilon;
    size_t time_step;
    Tensor<T> first_moment_vector;
    Tensor<T> second_moment_vector;
};

#include "../src/Optimizer.tpp"

#endif //OPTIMIZER_H