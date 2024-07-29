#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#pragma once
#include "../include/Tensor.h"
#include <vector>

template <typename T>
class Optimizer {
public:
    class LearningRateSchedule {
    public:
        virtual T getLearningRate(size_t epoch) = 0;
        virtual ~LearningRateSchedule() = default;

        class StepDecaySchedule;
        class ExponentialDecaySchedule;
    };

    explicit Optimizer(T learning_rate, LearningRateSchedule& lr_schedule = nullptr)
     : beta1(T(0.9)), beta2(T(0.999)), epsilon(T(1e-8)), learning_rate(learning_rate), lr_schedule(lr_schedule) {}

    virtual void initialize(std::vector<int> param_shape) = 0;
    virtual void update(Tensor<T>& params, const Tensor<T>& grads, size_t epoch) = 0;
    virtual ~Optimizer() = default;

    class Adam;
    class RMSprop;
    class SGD;

protected:
    T beta1;
    T beta2;
    T epsilon;

    size_t time_step = 0;
    Tensor<T> first_moment_vector;
    Tensor<T> second_moment_vector;
    Tensor<T> mean_squared_gradients;

    T learning_rate;
    LearningRateSchedule& lr_schedule;
    T decay_rate;

    void updateLearningRate(size_t epoch);

    static void apply_weight_decay(Tensor<T>& params, T weight_decay);

    void clip_gradients(Tensor<T>& grads, T clip_value);

    void save_state(const std::string& filename) const;

    void load_state(const std::string& filename);

    void deserialize(std::istream& is);
};

#include "../src/Optimizer.tpp"

#endif // OPTIMIZER_H