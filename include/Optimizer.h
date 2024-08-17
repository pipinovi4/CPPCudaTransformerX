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

    explicit Optimizer(T beta1, T beta2, T epsilon, T learning_rate, LearningRateSchedule& lr_schedule)
     : beta1(beta1), beta2(beta2), epsilon(epsilon), learning_rate(learning_rate), lr_schedule(lr_schedule) {}

    virtual void initialize_params(const std::vector<std::vector<int>>& param_shape) = 0;
    virtual void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params, const std::vector<std::reference_wrapper<Tensor<T>>>& grads, const size_t& epoch) = 0;
    virtual ~Optimizer() = default;

    class Adam;
    class RMSprop;
    class SGD;

protected:
    T beta1;
    T beta2;
    T epsilon;

    size_t time_step = 0;
    std::vector<Tensor<T>> first_moment_vector;
    std::vector<Tensor<T>> second_moment_vector;
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