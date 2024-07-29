#ifndef OPTIMIZER_TPP
#define OPTIMIZER_TPP

#include "../include/Optimizer.h"
#include "../include/Tensor.h"
#include <vector>
#include <cmath>
#include <fstream>

template <typename T>
class Optimizer<T>::LearningRateSchedule::StepDecaySchedule final : public Optimizer<T>::LearningRateSchedule {
public:
    StepDecaySchedule(T initial_learning_rate, T decay_rate, const size_t decay_steps)
        : initial_learning_rate(initial_learning_rate), decay_rate(decay_rate), decay_steps(decay_steps) {}

    T getLearningRate(size_t epoch) override {
        return initial_learning_rate * std::pow(decay_rate, static_cast<T>(epoch) / decay_steps);
    }
private:
    T initial_learning_rate;
    T decay_rate;
    size_t decay_steps;
};

template <typename T>
class Optimizer<T>::LearningRateSchedule::ExponentialDecaySchedule final : public Optimizer<T>::LearningRateSchedule {
public:
    ExponentialDecaySchedule(T initial_learning_rate, T decay_rate)
        : initial_learning_rate(initial_learning_rate), decay_rate(decay_rate) {}

    T getLearningRate(size_t epoch) override {
        return initial_learning_rate * std::pow(decay_rate, static_cast<T>(epoch));
    }
private:
    T initial_learning_rate;
    T decay_rate;
};

template <typename T>
class Optimizer<T>::Adam final : public Optimizer<T> {
public:
    Adam(T beta1, T beta2, T epsilon, T learning_rate, LearningRateSchedule& lr_schedule)
        : Optimizer<T>(learning_rate, lr_schedule) {}

    void initialize(std::vector<int> param_shape) override {
        first_moment_vector = Tensor<T>(param_shape);
        second_moment_vector = Tensor<T>(param_shape);
    }

    void update(Tensor<T>& params, const Tensor<T>& grads, const size_t epoch) override {
        this->updateLearningRate(epoch);
        for (size_t i = 0; i < params.size(); ++i) {
            first_moment_vector.data[i] = beta1 * first_moment_vector.data[i] + (1 - beta1) * grads.data[i];
            second_moment_vector.data[i] = beta2 * second_moment_vector.data[i] + (1 - beta2) * grads.data[i] * grads.data[i];

            T m_hat = first_moment_vector.data[i] / (1 - std::pow(beta1, epoch + 1));
            T v_hat = second_moment_vector.data[i] / (1 - std::pow(beta2, epoch + 1));

            if (std::isnan(m_hat) || std::isnan(v_hat) || std::isinf(m_hat) || std::isinf(v_hat)) {
                throw std::runtime_error("NaN or Inf detected in Adam optimizer update");
            }

            params.data[i] -= this->learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }

};

template <typename T>
class Optimizer<T>::RMSprop final : public Optimizer<T> {
public:
    explicit RMSprop(T learning_rate, T decay_rate, T epsilon, LearningRateSchedule& lr_schedule = nullptr)
        : Optimizer<T>(learning_rate, lr_schedule) {
        this->decay_rate = decay_rate;
        this->epsilon = epsilon;
    }

    void initialize(std::vector<int> param_shape) override {
        this->mean_squared_gradients = Tensor<T>(param_shape);
    }

    void update(Tensor<T>& params, const Tensor<T>& grads, const size_t epoch) override {
        this->updateLearningRate(epoch);
        for (size_t i = 0; i < params.size(); ++i) {
            mean_squared_gradients.data[i] = decay_rate * mean_squared_gradients.data[i] + (1 - decay_rate) * grads.data[i] * grads.data[i];

            T denom = std::sqrt(mean_squared_gradients.data[i]) + epsilon;
            if (std::isnan(denom) || std::isinf(denom)) {
                throw std::runtime_error("NaN or Inf detected in RMSprop optimizer update");
            }

            params.data[i] -= this->learning_rate * grads.data[i] / denom;
        }
    }
};

template <typename T>
class Optimizer<T>::SGD final : public Optimizer<T> {
public:
    explicit SGD(T learning_rate, LearningRateSchedule& lr_schedule = nullptr)
        : Optimizer<T>(learning_rate, lr_schedule) {}

    void initialize(std::vector<int> param_shape) override {
        first_moment_vector = Tensor<T>(param_shape);
        second_moment_vector = Tensor<T>(param_shape);
    }

    void update(Tensor<T>& params, const Tensor<T>& grads, const size_t epoch) override {
        this->updateLearningRate(epoch);
        for (size_t i = 0; i < params.size(); ++i) {
            params.data[i] -= this->learning_rate * grads.data[i];
        }
    }
};

template <typename T>
void Optimizer<T>::updateLearningRate(size_t epoch) {
    learning_rate = lr_schedule.getLearningRate(epoch);
}

template <typename T>
void Optimizer<T>::apply_weight_decay(Tensor<T>& params, T weight_decay) {
    params -= params * weight_decay;
}

template <typename T>
void Optimizer<T>::clip_gradients(Tensor<T>& grads, T clip_value) {
    for (T& grad : grads.data) {
        if (grad < -clip_value) grad = -clip_value;
        if (grad > clip_value) grad = clip_value;
    }
}

template <typename T>
void Optimizer<T>::save_state(const std::string& filename) const {
    std::ofstream model_file(filename);
    if (model_file.is_open()) {
        model_file << "{";
        model_file << this->time_step << ", ";
        this->first_moment_vector.serialize(model_file);
        model_file << ", ";
        this->second_moment_vector.serialize(model_file);
        model_file << "}";
        model_file.close();
    } else {
        throw std::runtime_error("Unable to open file for writing");
    }
}

template <typename T>
void Optimizer<T>::deserialize(std::istream& is) {
    char ch;
    is >> ch; // Read '{'
    is >> this->time_step;
    is >> ch; // Read ','

    // Read first_moment_vector
    this->first_moment_vector.deserialize(is);

    // Read second_moment_vector
    this->second_moment_vector.deserialize(is);
    is >> ch; // Read '}'
}

template <typename T>
void Optimizer<T>::load_state(const std::string& filename) {
    std::ifstream model_file(filename);
    if (model_file.is_open()) {
        this->deserialize(model_file);
        model_file.close();
        this->first_moment_vector.print();
        this->second_moment_vector.print();
    } else {
        throw std::runtime_error("Unable to open file for reading");
    }
}

#endif // OPTIMIZER_TPP