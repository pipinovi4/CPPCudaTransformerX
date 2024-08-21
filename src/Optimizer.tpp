#ifndef OPTIMIZER_TPP
#define OPTIMIZER_TPP

#include "../include/Optimizer.h"
#include "../include/Tensor.h"
#include <vector>
#include <cmath>
#include <fstream>

template <typename T>
T clamp_value(T value, T min_value, T max_value) {
    if (value < min_value) {
        return min_value;
    } else if (value > max_value) {
        return max_value;
    } else {
        return value;
    }
}

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
    Adam(const std::vector<std::vector<int>>& param_shape, T learning_rate, LearningRateSchedule& lr_schedule, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : Optimizer<T>(beta1, beta2, epsilon, learning_rate, lr_schedule) {
        initialize_params(param_shape);
    }

    explicit Adam(LearningRateSchedule& lr_schedule, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, T learning_rate = 0.001)
        : Optimizer<T>(beta1, beta2, epsilon, learning_rate, lr_schedule) {}

    void initialize_params(const std::vector<std::vector<int>>& param_shape) override {
        for (const auto& shape : param_shape) {
            this->first_moment_vector.push_back(Tensor<T>(shape));
            this->second_moment_vector.push_back(Tensor<T>(shape));
        }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        this->updateLearningRate(epoch);

        T beta1_pow_epoch = std::pow(beta1, epoch + 1);
        T beta2_pow_epoch = std::pow(beta2, epoch + 1);
        T inv_beta1 = 1 - beta1;
        T inv_beta2 = 1 - beta2;

        for (size_t i = 0; i < first_moment_vector.size(); ++i) {
            auto& fm = first_moment_vector[i].data;
            auto& sm = second_moment_vector[i].data;
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;

            if (fm.size() != sm.size() || sm.size() != p.size() || p.size() != g.size()) {
                std::cerr << "Size mismatch between vectors in method update of the Optimizer!" << std::endl;
                std::cerr << "First moment: " << fm.size() << " | ";
                std::cerr << "Second moment: " << sm.size() << std::endl;
                std::cerr << "Parameters: " << p.size() << " | ";
                std::cerr << "Gradients: " << g.size() << std::endl;
                return;
            }

            // Pointer to raw data
            auto fm_current = fm.data();
            auto sm_current = sm.data();
            auto g_current = g.data();
            auto p_current = p.data();

            for (size_t j = 0; j < p.size(); ++j) {
                // Perform the updates
                fm_current[j] = beta1 * fm_current[j] + inv_beta1 * g_current[j];
                sm_current[j] = beta2 * sm_current[j] + inv_beta2 * std::pow(g_current[j], 2);

                // Compute m_hat and v_hat
                T m_hat = fm_current[j] / (1 - beta1_pow_epoch);
                T v_hat = sm_current[j] / (1 - beta2_pow_epoch);

                // Apply the update to parameters
                p[j] -= this->learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    }
};

template <typename T>
class Optimizer<T>::RMSprop final : public Optimizer<T> {
public:
    RMSprop(const std::vector<std::vector<int>>& param_shape, T learning_rate, T decay_rate, T epsilon, LearningRateSchedule& lr_schedule)
        : Optimizer<T>(0.9, 0.999, epsilon, learning_rate, lr_schedule), decay_rate(decay_rate), epsilon(epsilon) {
        initialize_params(param_shape);
    }

    explicit RMSprop(LearningRateSchedule& lr_schedule, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, T learning_rate = 0.001)
    : Optimizer<T>(beta1, beta2, epsilon, learning_rate, lr_schedule) {}

    void initialize_params(const std::vector<std::vector<int>>& param_shape) override {
        for (const auto& shape : param_shape) {
            this->first_moment_vector.push_back(Tensor<T>(shape));  // Unused in RMSprop
            this->second_moment_vector.push_back(Tensor<T>(shape)); // Used to store mean squared gradients
        }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        this->updateLearningRate(epoch);

        for (size_t i = 0; i < params.size(); ++i) {
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;
            auto& sm = second_moment_vector[i].data; // Mean squared gradients

            for (size_t j = 0; j < p.size(); ++j) {
                T& sm_current = sm[j];
                T& g_current = g[j];

                sm_current = decay_rate * sm_current + (1 - decay_rate) * std::pow(g_current, 2);

                p[j] -= this->learning_rate * g_current / (std::sqrt(sm_current) + epsilon);
            }
        }
    }

private:
    T decay_rate;
    T epsilon;
};

template <typename T>
class Optimizer<T>::SGD final : public Optimizer<T> {
public:
    SGD(const std::vector<std::vector<int>>& param_shape, T learning_rate, LearningRateSchedule& lr_schedule)
        : Optimizer<T>(0.0, 0.0, 0.0, learning_rate, lr_schedule) {
        initialize_params(param_shape);
    }

    explicit SGD(LearningRateSchedule& lr_schedule, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8, T learning_rate = 0.001)
    : Optimizer<T>(beta1, beta2, epsilon, learning_rate, lr_schedule) {}

    void initialize_params(const std::vector<std::vector<int>>& param_shape) override {
        // Initialize moment vectors (if necessary)
        // In SGD, we typically don't need these, but we'll follow the structure
        for (const auto& shape : param_shape) {
            this->first_moment_vector.push_back(Tensor<T>(shape));  // Unused in SGD
            this->second_moment_vector.push_back(Tensor<T>(shape)); // Unused in SGD
        }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        this->updateLearningRate(epoch);

        for (size_t i = 0; i < params.size(); ++i) {
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;

            for (size_t j = 0; j < p.size(); ++j) {
                p[j] -= this->learning_rate * g[j];
            }
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
    } else {
        throw std::runtime_error("Unable to open file for reading");
    }
}

#endif // OPTIMIZER_TPP