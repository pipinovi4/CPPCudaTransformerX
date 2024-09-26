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
    }

    if (value > max_value) {
        return max_value;
    }

    return value;
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
    Adam(std::vector<std::reference_wrapper<Tensor<T>>> parameters, T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
            : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {
        initialize_params(parameters);
    }

    explicit Adam(T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {}

    void initialize_params(std::vector<std::reference_wrapper<Tensor<T>>> parameters) override {
        for (auto parameter : parameters) {
            this->first_moment_vector_.push_back(parameter.get());  // First moment vector
            this->second_moment_vector_.push_back(parameter.get()); // Second moment vector
        }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        // Update learning rate based on the current epoch
        this->updateLearningRate(epoch);
        //
        // Clip gradients to avoid exploding gradients
        for (auto& grad : grads) {
            clip_gradients(grad.get(), 1.0);  // Clip before updating parameters
        }

        // Update parameters
        T beta1_pow_t = std::pow(beta1_, time_step_);
        T beta2_pow_t = std::pow(beta2_, time_step_);
        T inv_beta1 = 1 - beta1_;
        T inv_beta2 = 1 - beta2_;

        // Update parameters with Adam
        for (size_t i = 0; i < first_moment_vector_.size(); ++i) {
            // Get references to the data
            auto& fm = first_moment_vector_[i].data;
            auto& sm = second_moment_vector_[i].data;
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;

            if (fm.size() != sm.size() || sm.size() != p.size() || p.size() != g.size()) {
                std::cerr << "Error in Adam Optimizer: Size mismatch detected between tensors during the update step." << std::endl;
                return;
            }

            // Pointer to raw data
            auto fm_current = fm.data();
            auto sm_current = sm.data();
            auto g_current = g.data();
            auto p_current = p.data();

            for (size_t j = 0; j < p.size(); ++j) {
                //
                // Perform the updates for first and second moment estimates
                fm_current[j] = beta1_ * fm_current[j] + inv_beta1 * g_current[j];
                sm_current[j] = beta2_ * sm_current[j] + inv_beta2 * std::pow(g_current[j], 2);

                // Compute bias-corrected m_hat and v_hat
                T m_hat = fm_current[j] / (1 - beta1_pow_t);
                // if (fm_current[j] < 0) {
                //     std::cerr << "Wictory: fm_current[j] is negative." << std::endl;
                // }
                T v_hat = std::max(sm_current[j] / (1 - beta2_pow_t), static_cast<T>(0.0));

                // Update the parameters with Adam step
                p_current[j] -= this->learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);

                // Apply weight decay (L2 regularization)
                p_current[j] -= weight_decay_ * p_current[j];

                // If is nan (std::sqrt(v_hat) + epsilon_) that print them
                // if (p_current[j] < 0) {
                //     std::cerr << "Wictory: p_current[j] is negative." << std::endl;
                // }
            }
            // // Print statistics about m_hat and p_current
            // T min_m_hat = *std::min_element(fm.begin(), fm.end());
            // T max_m_hat = *std::max_element(fm.begin(), fm.end());
            // T min_p = *std::min_element(p.begin(), p.end());
            // T max_p = *std::max_element(p.begin(), p.end());
            //
            // std::cout << "m_hat: Min = " << min_m_hat << ", Max = " << max_m_hat << std::endl;
            // std::cout << "p_current: Min = " << min_p << ", Max = " << max_p << std::endl;
        }

        // Increment time step for future use
        this->time_step_ += 1;
    }
};

template <typename T>
class Optimizer<T>::RMSprop final : public Optimizer<T> {
public:
    RMSprop(std::vector<std::reference_wrapper<Tensor<T>>> parameters, T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {
        initialize_params(parameters);
    }

    explicit RMSprop(T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
    : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {}

    void initialize_params(std::vector<std::reference_wrapper<Tensor<T>>> parameters) override {
        for (auto parameter : parameters) {
            this->first_moment_vector_.push_back(parameter.get());  // First moment vector
            this->second_moment_vector_.push_back(parameter.get()); // Second moment vector
        }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        // Update learning rate based on the current epoch
        this->updateLearningRate(epoch);

        // Clip gradients to avoid exploding gradients
        for (auto& grad : grads) {
            clip_gradients(grad.get(), 1.0);  // Clip before updating parameters
        }

        // Update parameters
        for (size_t i = 0; i < params.size(); ++i) {
            // Get references to the data
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;
            auto& sm = second_moment_vector_[i].data; // Mean squared gradients

            // Update the parameters
            for (size_t j = 0; j < p.size(); ++j) {
                // Get references to the current values
                T& sm_current = sm[j];
                T& g_current = g[j];

                // Update the mean squared gradients
                sm_current = weight_decay_ * sm_current + (1 - weight_decay_) * std::pow(g_current, 2);

                // Update the parameters
                p[j] -= this->learning_rate_ * g_current / (std::sqrt(sm_current) + epsilon_);

                // Apply weight decay (L2 regularization)
                p[j] -= weight_decay_ * p[j];
            }
        }

        // Increment time step for future use
        this->time_step_ += 1;
    }
};

template <typename T>
class Optimizer<T>::SGD final : public Optimizer<T> {
public:
    SGD(std::vector<std::reference_wrapper<Tensor<T>>> parameters, T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {
        initialize_params(parameters);
    }

    explicit SGD(T learning_rate, LearningRateSchedule& lr_schedule, T weight_decay = 0.0, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8)
        : Optimizer<T>(learning_rate, lr_schedule, weight_decay, beta1, beta2, epsilon) {}

    void initialize_params(std::vector<std::reference_wrapper<Tensor<T>>> parameters) override {
        // Initialize moment vectors (if necessary)
        // In SGD, we typically don't need these, but we'll follow the structure
        // for (auto parameter : parameters) {
        //      this->first_moment_vector_.push_back(parameter.get());  // First moment vector
        //      this->second_moment_vector_.push_back(parameter.get()); // Second moment vector
        // }
    }

    void update(const std::vector<std::reference_wrapper<Tensor<T>>>& params,
                const std::vector<std::reference_wrapper<Tensor<T>>>& grads,
                const size_t& epoch) override {
        // Update learning rate based on the current epoch
        this->updateLearningRate(epoch);

        // Clip gradients to avoid exploding gradients
        for (auto& grad : grads) {
            clip_gradients(grad.get(), 1.0);  // Clip before updating parameters
        }

        for (size_t i = 0; i < params.size(); ++i) {
            auto& p = params[i].get().data;
            auto& g = grads[i].get().data;

            for (size_t j = 0; j < p.size(); ++j) {
                // Update the parameters
                p[j] -= this->learning_rate_ * g[j];

                // Apply weight decay (L2 regularization)
                // Apply weight decay (L2 regularization)
                p[j] -= weight_decay_ * p[j];
            }
        }

        // Increment time step for future use
        this->time_step_ += 1;
    }
};

template <typename T>
void Optimizer<T>::updateLearningRate(size_t epoch) {
    learning_rate_ = lr_schedule_.getLearningRate(epoch);
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
        model_file << this->time_step_ << ", ";
        this->first_moment_vector_.serialize(model_file);
        model_file << ", ";
        this->second_moment_vector_.serialize(model_file);
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
    is >> this->time_step_;
    is >> ch; // Read ','

    // Read first_moment_vector
    this->first_moment_vector_.deserialize(is);

    // Read second_moment_vector
    this->second_moment_vector_.deserialize(is);
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