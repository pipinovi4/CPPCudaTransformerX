#ifndef OPTIMIZER_TPP
#define OPTIMIZER_TPP

#include "../include/Optimizer.h"
#include "../include/Tensor.h"
#include <vector>
#include <cmath>
#include <fstream>

template <typename T>
void Optimizer<T>::initialize(std::vector<int> param_shape) {
    first_moment_vector = Tensor<T>::zeros(param_shape);
    second_moment_vector = Tensor<T>::zeros(param_shape);
}

template <typename T>
void Optimizer<T>::Adam(Tensor<T>& params, const Tensor<T>& grads) {
    if (first_moment_vector.shape() != grads.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }
    time_step += 1;
    first_moment_vector = (first_moment_vector * beta1) + (grads * (1 - beta1));
    second_moment_vector = (second_moment_vector * beta2) + (grads * grads * (1 - beta2));

    Tensor<T> first_moment_vector_hat = first_moment_vector / (1 - std::pow(beta1, time_step));
    Tensor<T> second_moment_vector_hat = second_moment_vector / (1 - std::pow(beta2, time_step));

    params -= (first_moment_vector_hat / (second_moment_vector_hat.sqrt() + epsilon)) * learning_rate;
}

template <typename T>
void Optimizer<T>::RMSprop(Tensor<T>& params, const Tensor<T>& grads) {
    if (second_moment_vector.shape() != grads.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }

    second_moment_vector = (second_moment_vector * beta2) + (grads * grads * (1 - beta2));
    Tensor<T> second_moment_vector_hat = second_moment_vector / (1 - std::pow(beta2, time_step));
    params -= (grads / (second_moment_vector_hat.sqrt() + epsilon)) * learning_rate;
}

template <typename T>
void Optimizer<T>::SGD(Tensor<T>& params, const Tensor<T>& grads) {
    if (first_moment_vector.shape() != grads.shape()) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }

    // Comute the liikahead position
    Tensor<T> lookahead_params = params - (first_moment_vector * beta1 * learning_rate);

    // Compute the gradient at the lookahead position
    Tensor<T> lookahead_grads = grads;

    // Update the first moment vector
    first_moment_vector = (first_moment_vector * beta1) + (lookahead_grads * learning_rate);

    // Update the parameters
    params -= first_moment_vector;
}

template <typename T>
void Optimizer<T>::reset() {
    time_step = 0;
    first_moment_vector = Tensor<T>({1});
    second_moment_vector = Tensor<T>({1});
}

template <typename T>
void Optimizer<T>::save_state(const std::string& filename) const {
    std::ofstream model_file(filename);
    if (model_file.is_open()) {
        model_file << "{";
        model_file << time_step << ", ";
        first_moment_vector.serialize(model_file);
        model_file << ", ";
        second_moment_vector.serialize(model_file);
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
    is >> time_step;
    is >> ch; // Read ','

    // Read first_moment_vector
    first_moment_vector.deserialize(is);

    // Read second_moment_vector
    second_moment_vector.deserialize(is);
    is >> ch; // Read '}'
}

template <typename T>
void Optimizer<T>::load_state(const std::string& filename) {
    std::ifstream model_file(filename);
    if (model_file.is_open()) {
        deserialize(model_file);
        model_file.close();
        first_moment_vector.print();
        second_moment_vector.print();
    } else {
        throw std::runtime_error("Unable to open file for reading");
    }
}

// template <typename T>
// void Optimizer<T>::apply_weight_decay(Tensor<T>& params, T weight_decay) {
//
// }

// template <typename T>
// void Optimizer<T>::clip_gradients(Tensor<T>& grads, T clip_value) {
//
// }

#endif // OPTIMIZER_TPP
