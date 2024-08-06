#ifndef EMBEDDING_TPP
#define EMBEDDING_TPP

#include "../include/Embedding.h"
#include <stdexcept>

/**
 * @brief Constructs an Embedding layer with the specified dimensions and learning rate parameters.
 *
 * @param input_dim The size of the input dimension (vocabulary size).
 * @param output_dim The size of the output dimension (embedding vector size).
 * @param learning_rate The initial learning rate for updating the weights.
 * @param decay_rate The rate at which the learning rate decays.
 * @param decay_step The number of steps after which the learning rate decays.
 * @param init_func Optional custom initialization function for weights.
 */
template <typename T>
Embedding<T>::Embedding(const int& input_dim, const int& output_dim, const T& learning_rate, const T& decay_rate, const size_t& decay_step, std::function<void(Tensor<T>&)> init_func)
: learning_rate_scheduler(learning_rate, decay_rate, decay_step), input_dim(input_dim), output_dim(output_dim) {
    this->weights = Tensor<T>({input_dim, output_dim});
    this->grad = Tensor<T>({input_dim, output_dim});

    // Use custom initialization if provided, otherwise use default Xavier initialization
    if (init_func) {
        init_func(this->weights);
    } else {
        initializeWeights();
    }
}

/**
 * @brief Initializes the weights of the Embedding layer using Xavier initialization.
 */
template <typename T>
void Embedding<T>::initializeWeights() {
    T fan_in = static_cast<T>(input_dim);
    T fan_out = static_cast<T>(output_dim);
    T limit = std::sqrt(6.0 / (fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-limit, limit);

    std::generate(weights.data.begin(), weights.data.end(), [&]() { return dis(gen); });
}

/**
 * @brief Performs the forward pass of the Embedding layer.
 *
 * @param input_data A tensor of input indices representing the input tokens.
 * @return Tensor<T> The corresponding embedding vectors for the input tokens.
 */
template <typename T>
Tensor<T> Embedding<T>::forward(const Tensor<T>& input_data) {
    if (input_data.shape().size() != 1 || input_data.shape()[0] > input_dim) {
        throw std::invalid_argument("Input data dimensions do not match embedding input dimensions.");
    }

    int batch_size = input_data.shape()[0];
    this->input = input_data;
    this->output = Tensor<T>({batch_size, output_dim});

    #pragma omp parallel
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            output.data[i * output_dim + j] = weights.data[input_data.data[i] * output_dim + j];
        }
    }

    return output;
}

/**
 * @brief Performs the backward pass of the Embedding layer, computing gradients with respect to the weights.
 *
 * @param grad_data The gradient of the loss with respect to the output of the Embedding layer.
 */
template <typename T>
void Embedding<T>::backward(const Tensor<T>& grad_data) {
    if (grad_data.shape().size() != 2 || grad_data.shape()[1] != output_dim) {
        throw std::invalid_argument("Gradient data dimensions do not match embedding output dimensions.");
    }

    int batch_size = grad_data.shape()[0];

    #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            this->grad.data[input.data[i] * output_dim + j] += grad_data.data[i * output_dim + j];
        }
    }
}

/**
 * @brief Updates the weights of the Embedding layer using the accumulated gradients.
 */
template <typename T>
void Embedding<T>::update(const int& epoch) {
    T learning_rate = learning_rate_scheduler.getLearningRate(epoch);

    #pragma omp parallel for  // Use OpenMP for parallel processing if supported
    for (int i = 0; i < input_dim; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            weights.data[i * output_dim + j] -= learning_rate * grad.data[i * output_dim + j];
        }
    }
}

/**
 * @brief Zeros the gradient tensor of the Embedding layer.
 */
template <typename T>
void Embedding<T>::zero_grad() {
    std::fill(this->grad.data.begin(), this->grad.data.end(), static_cast<T>(0));
}

/**
 * @brief Sets the weights of the Embedding layer to the provided tensor.
 *
 * @param new_weights The tensor containing the new weights.
 */
template <typename T>
void Embedding<T>::setWeights(const Tensor<T>& new_weights) {
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("New weights dimensions do not match current embedding weights dimensions.");
    }
    this->weights = new_weights;
}

/**
 * @brief Returns the weights tensor of the Embedding layer.
 *
 * @return Tensor<T> The weights tensor.
 */
template <typename T>
Tensor<T> Embedding<T>::getWeights() const {
    return this->weights;
}

/**
 * @brief Returns the gradient tensor of the Embedding layer.
 *
 * @return Tensor<T> The gradient tensor.
 */
template <typename T>
Tensor<T> Embedding<T>::getGrad() const {
    return this->grad;
}

#endif // EMBEDDING_TPP
