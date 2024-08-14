#ifndef EMBEDDING_TPP
#define EMBEDDING_TPP

#include "../include/Embedding.h"
#include <stdexcept>

/**
 * @brief Constructs an Embedding layer with the specified dimensions and learning rate parameters.
 *
 * @param init_func Optional custom initialization function for weights.
 * @param lr_schedule The learning rate scheduler to use for updating the learning rate over epochs.
 * @param vocab_size The size of the input dimension (vocabulary size).
 * @param embedding_dims The size of the output dimension (embedding vector size).
 * @tparam T Data type of the input tensor.
 */
template <typename T>
Embedding<T>::Embedding(const int& vocab_size, const int& embedding_dims,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule, std::function<void(Tensor<T>&)> init_func)
    : lr_schedule(lr_schedule), vocab_size(vocab_size), embedding_dims(embedding_dims) {
    this->weights = Tensor<T>({vocab_size, embedding_dims});
    this->grad = Tensor<T>({vocab_size, embedding_dims});

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
    T fan_in = static_cast<T>(vocab_size);
    T fan_out = static_cast<T>(embedding_dims);
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
    this->input_cache = input_data; // Store input data for backward pass

    // Define output shape as [batch_size, sequence_length, embedding_dims]
    std::vector<int> output_shape = {input_data.shape()[0], input_data.shape()[1], embedding_dims};
    Tensor<T> output(output_shape, input_data.shape()[0] * input_data.shape()[1] * embedding_dims);

    // Iterate over batch size and sequence length
    for (int i = 0; i < input_data.shape()[0]; ++i) {
        for (int j = 0; j < input_data.shape()[1]; ++j) {
            const int index = static_cast<int>(input_data.data[i * input_data.shape()[1] + j]);
            for (int k = 0; k < embedding_dims; ++k) {
                output.data.push_back(weights.data[index * embedding_dims + k]);
            }
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
void Embedding<T>::backward(Tensor<T>& grad_data) {
    // Zero the gradient tensor before accumulating new gradients
    zero_grad();

    // Iterate over batch size and sequence length
    for (int i = 0; i < grad_data.shape()[0]; ++i) {
        for (int j = 0; j < grad_data.shape()[1]; ++j) {
            const int index = static_cast<int>(input_cache.data[i * input_cache.shape()[1] + j]);
            for (int k = 0; k < embedding_dims; ++k) {
                grad_data.data[index * embedding_dims + k] += grad_data.data[(i * grad_data.shape()[1] + j) * grad_data.shape()[2] + k];
            }
        }
    }
}

/**
 * @brief Updates the weights of the Embedding layer using the accumulated gradients.
 */
template <typename T>
void Embedding<T>::update(const int& epoch) {
    T learning_rate = lr_schedule.getLearningRate(epoch);

    auto weights_data_access = weights.data.data();

    #pragma omp parallel for  // Use OpenMP for parallel processing if supported
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < embedding_dims; ++j) {
            weights_data_access[i * embedding_dims + j] -= learning_rate * grad.data[i * embedding_dims + j];
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
Tensor<T>& Embedding<T>::getWeights() {
    return this->weights;
}

/**
 * @brief Returns the gradient tensor of the Embedding layer.
 *
 * @return Tensor<T> The gradient tensor.
 */
template <typename T>
Tensor<T>& Embedding<T>::getGrad() {
    return this->grad;
}

#endif // EMBEDDING_TPP