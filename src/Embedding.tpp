#ifndef EMBEDDING_TPP
#define EMBEDDING_TPP

#include "../include/Embedding.h"
#include <stdexcept>

template <typename T>
Embedding<T>::Embedding(const int& vocab_size, const int& embedding_dims,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule, std::function<void(Tensor<T>&)> init_func)
    : lr_schedule(lr_schedule), vocab_size(vocab_size), embedding_dims(embedding_dims) {

    // Initialize the weights and gradient tensors with appropriate shapes
    this->weights = Tensor<T>({vocab_size, embedding_dims});
    this->grad = Tensor<T>({vocab_size, embedding_dims});

    // Use custom initialization function if provided, otherwise initialize weights with default method
    if (init_func) {
        init_func(this->weights);
    } else {
        initializeWeights();
    }
}

template <typename T>
void Embedding<T>::initializeWeights() {
    // Calculate the limit for Xavier/Glorot initialization
    T fan_in = static_cast<T>(vocab_size);
    T fan_out = static_cast<T>(embedding_dims);
    T limit = std::sqrt(6.0 / (fan_in + fan_out));

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-limit, limit);

    // Populate weights tensor with random values within the calculated limit
    std::generate(weights.data.begin(), weights.data.end(), [&]() { return dis(gen); });
}

template <typename T>
Tensor<T> Embedding<T>::forward(const Tensor<T>& input) {
    // Cache the input for use in the backward pass
    this->input_cache = input;

    // Define the shape of the output tensor [batch_size, sequence_length, embedding_dims]
    std::vector<int> output_shape = {input.shape()[0], input.shape()[1], embedding_dims};
    Tensor<T> output(output_shape, input.shape()[0] * input.shape()[1] * embedding_dims);

    // Populate the output tensor by looking up the embeddings for each token in the input
    for (int i = 0; i < input.shape()[0]; ++i) {  // Iterate over batch size
        for (int j = 0; j < input.shape()[1]; ++j) {  // Iterate over sequence length
            const int index = static_cast<int>(input.data[i * input.shape()[1] + j]);
            for (int k = 0; k < embedding_dims; ++k) {  // Iterate over embedding dimensions
                output.data.push_back(weights.data[index * embedding_dims + k]);
            }
        }
    }
    return output;
}

template <typename T>
void Embedding<T>::backward(Tensor<T>& grad) {
    // Zero the gradients before accumulation
    zero_grad();

    // Accumulate gradients for the embeddings based on the backward pass
    for (int i = 0; i < grad.shape()[0]; ++i) {  // Iterate over batch size
        for (int j = 0; j < grad.shape()[1]; ++j) {  // Iterate over sequence length
            const int index = static_cast<int>(input_cache.data[i * input_cache.shape()[1] + j]);
            for (int k = 0; k < embedding_dims; ++k) {  // Iterate over embedding dimensions
                grad.data[index * embedding_dims + k] += grad.data[(i * grad.shape()[1] + j) * grad.shape()[2] + k];
            }
        }
    }
}

template <typename T>
void Embedding<T>::update(const int& epoch) {
    // Retrieve the current learning rate based on the epoch
    T learning_rate = lr_schedule.getLearningRate(epoch);

    // Access the raw data pointer for weights for faster updates
    auto weights_data_access = weights.data.data();

    // Update weights in parallel if supported by the compiler
    #pragma omp parallel for
    for (int i = 0; i < vocab_size; ++i) {  // Iterate over vocabulary size
        for (int j = 0; j < embedding_dims; ++j) {  // Iterate over embedding dimensions
            weights_data_access[i * embedding_dims + j] -= learning_rate * grad.data[i * embedding_dims + j];
        }
    }
}

template <typename T>
void Embedding<T>::zero_grad() {
    // Reset all gradient values to zero
    std::fill(this->grad.data.begin(), this->grad.data.end(), static_cast<T>(0));
}

template <typename T>
void Embedding<T>::setWeights(const Tensor<T>& new_weights) {
    // Ensure the new weights match the shape of the existing weights
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("New weights dimensions do not match current embedding weights dimensions.");
    }
    this->weights = new_weights;
}

template <typename T>
Tensor<T>& Embedding<T>::getWeights() {
    // Return a reference to the weights tensor
    return this->weights;
}

template <typename T>
Tensor<T>& Embedding<T>::getGrad() {
    // Return a reference to the gradient tensor
    return this->grad;
}

#endif // EMBEDDING_TPP
