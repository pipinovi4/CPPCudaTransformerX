#ifndef EMBEDDING_TPP
#define EMBEDDING_TPP

#include "../include/Embedding.h"
#include <stdexcept>

template <typename T>
Embedding<T>::Embedding(const int& vocab_size, const int& embedding_dims,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule, std::function<void(Tensor<T>&)> init_func)
    : lr_schedule_(lr_schedule), vocab_size_(vocab_size), embedding_dims_(embedding_dims) {

    // Initialize the weights and gradient tensors with appropriate shapes
    this->weights_ = Tensor<T>({vocab_size, embedding_dims});
    this->grad_ = Tensor<T>({vocab_size, embedding_dims});

    // Use custom initialization function if provided, otherwise initialize weights with default method
    if (init_func) {
        init_func(this->weights_);
    } else {
        initializeWeights();
    }
}

template <typename T>
void Embedding<T>::initializeWeights() {
    // Calculate the limit for Xavier/Glorot initialization
    T fan_in = static_cast<T>(vocab_size_);
    T fan_out = static_cast<T>(embedding_dims_);
    T limit = std::sqrt(6.0 / (fan_in + fan_out));

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-limit, limit);

    // Populate weights tensor with random values within the calculated limit
    std::generate(weights_.data.begin(), weights_.data.end(), [&]() { return dis(gen); });
}

template <typename T>
Tensor<T> Embedding<T>::forward(const Tensor<T>& input) {
    // Cache the input for use in the backward pass
    this->input_cache_ = input;

    // Ensure the input tensor is 2D [batch_size, sequence_length]
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 2) {
        throw std::invalid_argument("Input tensor must be 2D for embedding layer. Received shape: " + std::to_string(input_shape.size()));
    }

    // Define the shape of the output tensor [batch_size, sequence_length, embedding_dims]
    std::vector<int> output_shape = {input_shape[0], input_shape[1], embedding_dims_};
    Tensor<T> output(output_shape);

    // Data access for better performance
    auto input_data = input.data.data();
    auto weights_data = weights_.data.data();
    auto output_data = output.data.data();

    // Populate the output tensor by looking up the embeddings for each token in the input
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < input_shape[0]; ++i) {  // Iterate over batch size
        for (int j = 0; j < input_shape[1]; ++j) {  // Iterate over sequence length
            const int index = static_cast<int>(input_data[i * input_shape[1] + j]);
            for (int k = 0; k < embedding_dims_; ++k) {  // Iterate over embedding dimensions
                output_data[(i * input_shape[1] + j) * embedding_dims_ + k] = weights_data[index * embedding_dims_ + k];
            }
        }
    }
    return output;
}

template <typename T>
void Embedding<T>::backward(Tensor<T>& grad) {
    // Zero the gradients before accumulation
    zero_grad();

    // Ensure the gradient tensor is 3D [batch_size, sequence_length, embedding_dims]
    const std::vector<int> grad_shape = grad.shape();
    const std::vector<int> input_cache_shape = input_cache_.shape();
    if (grad_shape.size() != 3) {
        throw std::invalid_argument("Gradient tensor must be 3D for embedding layer. Received shape: " + std::to_string(grad_shape.size()));
    }

    // Data access for better performance
    auto grad_data = grad.data.data();
    auto input_cache_data = input_cache_.data.data();
    auto grad_data_ = grad_.data.data();

    // Accumulate gradients for the embeddings based on the backward pass
    for (int k = 0; k < embedding_dims_; ++k) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < grad_shape[0]; ++i) {
            for (int j = 0; j < grad_shape[1]; ++j) {
                const int index = static_cast<int>(input_cache_data[i * input_cache_shape[1] + j]);
                grad_data_[index * embedding_dims_ + k] += grad_data[(i * grad_shape[1] + j) * grad_shape[2] + k];
            }
        }
    }
}

template <typename T>
void Embedding<T>::zero_grad() {
    std::fill(grad_.data.begin(), grad_.data.end(), static_cast<T>(0));
}

template <typename T>
void Embedding<T>::setWeights(const Tensor<T>& new_weights) {
    // Ensure the new weights match the shape of the existing weights
    if (new_weights.shape() != weights_.shape()) {
        throw std::invalid_argument("New weights dimensions do not match current embedding weights dimensions.");
    }
    this->weights_ = new_weights;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<float>>> Embedding<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<float>>> weights_vector;
    weights_vector.push_back(std::ref(this->weights_));
    return weights_vector;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<float>>> Embedding<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<float>>> grad_vector;
    grad_vector.push_back(std::ref(this->grad_));
    return grad_vector;
}


#endif // EMBEDDING_TPP
