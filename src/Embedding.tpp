#ifndef EMBEDDING_TPP
#define EMBEDDING_TPP

#include "../include/Embedding.h"

template <typename T>
Embedding<T>::Embedding(const int& vocab_size, const int& embedding_dims,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule)
    : lr_schedule_(lr_schedule), vocab_size_(vocab_size), embedding_dims_(embedding_dims) {

    // Initialize the weights and gradient tensors with appropriate shapes
    this->weights_ = Tensor<T>({vocab_size, embedding_dims});
    this->grad_ = Tensor<T>({vocab_size, embedding_dims});

    // Initialize the weights using Xavier/Glorot initialization
    initializeWeights();
}

// Initialize the weights using the Xavier (Glorot) initialization.
template <typename T>
void Embedding<T>::initializeWeights() {
    // Calculate the limit for Xavier/Glorot initialization
    T fan_in = static_cast<T>(vocab_size_);
    T fan_out = static_cast<T>(embedding_dims_);
    T limit = std::sqrt(6.0 / (fan_in + fan_out));

    // Map the weights tensor to an Eigen matrix
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat(weights_.data.data(), vocab_size_, embedding_dims_);

    // Initialize the matrix with random values using Eigen's built-in functions
    mat = mat.NullaryExpr(mat.rows(), mat.cols(),
        [limit]() { return Eigen::internal::random<T>(-limit, limit); });
}

template <typename T>
Tensor<T> Embedding<T>::forward(const Tensor<T>& input) {
    // Cache the input for use in the backward pass
    this->input_cache_ = input;

    // Ensure the input tensor is 1D [sequence_length]
    const std::vector<int> input_shape = input.shape();
    if (input_shape.size() != 1) {
        throw std::invalid_argument("Input tensor must be 1D for embedding layer. Received shape: " + std::to_string(input_shape.size()));
    }

    // Define the shape of the output tensor [sequence_length, embedding_dims]
    std::vector<int> output_shape = {input_shape[0], embedding_dims_};

    Tensor<T> output(output_shape);

    // Data access for better performance
    const T* input_data = input.data.data();
    const T* weights_data = weights_.data.data();
    T* output_data = output.data.data();

    // Populate the output tensor by looking up the embeddings for each token in the input
    for (int j = 0; j < input_shape[0]; ++j) {  // Iterate over sequence length
        const int index = static_cast<int>(input_data[j]);
        for (int k = 0; k < embedding_dims_; ++k) {  // Iterate over embedding dimensions
            output_data[j * embedding_dims_ + k] = weights_data[index * embedding_dims_ + k];
        }
    }

    return output;
}

template <typename T>
void Embedding<T>::backward(Tensor<T>& grad) {
    // Ensure the gradient tensor is 2D [sequence_length, embedding_dims]
    const std::vector<int> grad_shape = grad.shape();
    const std::vector<int> input_cache_shape = input_cache_.shape();

    if (grad_shape.size() != 2) {
        throw std::invalid_argument("Gradient tensor must be 2D for embedding layer. Received shape: " + std::to_string(grad_shape.size()));
    }

    // Check the input cache is 1D and matches the sequence length
    if (input_cache_shape.size() != 1 || input_cache_shape[0] != grad_shape[0]) {
        throw std::invalid_argument("Input cache must be 1D and match the sequence length.");
    }

    // Data access for better performance
    T* grad_data_ptr = grad.data.data();  // Incoming gradient [sequence_length, embedding_dims]
    T* input_cache_data_ptr = input_cache_.data.data();  // Input token indices [sequence_length]
    T* grad_data_ptr_ = grad_.data.data();  // Embedding weight gradient [vocab_size, embedding_dims]

    // Accumulate gradients for the embeddings based on the backward pass
    for (int i = 0; i < vocab_size_; ++i) {
        for (int j = 0; j < grad_shape[0]; ++j) {  // Iterate over the sequence length
            const int index = static_cast<int>(input_cache_data_ptr[j]);  // Embedding index from input cache
            for (int k = 0; k < grad_shape[1]; ++k) {  // Iterate over the embedding dimensions
                grad_data_ptr_[index * grad_shape[1] + k] += grad_data_ptr[j * grad_shape[1] + k];  // Accumulate gradients for the embedding weights
            }
        }
    }
}

template <typename T>
void Embedding<T>::setWeights(const Tensor<T>& new_weights) {
    // Ensure the new weights match the shape of the existing weights
    if (new_weights.shape() != weights_.shape()) {
        throw std::invalid_argument("New weights dimensions do not match current embedding weights dimensions.");
    }
    this->weights_ = new_weights;
}

// Getter for the model parameters
template <typename T>
std::vector<std::reference_wrapper<Tensor<float>>> Embedding<T>::parameters() {
    // Return a vector of references to the weights tensor
    std::vector<std::reference_wrapper<Tensor<float>>> weights_vector;
    weights_vector.push_back(std::ref(this->weights_));
    return weights_vector;
}

// Getter for the model parameters shape
template <typename T>
std::vector<std::reference_wrapper<Tensor<float>>> Embedding<T>::gradients() {
    // Return a vector of references to the gradient tensor
    std::vector<std::reference_wrapper<Tensor<float>>> grad_vector;
    grad_vector.push_back(std::ref(this->grad_));
    return grad_vector;
}


#endif // EMBEDDING_TPP
