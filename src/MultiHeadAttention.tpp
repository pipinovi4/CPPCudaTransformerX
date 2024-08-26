#ifndef MULTIHEADATTENTION_TPP
#define MULTIHEADATTENTION_TPP

#include "../include/MultiHeadAttention.h"

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const int& hidden_dim, const int& num_heads, const int& head_dim, ActivationFunction<T>* activation)
    : hidden_dim(hidden_dim), num_heads(num_heads), head_dim(head_dim), activation(activation) {

    // Initialize the weight matrices for queries, keys, values, and output projections (without data only resized)
    this->W_q = Tensor<T>({hidden_dim, num_heads * head_dim}, hidden_dim * num_heads * head_dim);
    this->W_k = Tensor<T>({hidden_dim, num_heads * head_dim}, hidden_dim * num_heads * head_dim);
    this->W_v = Tensor<T>({hidden_dim, num_heads * head_dim}, hidden_dim * num_heads * head_dim);
    this->W_o = Tensor<T>({num_heads * head_dim, hidden_dim}, num_heads * head_dim * hidden_dim);

    // Initialize the bias terms for queries, keys, values, and output projections (without data only resized)
    this->b_q = Tensor<T>({num_heads * head_dim}, num_heads * head_dim);
    this->b_k = Tensor<T>({num_heads * head_dim}, num_heads * head_dim);
    this->b_v = Tensor<T>({num_heads * head_dim}, num_heads * head_dim);
    this->b_o = Tensor<T>({hidden_dim}, hidden_dim);

    // Initialize the parameters using a suitable initialization method
    initializeParameter(W_q);
    initializeParameter(W_k);
    initializeParameter(W_v);
    initializeParameter(W_o);

    // Initialize the biases with a constant value of 0.01
    for (int i = 0; i < num_heads * head_dim; i++) {
        b_q.data.emplace_back(0.01);
        b_k.data.emplace_back(0.01);
        b_v.data.emplace_back(0.01);
    }

    for (int i = 0; i < hidden_dim; i++) {
        b_o.data.emplace_back(0.01);
    }

    // Initialize gradients for all parameters as zeros
    grad_W_q = Tensor<T>({hidden_dim, num_heads * head_dim});
    grad_W_k = Tensor<T>({hidden_dim, num_heads * head_dim});
    grad_W_v = Tensor<T>({hidden_dim, num_heads * head_dim});
    grad_W_o = Tensor<T>({num_heads * head_dim, hidden_dim});

    grad_b_q = Tensor<T>({num_heads * head_dim});
    grad_b_k = Tensor<T>({num_heads * head_dim});
    grad_b_v = Tensor<T>({num_heads * head_dim});
    grad_b_o = Tensor<T>({hidden_dim});
}

// Initialize the weights using a normal distribution (Xavier/Glorot initialization)
template <typename T>
void MultiHeadAttention<T>::initializeParameter(Tensor<T>& weights) {
    // Calculate the limit for Xavier/Glorot initialization
    T limit = std::sqrt(6.0 / (weights.shape()[0] + weights.shape()[1]));

    if (weights.data.size() != weights.shape()[0] * weights.shape()[1]) {
        weights.data.resize(weights.shape()[0] * weights.shape()[1]);  // Allocate memory for the weights
    }

    // Map the weights tensor to an Eigen matrix
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat(weights.data.data(), weights.shape()[0], weights.shape()[1]);

    // Initialize the matrix with random values using Eigen's built-in functions
    mat = mat.NullaryExpr(mat.rows(), mat.cols(),
        [limit]() { return Eigen::internal::random<T>(-limit, limit); });
}

// Split the input tensor into multiple heads
template <typename T>
std::vector<Tensor<T>> MultiHeadAttention<T>::split_heads(const Tensor<T>& x) const {
    // Initialize a vector to hold the heads and preallocate memory
    std::vector<Tensor<T>> heads(num_heads, Tensor<T>({x.shape()[0], x.shape()[1] / num_heads}));

    // Get the shape of the input tensor
    const int seq_len = x.shape()[0];
    const int hidden_dim = x.shape()[1];
    const int head_dim = hidden_dim / num_heads;

    // Data access input data (x) for better performance
    const T* x_data = x.data.data();

    // Manually extract the slices for each head
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < seq_len; ++s) {
        for (int h = 0; h < hidden_dim; ++h) {
            const int head_index = h / head_dim;
            const int head_offset = h % head_dim;
            heads[head_index].data[s * head_dim + head_offset] = x_data[s * hidden_dim + h];
        }
    }

    return heads;
}


// Split the input tensor into multiple heads
template <typename T>
Tensor<T> MultiHeadAttention<T>::concat_heads(const std::vector<Tensor<T>>& heads) const {
    // Determine the shape of the heads
    const int seq_len = heads[0].shape()[0];
    const int head_dim = heads[0].shape()[1];

    // Initialize a tensor to hold the concatenated data with reserved space
    Tensor<T> concatenated_tensor({seq_len, head_dim * num_heads});

    // Data access for concatenated data for better performance
    T* concatenated_data_ptr = concatenated_tensor.data.data();

    // Concatenate all heads along the last dimension (head_dim)
    #pragma omp parallel for collapse(2)
    for (int s = 0; s < seq_len; ++s) {
        for (int n = 0; n < num_heads; ++n) {
            const T* head_data = heads[n].data.data();
            std::copy(
                head_data + s * head_dim, 
                head_data + (s + 1) * head_dim, 
                concatenated_data_ptr + s * head_dim * num_heads + n * head_dim
            );
        }
    }

    return concatenated_tensor;
}


// Forward pass for MultiHeadAttention layer
template <typename T>
Tensor<T> MultiHeadAttention<T>::forward(const Tensor<T>& input, const Tensor<T>* mask) {
     // Cache the input data for backpropagation
     input_cache = input;

     // Step 1: Linear projections to get queries, keys, and values
     Tensor<T> queries = input.dot(W_q);
     Tensor<T> keys = input.dot(W_k);
     Tensor<T> values = input.dot(W_v);

     // Add the bias terms to the projections
     queries += b_q;
     keys += b_k;
     values += b_v;

     // Step 2: Split the projections into multiple heads
     queries_heads = split_heads(queries);
     keys_heads = split_heads(keys);
     values_heads = split_heads(values);

     // Clear attention_heads vector
     attention_heads.clear();

     // Step 3: Iterate over each head, compute attention, and store the results
#pragma omp parallel for
     for (int i = 0; i < num_heads; ++i) {
         // Compute the attention scores
         Tensor<T> attention_scores = queries_heads[i].dot(keys_heads[i].transpose({1, 0}));
         attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale by the square root of the head dimension

         // Apply mask (if provided)
         if (mask != nullptr) {
             // Assuming mask has the same shape as attention_scores
            for (int j = 0; j < attention_scores.size(); ++j) {
                attention_scores.data[j] += (*mask).data[j] ? 0 : -1e9;  // Apply large negative value to masked positions
            }
         }

         // // Apply the activation function (typically softmax)
         typename ActivationFunction<T>::Softmax softmax;
         softmax.forward(attention_scores);

         Tensor<T> attention_output = attention_scores.dot(values_heads[i]);

         // Compute the output as a weighted sum of the values and store the result
         attention_heads.push_back(attention_output);
     }

     // Step 4: Concatenate the attention outputs from all heads
     Tensor<T> concatenated_output = concat_heads(attention_heads);  // Shape: [batch_size, seq_len, d_model]

     // Step 5: Final linear projection to get the output
     Tensor<T> output = concatenated_output.dot(W_o);  // Shape: [batch_size, seq_len, d_model]

     return output + b_o;;  // Shape: [batch_size, seq_len, d_model]
}

// Backward pass for MultiHeadAttention layer
template <typename T>
void MultiHeadAttention<T>::backward(Tensor<T>& grad) {
    // Step 1: Compute gradients of the final projection (output layer)
    Tensor<T> grad_concatenated = grad.dot(W_o.transpose({1, 0}));

    // In-place operation for weight gradient computation
    grad_W_o = concat_heads(attention_heads).transpose({1, 0}).dot(grad);
    grad_b_o = grad.sum(0); // Assuming b_o is a bias vector

    // Step 2: Split the gradient of the concatenated heads back into multiple heads
    std::vector<Tensor<T>> grad_attention_heads = split_heads(grad_concatenated);

    // Initialize gradients for queries, keys, and values
    std::vector<Tensor<T>> grad_queries_heads(num_heads);
    std::vector<Tensor<T>> grad_keys_heads(num_heads);
    std::vector<Tensor<T>> grad_values_heads(num_heads);

    // Step 3: Iterate over each head to compute the gradients
    #pragma omp parallel for
    for (int i = 0; i < num_heads; ++i) {
        // Compute gradients of the attention scores
        Tensor<T> grad_attention_scores = grad_attention_heads[i].dot(values_heads[i].transpose({1, 0}));
        grad_attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale gradient

        // Apply the softmax gradient in-place
        typename ActivationFunction<T>::Softmax softmax;
        softmax.backward(grad_attention_scores);

        // Compute gradients w.r.t. queries, keys, and values
        grad_queries_heads[i] = grad_attention_scores.dot(keys_heads[i]);
        grad_keys_heads[i] = grad_attention_scores.dot(queries_heads[i]);
        grad_values_heads[i] = grad_attention_scores.dot(grad_attention_heads[i]);
    }

    // Step 4: Concatenate the gradients of the heads
    Tensor<T> grad_queries = concat_heads(grad_queries_heads);
    Tensor<T> grad_keys = concat_heads(grad_keys_heads);
    Tensor<T> grad_values = concat_heads(grad_values_heads);

    // Step 5: Compute gradients of the linear projections (W_q, W_k, W_v) efficiently
    grad_W_q = input_cache.transpose({1, 0}).dot(grad_queries);
    grad_W_k = input_cache.transpose({1, 0}).dot(grad_keys);
    grad_W_v = input_cache.transpose({1, 0}).dot(grad_values);

    // Compute the gradients with respect to the biases
    grad_b_q = grad_queries.sum(0);  // Shape: [d_model]
    grad_b_k = grad_keys.sum(0);  // Shape: [d_model]
    grad_b_v = grad_values.sum(0);  // Shape: [d_model]

    // Step 6: Compute the gradients of the input
    grad = grad_queries.dot(W_q.transpose({1, 0})) + grad_keys.dot(W_k.transpose({1, 0})) + grad_values.dot(W_v.transpose({1, 0}));

    // Optionally, clear intermediate tensors to free memory
    grad_concatenated.data.clear();
    grad_queries_heads.clear();
    grad_keys_heads.clear();
    grad_values_heads.clear();
}

// Explicit template instantiation for the following types (override the class declaration)
template <typename T>
Tensor<T> MultiHeadAttention<T>::forward(const Tensor<T>& input) {
    return forward(input, nullptr);  // Overloaded forward method with no mask
}

// Getter for the model parameters
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttention<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> params = { W_q, W_k, W_v, W_o, b_q, b_k, b_v, b_o };  // Return all parameters
    return params;
}

// Getter for the model gradients
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> MultiHeadAttention<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grads = { grad_W_q, grad_W_k, grad_W_v, grad_W_o, grad_b_q, grad_b_k, grad_b_v, grad_b_o };  // Return all gradients
    return grads;
}

#endif // MULTIHEADATTENTION_TPP