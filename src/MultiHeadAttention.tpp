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
    std::random_device rd;
    std::mt19937 gen(rd());
    T limit = std::sqrt(6.0 / (weights.shape()[0] + weights.shape()[1]));
    std::normal_distribution<T> dist(-limit, limit);  // Xavier/Glorot initialization

    size_t weights_size = 1;
    for (auto dim : weights.dimensions) {
        weights_size *= dim;  // Calculate the total number of elements in the tensor
    }

    weights.data.clear();  // Clear any existing data in the tensor
    for (size_t i = 0; i < weights_size; i++) {
        weights.data.push_back(dist(gen));  // Initialize the weights with values drawn from a normal distribution
    }
}

// Split the input tensor into multiple heads
template <typename T>
std::vector<Tensor<T>> MultiHeadAttention<T>::split_heads(const Tensor<T>& x) const {
    // Initialize a vector to hold the heads
    std::vector<Tensor<T>> heads{};

    // Reserve the heads vector to the number of heads
    heads.reserve(num_heads);

    // Get the shape of the input tensor
    const std::vector<int> shape = x.shape();
    const int batch_size = shape[0];  // Batch size
    const int seq_len = shape[1];  // Sequence length
    const int hidden_dim = shape[2];  // Hidden dimension
    const int head_dim = hidden_dim / num_heads;  // Dimension of each head

    // Ensure hidden_dim is divisible by num_heads
    if (hidden_dim % num_heads != 0) {
        throw std::invalid_argument("hidden_dim must be divisible by num_heads");
    }

    // Data access input data (x) for better performance
    const T* x_data = x.data.data();

    // Manually extract the slices for each head
    for (int i = 0; i < num_heads; ++i) {
        // Initialize a vector to hold the data for the head
        std::vector<T> head_data(batch_size * seq_len * head_dim);

        // Data access for head data
        T* head_data_ptr = head_data.data();

        // Extract the data for the head
        #pragma omp parallel for collapse(3)
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < head_dim; ++h) {
                    // Calculate the index in the input tensor
                    int index = b * seq_len * hidden_dim + s * hidden_dim + i * head_dim + h;

                    // Add the data to the head
                    head_data_ptr[b * seq_len * head_dim + s * head_dim + h] = x_data[index];
                }
            }
        }
        // Create a tensor from the head data
        heads.emplace_back(Tensor<T>({batch_size, seq_len, head_dim}, std::move(head_data)));
    }

    return heads;
}

// Concatenate the output from multiple heads
template <typename T>
Tensor<T> MultiHeadAttention<T>::concat_heads(const std::vector<Tensor<T>>& heads) const {
    // Determine the shape of the heads
    const int batch_size = heads[0].shape()[0];
    const int seq_len = heads[0].shape()[1];
    const int head_dim = heads[0].shape()[2];

    // Initialize a vector to hold the concatenated data and reserve space
    std::vector<T> concatenated_data(batch_size * seq_len * head_dim * num_heads);

    // Data access for concatenated data for better performance
    T* concatenated_data_ptr = concatenated_data.data();

    // Concatenate all heads along the last dimension (head_dim)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const int offset = b * seq_len * head_dim * num_heads + s * head_dim * num_heads; // Calculate the starting point for each sequence

            #pragma omp parallel for
            for (int n = 0; n < num_heads; ++n) {
                const T* head_data = heads[n].data.data();

                // Copy the data from the head to the concatenated data
                for (int h = 0; h < head_dim; ++h) {
                    concatenated_data_ptr[offset + n * head_dim + h] = head_data[b * seq_len * head_dim + s * head_dim + h];
                }
            }
        }
    }

    return Tensor<T>({batch_size, seq_len, head_dim * num_heads}, std::move(concatenated_data));
}

// Forward pass for MultiHeadAttention layer
template <typename T>
Tensor<T> MultiHeadAttention<T>::forward(const Tensor<T>& input, const Tensor<T>* mask) {
    // Cache the input data for backpropagation
    input_cache = input; // 8 100 32

     // Step 1: Linear projections to get queries, keys, and values
     Tensor<T> queries = input.dot(W_q);  // 8 100 32
     Tensor<T> keys = input.dot(W_k);     // 8 100 32
     Tensor<T> values = input.dot(W_v);   // 8 100 32

     // Add the bias terms to the projections
     queries += b_q; // 8 100 32
     keys += b_k; // 8 100 32
     values += b_v; // 8 100 32

     // Step 2: Split the projections into multiple heads
     queries_heads = split_heads(queries);  // 8 100 4
     keys_heads = split_heads(keys);        // 8 100 4
     values_heads = split_heads(values);    // 8 100 4

     // Clear attention_heads vector
     attention_heads.clear();

     // Step 3: Iterate over each head, compute attention, and store the results
#pragma omp parallel for
     for (int i = 0; i < num_heads; ++i) {
         // Compute the attention scores
         Tensor<T> attention_scores = queries_heads[i].dot(keys_heads[i].transpose({0, 2, 1})); // (8, 100, 4).dot((8, 4, 100)) = [8 100 8 100]
         attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale by the square root of the head dimension
         attention_scores = attention_scores.sum(2); // 8 100 100

         // Apply mask (if provided)
         if (mask != nullptr) {
             // Assuming mask has the same shape as attention_scores
             for (int b = 0; b < attention_scores.shape()[0]; ++b) {  // batch_size loop
                 for (int j = 0; j < attention_scores.shape()[1]; ++j) {
                     for (int k = 0; k < attention_scores.shape()[2]; ++k) {
                         attention_scores.data[b*j*k] += (*mask).data[b*j*k] ? 0 : -1e9;  // Apply large negative value to masked positions
                     }
                 }
             }
         }

         // // Apply the activation function (typically softmax)
         typename ActivationFunction<T>::Softmax softmax;
         softmax.forward(attention_scores);  // 8 100 100

         Tensor<T> attention_output = attention_scores.dot(values_heads[i]).sum(2);  // 8 100 100

         // Compute the output as a weighted sum of the values and store the result
         attention_heads.push_back(attention_output); // (8 100 8 4).sum(2) = [8 100 4]
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
    Tensor<T> grad_concatenated = grad.dot(W_o.transpose({1, 0}));  // Gradient w.r.t. concatenated output // 8 100 4

    grad_W_o = concat_heads(attention_heads).transpose({0, 2, 1}).dot(grad).sum(2).sum(0);  // Gradient w.r.t. W_o, normalized by batch size // 100 32
    grad_b_o = grad.sum(0).sum(0);  // Gradient w.r.t. b_o // 32

    // Step 2: Split the gradient of the concatenated heads back into multiple heads
    std::vector<Tensor<T>> grad_attention_heads = split_heads(grad_concatenated);  // 8 for [ 8 100 4]

    // Initialize gradients for queries, keys, and values
    std::vector<Tensor<T>> grad_queries_heads(num_heads);
    std::vector<Tensor<T>> grad_keys_heads(num_heads);
    std::vector<Tensor<T>> grad_values_heads(num_heads);

    // Step 3: Iterate over each head to compute the gradients
    #pragma omp parallel for
    for (int i = 0; i < num_heads; ++i) {
        // Compute gradients of the attention scores
        Tensor<T> grad_attention_scores = grad_attention_heads[i].dot(values_heads[i].transpose({0, 2, 1}));  // 2 8 2 8
        grad_attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale gradient
        grad_attention_scores = grad_attention_scores.sum(2);  // 2 8 8


        // Apply the activation function (softmax) gradient
        typename ActivationFunction<T>::Softmax softmax;
        softmax.forward(grad_attention_scores);  // 8 100 100

        // Compute gradients w.r.t. queries, keys, and values
        grad_queries_heads[i] = grad_attention_scores.dot(keys_heads[i]).sum(2); // 8 100 4 / 8 100 4
        grad_keys_heads[i] = grad_attention_scores.dot(queries_heads[i]).sum(2);  // Shape: [batch_size, seq_len, head_dim]
        grad_values_heads[i] = grad_attention_scores.dot(grad_attention_heads[i]).sum(2);  // Shape: [batch_size, seq_len, head_dim]
    }

    // Step 4: Concatenate the gradients of the heads
    Tensor<T> grad_queries = concat_heads(grad_queries_heads);  // Shape: [batch_size, seq_len, d_model]
    Tensor<T> grad_keys = concat_heads(grad_keys_heads);  // Shape: [batch_size, seq_len, d_model]
    Tensor<T> grad_values = concat_heads(grad_values_heads);  // Shape: [batch_size, seq_len, d_model]

    // Step 5: Compute gradients of the linear projections (W_q, W_k, W_v)
    grad_W_q = input_cache.transpose({0, 2, 1}).dot(grad_queries).sum(2).sum(0);  // Shape: [d_model, d_model], normalized by batch size
    grad_W_k = input_cache.transpose({0, 2, 1}).dot(grad_keys).sum(2).sum(0);  // Shape: [d_model, d_model], normalized by batch size
    grad_W_v = input_cache.transpose({0, 2, 1}).dot(grad_values).sum(2).sum(0);  // Shape: [d_model, d_model], normalized by batch size

    // Compute the gradients with respect to the biases
    grad_b_q = grad_queries.sum(0).sum(0);  // Shape: [d_model]
    grad_b_k = grad_keys.sum(0).sum(0);  // Shape: [d_model]
    grad_b_v = grad_values.sum(0).sum(0);  // Shape: [d_model]

    // Step 6: Propagate gradients through the input data
    Tensor<T> grad_input_q = grad_queries.dot(W_q);  // Shape: [batch_size, seq_len, d_model]
    Tensor<T> grad_input_k = grad_keys.dot(W_k);  // Shape: [batch_size, seq_len, d_model]
    Tensor<T> grad_input_v = grad_values.dot(W_v);  // Shape: [batch_size, seq_len, d_model]

    grad = grad_input_q + grad_input_k + grad_input_v;
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