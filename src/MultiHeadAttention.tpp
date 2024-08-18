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
    std::normal_distribution<T> dist(0.0, 0.02);

    size_t weights_size = 1;
    for (auto dim : weights.dimensions) {
        weights_size *= dim;  // Calculate the total number of elements in the tensor
    }

    weights.data.clear();  // Clear any existing data in the tensor
    for (size_t i = 0; i < weights_size; i++) {
        weights.data.emplace_back(dist(gen));  // Initialize the weights with values drawn from a normal distribution
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
    const int seq_len = shape[0];  // Sequence length
    const int hidden_dim = shape[1];  // Hidden dimension
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
        std::vector<T> head_data(seq_len * head_dim);

        // Data access for head data
        T* head_data_ptr = head_data.data();

        // Extract the data for the head
        #pragma omp parallel for collapse(2)
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < head_dim; ++h) {
                // Calculate the index in the input tensor
                int index = s * hidden_dim + i * head_dim + h;

                // Add the data to the head
                head_data_ptr[s * head_dim + h] = x_data[index];
            }
        }
        // Create a tensor from the head data
        heads.emplace_back(Tensor<T>({seq_len, head_dim}, std::move(head_data)));
    }

    return heads;
}

// Concatenate the output from multiple heads
template <typename T>
Tensor<T> MultiHeadAttention<T>::concat_heads(const std::vector<Tensor<T>>& heads) const {
    // Determine the shape of the heads
    const int seq_len = heads[0].shape()[0];
    const int head_dim = heads[0].shape()[1];

    // Initialize a vector to hold the concatenated data and reserve space
    std::vector<T> concatenated_data(seq_len * head_dim * num_heads);

    // Data access for concatenated data for better performance
    T* concatenated_data_ptr = concatenated_data.data();

    // Concatenate all heads along the last dimension (head_dim)
    for (int s = 0; s < seq_len; ++s) {
        int offset = s * head_dim * num_heads; // Calculate the starting point for each sequence

        #pragma omp parallel for
        for (int n = 0; n < num_heads; ++n) {
            const T* head_data = heads[n].data.data();

            // Copy the data from the head to the concatenated data
            for (int h = 0; h < head_dim; ++h) {
                concatenated_data_ptr[offset + n * head_dim + h] = head_data[s * head_dim + h];
            }
        }
    }

    return Tensor<T>({seq_len, head_dim * num_heads}, std::move(concatenated_data));
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

    attention_heads.clear();

    // Step 3: Iterate over each head, compute attention, and store the results
#pragma omp parallel for
    for (int i = 0; i < num_heads; ++i) {
        // Scaled dot product attention
        Tensor<T> attention_scores = queries_heads[i].dot(keys_heads[i].transpose());
        attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale by the square root of the head dimension

        // Apply mask (if provided)
        if (mask != nullptr) {
            // Assuming mask has the same shape as attention_scores
            for (int j = 0; j < attention_scores.size(); ++j) {
                attention_scores.data[j] += (*mask).data[j] ? 0 : -1e9;  // Apply large negative value to masked positions
            }
        }

        // Apply the activation function (typically softmax)
        activation->forward(attention_scores);

        // Compute the output as a weighted sum of the values
        Tensor<T> attention_output = attention_scores.dot(values_heads[i]);

        // Store the attention outputs
        attention_heads.push_back(attention_output);
    }

    // Step 4: Concatenate the attention outputs from all heads
    Tensor<T> concatenated_output = concat_heads(attention_heads);

    // Step 5: Final linear projection to get the output
    Tensor<T> output = concatenated_output.dot(W_o);
    output += b_o;

    return output;
}

// Backward pass for MultiHeadAttention layer
template <typename T>
void MultiHeadAttention<T>::backward(Tensor<T>& grad) {
    // Step 1: Compute gradients of the final projection (output layer)
    Tensor<T> grad_concatenated = grad.dot(W_o.transpose());  // Gradient w.r.t. concatenated output
    grad_W_o = concat_heads(attention_heads).dot(grad.expandDims(0));  // Gradient w.r.t. W_o
    grad_b_o = grad.sum(0);  // Gradient w.r.t. b_o

    // Step 2: Split the gradient of the concatenated heads back into multiple heads
    std::vector<Tensor<T>> grad_attention_heads = split_heads(grad_concatenated);

    // Initialize gradients for queries, keys, and values
    std::vector<Tensor<T>> grad_queries_heads(num_heads);
    std::vector<Tensor<T>> grad_keys_heads(num_heads);
    std::vector<Tensor<T>> grad_values_heads(num_heads);

    // Step 3: Iterate over each head to compute the gradients
    #pragma omp parallel
    for (int i = 0; i < num_heads; ++i) {
        // Compute gradients of the attention scores
        Tensor<T> grad_attention_scores = grad_attention_heads[i].dot(values_heads[i].transpose());
        grad_attention_scores /= std::sqrt(static_cast<T>(head_dim));  // Scale gradient

        // Backward pass through the activation function
        activation->backward(grad_attention_scores);

        // Compute gradients w.r.t. queries, keys, and values
        grad_queries_heads[i] = grad_attention_scores.dot(keys_heads[i]);
        grad_keys_heads[i] = grad_attention_scores.transpose().dot(queries_heads[i]);
        grad_values_heads[i] = grad_attention_scores.dot(grad_attention_heads[i].transpose());
    }

    // Step 4: Concatenate the gradients of the heads
    Tensor<T> grad_queries = concat_heads(grad_queries_heads);
    Tensor<T> grad_keys = concat_heads(grad_keys_heads);
    Tensor<T> grad_values = concat_heads(grad_values_heads);

    // Step 5: Compute gradients of the linear projections (W_q, W_k, W_v)
    grad_W_q = input_cache.transpose().dot(grad_queries);
    grad_W_k = input_cache.transpose().dot(grad_keys);
    grad_W_v = input_cache.transpose().dot(grad_values);

    grad_b_q = grad_queries.sum(0);
    grad_b_k = grad_keys.sum(0);
    grad_b_v = grad_values.sum(0);

    // Step 6: Propagate gradients through the input data
    Tensor<T> grad_input_q = grad_queries.dot(W_q.transpose());
    Tensor<T> grad_input_k = grad_keys.dot(W_k.transpose());
    Tensor<T> grad_input_v = grad_values.dot(W_v.transpose());

    // Accumulate the gradients
    this->grad_W_k += grad_W_k;
    this->grad_W_q += grad_W_q;
    this->grad_W_v += grad_W_v;
    this->grad_W_o += grad_W_o;

    // Accumulate the biases gradients
    this->grad_b_k += grad_b_k;
    this->grad_b_q += grad_b_q;
    this->grad_b_v += grad_b_v;
    this->grad_b_o += grad_b_o;
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