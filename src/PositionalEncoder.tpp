#ifndef POSITIONALENCODER_TPP
#define POSITIONALENCODER_TPP

#include "../include/PositionalEncoder.h"
#include <cmath>

/**
 * @brief Constructor for the PositionalEncoder class.
 *
 * Initializes the positional encoder with the given maximum sequence length and hidden dimension.
 * It also creates the positional encodings based on the sine and cosine functions.
 *
 * @param max_seq_len Maximum sequence length for the input data.
 * @param hidden_dim Dimensionality of the hidden layer (embedding dimension).
 */
template <typename T>
PositionalEncoder<T>::PositionalEncoder(int max_seq_len, int hidden_dim)
    : maxSeqLen(max_seq_len), hiddenDim(hidden_dim) {
    // Initialize the positional encodings tensor
    positional_encodings = Tensor<T>({max_seq_len, hidden_dim}, max_seq_len * hidden_dim);
    initialize_positional_encodings(); // Initialize positional encodings
}

/**
 * @brief Initializes the positional encodings tensor.
 *
 * This method fills the positional encodings tensor with values based on sine and cosine functions.
 * The even indices use the sine function, while the odd indices use the cosine function.
 * This ensures that each position is encoded uniquely.
 */
template <typename T>
void PositionalEncoder<T>::initialize_positional_encodings() {
    for (int i = 0; i < maxSeqLen; ++i) {
        for (int j = 0; j < hiddenDim; ++j) {
            if (j % 2 == 0) {
                // Use sine for even indices
                positional_encodings.data[i * hiddenDim + j] = std::sin(static_cast<T>(i) / std::pow(10000, static_cast<T>(j) / hiddenDim));
            } else {
                // Use cosine for odd indices
                positional_encodings.data[i * hiddenDim + j] = std::cos(static_cast<T>(i) / std::pow(10000, static_cast<T>(j - 1) / hiddenDim));
            }
        }
    }
}

/**
 * @brief Applies positional encoding to the input data.
 *
 * This method adds the positional encodings to the input data to inject position-specific information
 * into the model's embeddings. This step is essential in transformer models to provide the model with
 * information about the order of the sequence.
 *
 * @param input_data The input tensor containing the embeddings of the sequence data.
 * @return Tensor<T> The output tensor with positional encodings applied.
 */
template <typename T>
void PositionalEncoder<T>::forward(Tensor<T>& input_data) {
    input_data.add(positional_encodings);
}

#endif // POSITIONALENCODER_TPP
