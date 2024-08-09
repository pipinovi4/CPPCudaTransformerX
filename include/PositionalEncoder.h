#ifndef POSITIONALENCODER_H
#define POSITIONALENCODER_H

#pragma once
#include "../include/Tensor.h"

template <typename T>
class PositionalEncoder {
public:
    // Constructor to initialize the positional encoder with max sequence length and hidden dimension
    PositionalEncoder(int max_seq_len, int hidden_dim);

    // Forward pass to apply positional encoding to the input data
    void forward(Tensor<T>& input_data);

private:
    int maxSeqLen; // Maximum sequence length
    int hiddenDim; // Dimensionality of the hidden layer

    Tensor<T> positional_encodings; // Positional encodings tensor

    void initialize_positional_encodings(); // Method to initialize encodings
};

#include "../src/PositionalEncoder.tpp"

#endif // POSITIONALENCODER_H
