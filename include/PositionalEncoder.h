#ifndef POSITIONALENCODER_H
#define POSITIONALENCODER_H

#pragma once
#include "../include/Tensor.h"

class PositionalEncoder {
public:
    PositionalEncoder(const int& max_seq_len, const int& hidden_dim);

    Tensor<float> forward(const Tensor<float>& input_data);
    void backward(const Tensor<float>& grad_data);

    Tensor<float> positional_encodings;

    int maxSeqLen;
    int hiddenDim;
};

#include "../src/PositionalEncoder.tpp"

#endif // POSITIONALENCODER_H