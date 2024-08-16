// include/ResidualBlock.h
#ifndef RESIDUALBLOCK_H
#define RESIDUALBLOCK_H

#include "LayerNorm.h"
#include "Tensor.h"
#include "MultiHeadAttention.h"
#include "PositionalWiseDenseLayer.h"

template <typename T, typename D>
class ResidualBlock {
public:
    explicit ResidualBlock(int d_model, float epsilon, D process_layer);

    Tensor<T> forward(const Tensor<T>& input, const Tensor<T>* mask = nullptr);

    Tensor<T> backward(const Tensor<T>& dout);

private:
    LayerNorm<T> layer_norm_;
    D process_layer_;
    Tensor<T> input_;
    Tensor<T> processed_;
    Tensor<T> output_;
};

#include "../src/ResidualBlock.tpp"

#endif // RESIDUALBLOCK_H