#ifndef RESIDUALBLOCK_TPP
#define RESIDUALBLOCK_TPP

#include "../include/ResidualBlock.h"

template <typename T, typename D>
ResidualBlock<T, D>::ResidualBlock(int d_model, float epsilon, D process_layer)
    : layer_norm_(d_model, epsilon), process_layer_(process_layer) {}

template <typename T, typename D>
Tensor<T> ResidualBlock<T, D>::forward(const Tensor<T>& input, const Tensor<T>* mask) {
    input_ = input;

    if constexpr (std::is_same<D, MultiHeadAttention<T>>::value) {
        if (mask == nullptr) {
            throw std::invalid_argument("Mask is required for MultiHeadAttention layer");
        }

        processed_ = process_layer_.forward(input, mask);
    } else {
        processed_ = process_layer_.forward(input);
    }

    output_ = input + processed_;
    return layer_norm_.forward(output_);
}

template <typename T, typename D>
Tensor<T> ResidualBlock<T, D>::backward(const Tensor<T>& dout) {
    Tensor<T> dnorm = layer_norm_.backward(dout);
    Tensor<T> dprocessed = dnorm;
    Tensor<T> dinput = dnorm;

    dprocessed = process_layer_.backward(dnorm);
    dinput = dnorm + dprocessed;

    return dinput;
}

#endif // RESIDUALBLOCK_TPP