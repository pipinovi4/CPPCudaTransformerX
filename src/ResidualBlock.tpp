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
void ResidualBlock<T, D>::backward(Tensor<T>& dout) {
    process_layer_.backward(dout);
    layer_norm_.backward(dout);
}

#endif // RESIDUALBLOCK_TPP