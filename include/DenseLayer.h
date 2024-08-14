#ifndef DENSELAYER_H
#define DENSELAYER_H

#pragma once
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "../include/DenseLayer.h"

template <typename T>
class DenseLayer {
public:
    DenseLayer(int input_units, int output_units, ActivationFunction<T>* activation = new typename ActivationFunction<T>::Linear(), T biasInitValue = 0.0);

    void initializeWeights(Tensor<T>& inputWeights);

    Tensor<T> forward(Tensor<T>& input);
    void backward(Tensor<T>& grad_output);

    int inputUnits;
    int outputUnits;

    ActivationFunction<T>* activation;

    Tensor<T> weights;
    Tensor<T> bias;

    Tensor<T> input_cache;

    Tensor<T> weightGradients;
    Tensor<T> biasGradients;
    Tensor<T> inputGradients;
};

#include "../src/DenseLayer.tpp"

#endif //DENSELAYER_H