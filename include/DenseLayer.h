#ifndef DENSELAYER_H
#define DENSELAYER_H

#pragma once
#include "../include/Tensor.h"
#include "../include/ActivationFunction.h"
#include "../include/Optimizer.h"

template <typename T>
class DenseLayer {
public:
    int inputUnits;
    int outputUnits;

    ActivationFunction<T>* activation;

    Tensor<T> weights;
    Tensor<T> bias;

    Tensor<T> input_cache;

    Tensor<T> weightGradients;
    Tensor<T> biasGradients;
    Tensor<T> inputGradients;
    void initializeWeights(Tensor<T>& inputWeights);

    DenseLayer(const int input_units, const int output_units, ActivationFunction<T>* activation, T biasInitValue = 0.0)
        : inputUnits(input_units), outputUnits(output_units), activation(activation) {
        weights = Tensor<T>({input_units, output_units});
        bias = Tensor<T>({output_units}, output_units);
        weightGradients = Tensor<T>({input_units, output_units});
        biasGradients = Tensor<T>({output_units}, output_units);
        bias.data.resize(output_units, T(biasInitValue));
        biasGradients.data.resize(output_units, T(biasInitValue));
        initializeWeights(weights);
    };

    Tensor<T> forward(const Tensor<T>& input);
    Tensor<T> backward(const Tensor<T>& grad_output);
    void updateParameters(Optimizer<T>* optimizer, size_t epoch);
};

#include "../src/DenseLayer.tpp"

#endif //DENSELAYER_H