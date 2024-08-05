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
        weightGradients = Tensor<T>({input_units, output_units});

        bias = Tensor<T>({output_units});
        biasGradients = Tensor<T>({output_units});

        initializeWeights(weights);
        bias.fill(biasInitValue);
    }

    Tensor<T> forward(Tensor<T>& input);
    void backward(Tensor<T>& grad_output);
};

#include "../src/DenseLayer.tpp"

#endif //DENSELAYER_H