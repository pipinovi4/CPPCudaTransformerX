/**
 * @file Embedding.h
 * @brief Defines the Embedding class, which implements a trainable embedding layer.
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#pragma once

#include "Layer.h"
#include "../include/Tensor.h"
#include "../include/Optimizer.h"

/**
 * @class Embedding
 * @brief A trainable embedding layer for converting discrete tokens into dense vectors.
 * 
 * This class represents an embedding layer, commonly used in natural language processing tasks
 * to map discrete vocabulary tokens into dense, continuous vector representations.
 * 
 * @tparam T The data type used for computations (e.g., float, double).
 */
template <typename T>
class Embedding final : public Layer<T> {
public:
    /**
     * @brief Constructs an Embedding layer with specified vocabulary size and embedding dimensions.
     * 
     * @param vocab_size The size of the vocabulary.
     * @param embedding_dims The dimensionality of the embedding vectors.
     * @param lr_schedule The learning rate schedule used for updating the embeddings.
     * @param init_func A function to initialize the weights of the embedding layer (optional).
     */
    Embedding(const int& vocab_size, const int& embedding_dims,
        typename Optimizer<T>::LearningRateSchedule& lr_schedule, std::function<void(Tensor<T>&)> init_func = nullptr);

    /**
     * @brief Performs the forward pass of the embedding layer.
     * 
     * @param input The input tensor containing indices of the vocabulary.
     * @return The output tensor containing the corresponding embedding vectors.
     */
    Tensor<T> forward(const Tensor<T>& input) override;

    /**
     * @brief Performs the backward pass of the embedding layer, computing gradients.
     * 
     * @param grad The gradient of the loss with respect to the output of this layer.
     */
    void backward(Tensor<T>& grad) override;

    /**
     * @brief Initializes the weights of the embedding layer.
     * 
     * This method sets the initial values of the embedding weights, either using a provided initialization function
     * or a default method if none is provided.
     */
    void initializeWeights();

    /**
     * @brief Updates the weights of the embedding layer based on the computed gradients.
     * 
     * @param epoch The current epoch of the training process.
     */
    void update(const int& epoch);

    /**
     * @brief Sets the gradients of the embedding layer to zero.
     * 
     * This method is typically called at the beginning of each training iteration to reset the gradients.
     */
    void zero_grad();

    /**
     * @brief Sets the weights of the embedding layer.
     * 
     * @param new_weights A tensor containing the new weights to be assigned to the embedding layer.
     */
    void setWeights(const Tensor<T>& new_weights);

    /**
     * @brief Retrieves the weights of the embedding layer.
     * 
     * @return A reference to the tensor containing the embedding weights.
     */
    Tensor<T>& getWeights();

    /**
     * @brief Retrieves the gradients of the embedding layer.
     * 
     * @return A reference to the tensor containing the gradients with respect to the embedding weights.
     */
    Tensor<T>& getGrad();

private:
    Optimizer<float>::LearningRateSchedule& lr_schedule; ///< Learning rate schedule for updating weights.

    Tensor<T> weights; ///< The embedding weights.
    Tensor<T> grad;    ///< Gradients with respect to the embedding weights.

    Tensor<T> input;   ///< Cached input tensor for use in the backward pass.
    Tensor<T> output;  ///< Cached output tensor after the forward pass.

    int vocab_size;    ///< The size of the vocabulary.
    int embedding_dims; ///< The number of dimensions for each embedding vector.

    Tensor<T> input_cache; ///< Cache of the input tensor for use in the backward pass.
};

#include "../src/Embedding.tpp"

#endif // EMBEDDING_H
