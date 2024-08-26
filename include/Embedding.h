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
#include <stdexcept>
#include "eigen3/Eigen/Dense"

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
     */
    Embedding(const int& vocab_size, const int& embedding_dims,
    typename Optimizer<T>::LearningRateSchedule& lr_schedule);

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
    * @brief Initializes the weights of the embedding layer using Xavier/Glorot initialization.
    *
    * This initialization method is chosen for its effectiveness in keeping the scale of gradients
    * roughly the same across layers, which helps mitigate issues like vanishing or exploding gradients
    * during training.
    */
    void initializeWeights();

    /**
     * @brief Sets the gradients of the embedding layer to zero.
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
    std::vector<std::reference_wrapper<Tensor<float>>> parameters() override;

    /**
     * @brief Retrieves the gradients of the embedding layer.
     *
     * @return A reference to the tensor containing the gradients with respect to the embedding weights.
     */
    std::vector<std::reference_wrapper<Tensor<float>>> gradients() override;

private:
    typename Optimizer<T>::LearningRateSchedule& lr_schedule_; ///< Learning rate schedule for updating weights.

    Tensor<T> weights_; ///< The embedding weights.
    Tensor<T> grad_;    ///< Gradients with respect to the embedding weights.

    int vocab_size_;    ///< The size of the vocabulary.
    int embedding_dims_; ///< The number of dimensions for each embedding vector.

    Tensor<T> input_cache_; ///< Cache of the input tensor for use in the backward pass.
};

#include "../src/Embedding.tpp"

#endif // EMBEDDING_H
