#ifndef DIGIT_RECOGNIZER_H
#define DIGIT_RECOGNIZER_H

#include <vector>
#include "../include/Tensor.h"
#include "../include/Optimizer.h"
#include "../include/LossFunction.h"
#include "../include/DenseLayer.h"

template <typename T>
class DigitRecognizer {
public:
    /**
     * \brief Constructs a DigitRecognizer model.
     *
     * \param input_dims Number of input dimensions.
     * \param hidden_dims Number of hidden layer dimensions.
     * \param output_dims Number of output dimensions.
     * \param loss_function Pointer to the loss function.
     */
    DigitRecognizer(int input_dims, int hidden_dims, int output_dims, LossFunction<T>* loss_function);

    /**
     * \brief Performs a forward pass through the network.
     *
     * \param input Input tensor.
     * \return void.
     */
     Tensor<T> forward(const Tensor<T>& input);

    /**
     * \brief Performs a backward pass through the network.
     *
     * \param grad_output Gradient of the output.
     * \return void.
     */
    void backward(Tensor<T>& grad_output);

   /**
    * \brief Trains the model using the provided training data and labels.
    *
    * \param train_data Training data tensor.
    * \param train_labels Training labels tensor.
    * \param num_epochs Number of epochs to train.
    * \param optimizer Pointer to the optimizer used for updating the parameters.
    * \param batch_size Size of the mini-batches used for training.
    */
    void train(Tensor<T>& train_data, Tensor<T>& train_labels, size_t num_epochs, Optimizer<T>* optimizer, size_t batch_size);

    /**
        * \brief Returns a vector of references to the parameters of each layer.
        *
        * \return Vector of references to the parameters.
        */
    std::vector<std::reference_wrapper<Tensor<T>>> parameters();

   /**
      * \brief Returns a vector of references to the graient of the parametrs of each layer.
      *
      * \return Vector of references to the gradient of the paranerts.
      */
   std::vector<std::reference_wrapper<Tensor<T>>> gradients();

private:
    int input_dims; ///< Number of input dimensions.
    int hidden_dims; ///< Number of hidden layer dimensions.
    int output_dims; ///< Number of output dimensions.
    LossFunction<T>* loss_function{}; ///< Pointer to the loss function.
    std::vector<DenseLayer<T>> layers{}; ///< Vector of layers in the network.
};

#include "DigitRecognizer.tpp"

#endif // DIGIT_RECOGNIZER_H