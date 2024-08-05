#ifndef DIGITRECOGNIZER_TPP
#define DIGITRECOGNIZER_TPP

#include "DigitRecognizer.h"
#include <thread>
#include <chrono>
#include "utils/readUInt32.h"

template <typename T>
Tensor<T> oneHotEncode(int target, int num_classes) {
    Tensor<T> one_hot({num_classes});
    if (target >= 0 && target < num_classes) {
        one_hot.data[target] = 1;
    }
    return one_hot;
}

template <typename T>
DigitRecognizer<T>::DigitRecognizer(int input_dims, int hidden_dims, int output_dims, LossFunction<T>* loss_function)
    : input_dims(input_dims), hidden_dims(hidden_dims), output_dims(output_dims), loss_function(loss_function) {
    layers.push_back(DenseLayer<T>(input_dims, hidden_dims, new typename ActivationFunction<T>::ReLU(), 0.0));
    layers.push_back(DenseLayer<T>(hidden_dims, hidden_dims, new typename ActivationFunction<T>::ReLU(), 0.0));
    layers.push_back(DenseLayer<T>(hidden_dims, output_dims, new typename ActivationFunction<T>::Softmax(), 0.0));
}

template <typename T>
Tensor<T> DigitRecognizer<T>::forward(const Tensor<T>& input) {
    Tensor<T> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return output;
}

template <typename T>
void DigitRecognizer<T>::backward(Tensor<T>& grad_output) {
    for (auto layer = layers.rbegin(); layer != layers.rend(); ++layer) {
        layer->backward(grad_output);
    }
}

template <typename T>
void DigitRecognizer<T>::train(Tensor<T>& train_data, Tensor<T>& train_labels, const size_t num_epochs, Optimizer<T>* optimizer, const size_t batch_size) {
    const size_t num_batches = train_data.shape()[0] / batch_size;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        T loss = 0;
        for (size_t batch = 0; batch < num_batches; ++batch) {
            std::vector<Tensor<T>> accumulated_gradients(parameters().size());
            size_t correct_predictions = 0;

            for (size_t i = 0; i < batch_size; ++i) {
                size_t index = batch * batch_size + i;
                Tensor<T> input = train_data[{static_cast<int>(index)}];
                Tensor<T> target = oneHotEncode<T>(train_labels.data[index], output_dims);

                Tensor<T> output = forward(input);
                loss += loss_function->forward(output, target);

                size_t predicted_label = std::distance(output.data.begin(), std::max_element(output.data.begin(), output.data.end()));
                if (predicted_label == static_cast<int>(train_labels.data[index])) {
                    correct_predictions++;
                }

                Tensor<T> grad_loss = loss_function->backward(output, target);
                backward(grad_loss);

                for (size_t j = 0; j < gradients().size(); ++j) {
                    if (i == 0) {
                        accumulated_gradients[j] = gradients()[j];
                    } else {
                        accumulated_gradients[j] = accumulated_gradients[j] + gradients()[j];
                    }
                }
            }

            // Average the accumulated gradients over the batch
            for (auto& grad : accumulated_gradients) {
                // Explicitly perform division for each element in the gradient tensor
                for (auto& value : grad.data) {
                    value /= static_cast<T>(batch_size);
                }
            }

            std::vector<std::reference_wrapper<Tensor<T>>> accumulated_grad_refs;
            for (auto& grad : accumulated_gradients) {
                accumulated_grad_refs.push_back(std::ref(grad));
            }

            optimizer->update(parameters(), accumulated_grad_refs, epoch);

            const float batch_accuracy = static_cast<float>(correct_predictions) / static_cast<float>(batch_size);
            T batch_loss = loss / static_cast<T>(batch_size);
            std::cout << "Epoch: " << epoch + 1 << " Batch: " << batch + 1 << " Loss: " << batch_loss << " Accuracy: " << batch_accuracy * 100.0f << "%" << std::endl;

            loss = 0;
        }
    }
}

template <typename T>
std::vector<std::vector<std::uint8_t>> DigitRecognizer<T>::loadMNISTImages(const std::string& filename) {
    std::ifstream imageStream(filename, std::ios::binary);
    if (!imageStream.is_open()) {
        throw std::runtime_error("Could not open image file.");
    }

    // Read dimensions
    readUInt32(imageStream); // magic_number
    const std::uint32_t num_images = readUInt32(imageStream);
    const std::uint32_t num_rows = readUInt32(imageStream);
    const std::uint32_t num_cols = readUInt32(imageStream);

    // Read image data
    std::vector<std::vector<std::uint8_t>> images(num_images, std::vector<std::uint8_t>(num_rows * num_cols));
    for (std::uint32_t i = 0; i < num_images; ++i) {
        imageStream.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
    }
    imageStream.close();

    return images;
}

template <typename T>
std::vector<std::uint8_t> DigitRecognizer<T>::loadMNISTLabels(const std::string& filename) {
    std::ifstream labelStream(filename, std::ios::binary);
    if (!labelStream.is_open()) {
        throw std::runtime_error("Could not open label file.");
    }

    // Read number of labels
    readUInt32(labelStream); // magic_number
    const std::uint32_t num_labels = readUInt32(labelStream);

    // Read label data
    std::vector<std::uint8_t> labels(num_labels);
    labelStream.read(reinterpret_cast<char*>(labels.data()), num_labels);
    labelStream.close();

    return labels;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> DigitRecognizer<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> param_refs;
    for (auto& layer : layers) {
        param_refs.push_back(std::ref(layer.weights));
        param_refs.push_back(std::ref(layer.bias));
    }
    return param_refs;
}

template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> DigitRecognizer<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grad_refs;
    for (auto& layer : layers) {
        grad_refs.push_back(std::ref(layer.weightGradients));
        grad_refs.push_back(std::ref(layer.biasGradients));
    }
    return grad_refs;
}

#endif //DIGITRECOGNIZER_TPP