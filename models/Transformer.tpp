#ifndef TRANSFORMER_TPP
#define TRANSFORMER_TPP

#include "Transformer.h"

template <typename T>
Transformer<T>::Transformer(const int vocab_size, const int d_model, const int n_heads, const int d_ff,
    const int max_len, const float dropout, const float label_smoothing, const int warmup_steps,
    typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule,
    LossFunction<T>* loss_function, Optimizer<T>* optimizer)
    : vocab_size_(vocab_size), d_model_(d_model), n_heads_(n_heads), d_ff_(d_ff),
      max_len_(max_len), dropout_(dropout), label_smoothing_(label_smoothing),
      warmup_steps_(warmup_steps), learning_rate_schedule_(learning_rate_schedule),
      loss_function_(loss_function), optimizer_(optimizer), input_embedding_(),
      output_embedding_(), positional_encoder_(), encoder_layers_(), decoder_layers_(),
      output_encoder_layers_(), output_layers_() {
    // Initialize activation functions
    typename ActivationFunction<T>::Linear linear;
    typename ActivationFunction<T>::ReLU relu;
    typename ActivationFunction<T>::Softmax softmax;

    // Initialize the embedding layers
    this->input_embedding_ = std::make_unique<Embedding<T>>(vocab_size, d_model, learning_rate_schedule);
    this->output_embedding_ = std::make_unique<Embedding<T>>(vocab_size, d_model, learning_rate_schedule);

    // Initialize the positional encoder
    positional_encoder_ = std::make_unique<PositionalEncoder<T>>(max_len, d_model);

    // Initialize the encoder layers
    this->encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
    d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, &relu))));
    this->encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new PositionalWiseDenseLayer<T>(d_model, d_ff, relu, 0))));

    // Initialize the decoder layers
    this->decoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, &relu))));
    this->decoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new PositionalWiseDenseLayer<T>(d_model, d_ff, relu, 0))));

    // Initialize output encoder layers for shifted target (MASKED MULTIHEAD ATTENTION)
    this->output_encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, &relu))));

    // Initialize the final dense layers
    PositionalWiseDenseLayer<T> linear_wise_dense_layer(d_model, vocab_size, linear);
    PositionalWiseDenseLayer<T> softmax_wise_dense_layer(d_model, vocab_size, softmax);

    // Initialize the final dense layers
    output_layers_.emplace_back(std::move(std::make_unique<PositionalWiseDenseLayer<T>*>(&linear_wise_dense_layer)));
    output_layers_.emplace_back(std::move(std::make_unique<PositionalWiseDenseLayer<T>*>(&softmax_wise_dense_layer)));

    // Initialize optimizer parameters
    optimizer_->initialize_params(parameters_shape());
}

// Getter for the model parameters
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> Transformer<T>::parameters() {
    std::vector<std::reference_wrapper<Tensor<T>>> params;
    for (auto& layer : encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    for (auto& layer : decoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    for (auto& layer : output_encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

// Getter for the model gradients
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> Transformer<T>::gradients() {
    std::vector<std::reference_wrapper<Tensor<T>>> grads;
    for (auto& layer : encoder_layers_) {
        auto layer_grads = layer->process_layer_->gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    for (auto& layer : decoder_layers_) {
        auto layer_grads = layer->process_layer_->gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    for (auto& layer : output_encoder_layers_) {
        auto layer_grads = layer->process_layer_->gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    return grads;
}

// Getter for model parameters shape
template <typename T>
std::vector<std::vector<int>> Transformer<T>::parameters_shape() {
    std::vector<std::vector<int>> shapes;
    for (auto& layer : encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }
    for (auto& layer : decoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }
    for (auto& layer : output_encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }
    return shapes;
}


#endif //TRANSFORMER_TPP
