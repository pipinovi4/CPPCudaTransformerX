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
      loss_function_(loss_function), optimizer_(optimizer), embedding_(),
      positional_encoder_(), encoder_layers_(), decoder_layers_(),
      output_encoder_layers_(), output_layer_linear_(), output_layer_softmax_() {
    // Initialize activation functions
    typename ActivationFunction<T>::ReLU relu;
    typename ActivationFunction<T>::Softmax softmax;

    // Initialize the embedding layers
    this->embedding_ = std::make_unique<Embedding<T>>(vocab_size, d_model, learning_rate_schedule);

    // Initialize the positional encoder
    positional_encoder_ = std::make_unique<Tokenizer<T>>(max_len - 2); // Subtract 2 for special tokens <sos> and <eos>

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
    this->output_layer_softmax_ = std::make_unique<PositionalWiseDenseLayer<T>>(d_model, vocab_size_, softmax);

    // Initialize optimizer parameters
    optimizer_->initialize_params(parameters_shape());
}

// Getter for the model parameters
template <typename T>
std::vector<std::reference_wrapper<Tensor<T>>> Transformer<T>::parameters() {
    // Return a vector containing references to all the parameters (weights and biases)
    std::vector<std::reference_wrapper<Tensor<T>>> params;

    // Add the parameters of the encoder layers
    for (auto& layer : encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    // Add the parameters of the output encoder layers
    for (auto& layer : output_encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    // Add the parameters of the decoder layers
    for (auto& layer : decoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    // Add the parameters of the output layers
    auto output_layer_linear_params = output_layer_linear_->parameters();
    params.insert(params.end(), output_layer_linear_params.begin(), output_layer_linear_params.end());

    auto output_layer_softmax_params = output_layer_softmax_->parameters();
    params.insert(params.end(), output_layer_softmax_params.begin(), output_layer_softmax_params.end());

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
    for (auto& layer : output_encoder_layers_) {
        auto layer_grads = layer->process_layer_->gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }
    for (auto& layer : decoder_layers_) {
        auto layer_grads = layer->process_layer_->gradients();
        grads.insert(grads.end(), layer_grads.begin(), layer_grads.end());
    }

    auto output_layer_linear_grads = output_layer_linear_->gradients();
    grads.insert(grads.end(), output_layer_linear_grads.begin(), output_layer_linear_grads.end());

    auto output_layer_softmax_grads = output_layer_softmax_->gradients();
    grads.insert(grads.end(), output_layer_softmax_grads.begin(), output_layer_softmax_grads.end());

    return grads;
}

// Getter for model parameters shape
template <typename T>
std::vector<std::vector<int>> Transformer<T>::parameters_shape() {
    // Vector to store the shapes of the parameters
    std::vector<std::vector<int>> shapes;

    // Add the parameters shape of the encoder layers
    for (auto& layer : encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }

    // Add the parameters shape of the output encoder layers
    for (auto& layer : output_encoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }

    // Add the parameters shape of the decoder layers
    for (auto& layer : decoder_layers_) {
        auto layer_params = layer->process_layer_->parameters();
        for (auto& param : layer_params) {
            shapes.push_back(param.get().shape());
        }
    }

    // Add the parameters shape of the output layers
    auto output_layer_linear_params = output_layer_linear_->parameters();
    for (auto& param : output_layer_linear_params) {
        shapes.push_back(param.get().shape());
    }

    auto output_layer_softmax_params = output_layer_softmax_->parameters();
    for (auto& param : output_layer_softmax_params) {
        shapes.push_back(param.get().shape());
    }

    return shapes;
}

template <typename T>
Tensor<T> Transformer<T>::forward(const Tensor<T>& src, const Tensor<T>& tgt) {
    // Initialize the source and target tensors
    Tensor<T> src_encoded = src; // Source tensor ["<SOS>", "am", "a", "student", "<PAD>"]
    Tensor<T> tgt_encoded = tgt; // Target tensor ["I", "am", "a", "student", "<EOS>"]

    // std::cout << "src_encoded shape: [";
    // for (auto dim : src_encoded.shape()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << "]" << std::endl;
    //
    //
    // std::cout << "tgt_encoded shape: [";
    // for (auto dim : tgt_encoded.shape()) {
    //     std::cout << dim << " ";
    // }
    // std::cout << "]" << std::endl;

    // // Pass the encoded data through the encoder layers
    // for (auto& layer : encoder_layers_) {
    //     src_encoded = layer->forward(src_encoded);
    // }
    //
    // // Pass the shifted target through the ResidualBlock with MASKED MULTIHEAD ATTENTION
    // for (auto& layer : output_encoder_layers_) {
    //     tgt_encoded = layer->forward(tgt_encoded, &src_encoded); // Use src_encoded as mask in case of cross-attention
    // }
    //
    // // Pass the output through the decoder layers
    // for (auto& layer : decoder_layers_) {
    //     tgt_encoded = layer->forward(tgt_encoded);
    // }
    //
    // // Pass the output through the pre final linear dense layer
    // tgt_encoded = output_layer_linear_->forward(tgt_encoded);
    //
    //
    // // Pass the output through the final softmax dense layer
    // tgt_encoded = output_layer_softmax_->forward(tgt_encoded);

    // return tgt_encoded; // Return the final output (processed tgt_encoded)
    return Tensor<T>();
}

template <typename T>
void Transformer<T>::backward(Tensor<T>& grad) {
    // Pass the gradient through the final softmax dense layer
    output_layer_softmax_->backward(grad);

    // Pass the gradient through the pre final linear dense layer
    output_layer_linear_->backward(grad);

    // Pass the gradient through the decoder layers
    for (auto& layer : decoder_layers_) {
        layer->backward(grad);
    }

    // Pass the gradient through the ResidualBlock with MASKED MULTIHEAD ATTENTION
    for (auto& layer : output_encoder_layers_) {
        layer->backward(grad);
    }

    // Pass the gradient through the encoder layers
    for (auto& layer : encoder_layers_) {
        layer->backward(grad);
    }
}

template <typename T>
void Transformer<T>::update(int epoch) {
    optimizer_->update(parameters(), gradients(), epoch);
}

// template <typename T>
// void Transformer<T>::load_weights(const std::string& filepath) {
//     // Load the weights of the model
// }
//
// template <typename T>
// void Transformer<T>::save_weights(const std::string& filepath) {
//     // Save the weights of the model
// }

template <typename T>
void Transformer<T>::train(const std::vector<std::vector<std::string>>& train_data, const int batch_size, const int n_epochs) {
    // Step 1: Tokenize the data once
    auto vocab = positional_encoder_->buildVocabulary(train_data);
    positional_encoder_->setVocabulary(vocab);

    std::vector<std::vector<int>> tokenized_data;
    for (const auto& sentence : train_data) {
        tokenized_data.push_back(positional_encoder_->textToIds(sentence));
    }

    // Step 2: Prepare batched data
    std::vector<Tensor<T>> batched_src;
    std::vector<Tensor<T>> batched_tgt;

    for (size_t i = 0; i < tokenized_data.size(); i += batch_size) {
        const size_t batch_end = std::min(i + batch_size, tokenized_data.size());
        std::vector<std::vector<int>> src_batch(tokenized_data.begin() + static_cast<int>(i), tokenized_data.begin() + static_cast<int>(batch_end));
        std::vector<std::vector<int>> tgt_batch = src_batch;

         // Shift tgt data in-place and adjust lengths
        for (int j = 0; j < tgt_batch.size(); ++j) {
            // Get the token ID for "<sos>" and "<eos>"
            int sos_id = positional_encoder_->textToIds({"<SOS>"})[0];
            int eos_id = positional_encoder_->textToIds({"<EOS>"})[0];
            int pad_id = positional_encoder_->textToIds({"<PAD>"})[0];

            // Insert <SOS> at the beginning of tgt_batch[j]
            src_batch[j].insert(tgt_batch[j].begin(), sos_id);

            // Add <EOS> at the end of tgt_batch[j]
            tgt_batch[j].push_back(eos_id);

            // Ensure src_batch[j] and tgt_batch[j] have the same length by padding
            while (src_batch[j].size() < tgt_batch[j].size()) {
                src_batch[j].push_back(pad_id);
            }
        }

        // Convert to Tensors
        batched_src.push_back(Tensor<T>(src_batch));
        batched_tgt.push_back(Tensor<T>(tgt_batch));
    }

    // Step 3: Training loop
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        T total_loss = 0.0;
        for (size_t i = 0; i < batched_src.size(); ++i) {
            // Pass the source and target data through the embedding layer
            Tensor<T> src_embedded = embedding_->forward(batched_src[i]);
            Tensor<T> tgt_embedded = embedding_->forward(batched_tgt[i]);

            // Forward pass
            Tensor<T> output = forward(src_embedded);
            T batch_loss = loss_function_->forward(output, tgt_embedded);
            total_loss += batch_loss;

            std::cout << "Batch: " << i << "/" << batched_src.size() << std::endl;

            // Backward pass
            Tensor<T> grad = loss_function_->backward(output, tgt_embedded);
            backward(grad);

            break;
        }
        // Update the weights after processing all batches
        update(epoch);

        // Print the average loss for the epoch
        std::cout << "Epoch: " << epoch << " Average Loss: " << total_loss / batched_src.size() << std::endl;
    }
}

// template <typename T>
// float Transformer<T>::evaluate(const std::vector<std::vector<std::string>>& val_data, int batch_size) {
//     return 0.0;
// }
//
template <typename T>
Tensor<T> Transformer<T>::predict(const std::vector<std::vector<std::string>>& src, const int max_len) {
   return forward(src);
}

template <typename T>
void Transformer<T>::load_weights(const std::string& filepath) {
    std::ifstream infile(filepath, std::ios::binary);
    if (!infile.is_open()) {
        throw std::runtime_error("Failed to open file for loading weights: " + filepath);
    }

    for (auto& param : this->parameters()) {
        Tensor<T>& tensor = param.get();
        infile.read(reinterpret_cast<char*>(tensor.data.data()), tensor.size() * sizeof(T));
        if (!infile) {
            throw std::runtime_error("Error reading weights from file: " + filepath);
        }
    }

    infile.close();
}

template <typename T>
void Transformer<T>::save_weights(const std::string& filepath) {
    std::ofstream outfile(filepath, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for saving weights: " + filepath);
    }

    for (auto& param : this->parameters()) {
        Tensor<T>& tensor = param.get();
        outfile.write(reinterpret_cast<const char*>(tensor.data.data()), tensor.size() * sizeof(T));
        if (!outfile) {
            throw std::runtime_error("Error writing weights to file: " + filepath);
        }
    }

    outfile.close();
}

#endif //TRANSFORMER_TPP
