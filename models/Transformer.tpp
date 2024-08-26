#ifndef TRANSFORMER_TPP
#define TRANSFORMER_TPP

#include "Transformer.h"

template <typename T>
Transformer<T>::Transformer(const int vocab_size, const int d_model, const int n_heads, const int d_ff,
    const int max_len, const float dropout, const float label_smoothing, const int warmup_steps,
    typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule,
    LossFunction<T>* loss_function, Optimizer<T>* optimizer, std::vector<std::string> vocab)
    : vocab_size_(vocab_size), d_model_(d_model), n_heads_(n_heads), d_ff_(d_ff),
      max_len_(max_len), dropout_(dropout), label_smoothing_(label_smoothing),
      warmup_steps_(warmup_steps), learning_rate_schedule_(learning_rate_schedule),
      loss_function_(loss_function), optimizer_(optimizer), embedding_(),
      positional_encoder_(), encoder_layers_(), decoder_layers_(),
      output_encoder_layers_(), output_layer_softmax_() {
    // Initialize activation functions
    typename ActivationFunction<T>::ReLU relu;
    typename ActivationFunction<T>::Softmax softmax;

    // Initialize the embedding layers
    this->embedding_ = std::make_unique<Embedding<T>>(vocab_size, d_model, learning_rate_schedule);

    // Initialize the positional encoder
    positional_encoder_ = std::make_unique<Tokenizer<T>>(max_len - 1); // Exclude <EOS> token for tgt and <SOS> token for src

    // build vocabulary
    const std::unordered_map<std::string, int> vocab_map = Tokenizer<T>::buildVocabulary(vocab);
    positional_encoder_->setVocabulary(vocab_map);

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
    this->output_layer_softmax_ = std::make_unique<DenseLayer<T>>(d_model, vocab_size_, new typename ActivationFunction<T>::Softmax());

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

    auto output_layer_softmax_params = output_layer_softmax_->parameters();
    for (auto& param : output_layer_softmax_params) {
        shapes.push_back(param.get().shape());
    }

    return shapes;
}

template <typename T>
Tensor<T> Transformer<T>::forward(const Tensor<T>& src, const Tensor<T>& tgt) {
    // Pass the input through the embedding layer
    Tensor<T> src_embeded = embedding_->forward(src);
    Tensor<T> tgt_embeded = embedding_->forward(tgt);

    // Pass the encoded data through the encoder layers
    for (auto& layer : encoder_layers_) {
        src_embeded = layer->forward(src_embeded);
    }

    // Pass the shifted target through the ResidualBlock with MASKED MULTIHEAD ATTENTION
    for (auto& layer : output_encoder_layers_) {
        tgt_embeded = layer->forward(tgt_embeded, &src_embeded); // Use src_encoded as mask in case of cross-attention
    }

    // Pass the output through the decoder layers
    for (auto& layer : decoder_layers_) {
        tgt_embeded = layer->forward(tgt_embeded);
    }

    // Pass the output through the final softmax dense layer
    tgt_embeded = output_layer_softmax_->forward(tgt_embeded);

    return tgt_embeded; // Return the final output (processed tgt_encoded)
}

template <typename T>
void Transformer<T>::backward(Tensor<T>& grad) {
    // Pass the gradient through the final softmax dense layer
    output_layer_softmax_->backward(grad);

    // Pass the gradient through the decoder layers in reverse order
    for (auto it = decoder_layers_.rbegin(); it != decoder_layers_.rend(); ++it) {
        (*it)->backward(grad);
    }

    // Pass the gradient through the ResidualBlock with MASKED MULTIHEAD ATTENTION in reverse order
    for (auto it = output_encoder_layers_.rbegin(); it != output_encoder_layers_.rend(); ++it) {
        (*it)->backward(grad);
    }

    // Pass the gradient through the encoder layers in reverse order
    for (auto it = encoder_layers_.rbegin(); it != encoder_layers_.rend(); ++it) {
        (*it)->backward(grad);
    }

    // Pass the gradient through the embedding layer
    embedding_->backward(grad);
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
void Transformer<T>::train(const std::vector<std::vector<std::string>>& train_data, const int n_epochs, const int batch_size) {
    // Step 1: Tokenize the data once
    const int num_sentences = static_cast<int>(train_data.size());

    // Tokenize the data
    std::vector<std::vector<int>> tokenized_data;
    for (const auto& sentence : train_data) {
        tokenized_data.push_back(positional_encoder_->textToIds(sentence));
    }

    // Step 2: Prepare batched data
    std::vector<Tensor<T>> src(num_sentences);
    std::vector<Tensor<T>> tgt(num_sentences);
    const int pad_token = positional_encoder_->textToIds({"<pad>"})[0];
    const int sos_token = positional_encoder_->textToIds({"<sos>"})[0];
    int eos_token = positional_encoder_->textToIds({"<eos>"})[0];

    for (size_t i = 0; i < num_sentences; ++i) {
        std::vector<int> src_sentence = tokenized_data[i];
        std::vector<int> tgt_sentence = tokenized_data[i];

        // Add <SOS> to the beginning and <EOS> to the end
        src_sentence.insert(src_sentence.begin(), sos_token);
        tgt_sentence.push_back(eos_token);

        // Convert to Tensor and pad the remaining length
        src[i] = Tensor<T>({max_len_}, src_sentence);
        tgt[i] = Tensor<T>({max_len_}, tgt_sentence);
        std::fill(src[i].data.begin() + src_sentence.size(), src[i].data.end(), pad_token);
        std::fill(tgt[i].data.begin() + tgt_sentence.size(), tgt[i].data.end(), pad_token);
    }

    // Step 3: Training loop with batch processing
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        T total_loss = 0.0;

            for (int batch_start = 0; batch_start < batch_size; batch_start += batch_size) {
            T batch_loss = 0.0;

            // Initialize accumulated gradients to zero
            Tensor<T> accumulated_grads({vocab_size_});

            #pragma omp parallel for reduction(+:batch_loss)
            for (int i = 0; i < batch_size; ++i) {
                int idx = batch_start + i;
                if (idx >= num_sentences) break; // Prevent out-of-bounds access

                Tensor<T> filled_src = src[idx];
                Tensor<T> filled_tgt = tgt[idx];

                Tensor<T> output;
                Tensor<T> target_tensor({vocab_size_});

                for (int j = 0; j < max_len_; j++) {
                    if (filled_tgt.data[j] == eos_token) break;

                    // Correct label for the current token
                    T true_label = src[batch_start + i].data[j + 1];
                    filled_src.data[j + 1] = src[batch_start + i].data[j];

                    // Forward pass for the current token
                    output = forward(filled_src, filled_tgt);

                    // Calculate loss for the current token
                    target_tensor.fill(T(0));
                    target_tensor.data[filled_tgt.data[j]] = T(1);
                    T current_loss = loss_function_->forward(output, target_tensor);
                    batch_loss += current_loss;

                    // Accumulate gradients for the current token
                    Tensor<T> grad = loss_function_->backward(output, target_tensor);
                    #pragma omp critical
                    accumulated_grads = accumulated_grads + grad;

                    Tensor<T> predicted = output.argmax();
                    filled_tgt.data[j] = true_label;
                    filled_src.data[j + 1] = true_label;

                    // std::cout << "Epoch: " << epoch + 1 << " | Batch: " << (batch_start / batch_size) + 1
                    // << " | Sentence: " << idx + 1 << " | Token: " << j + 1 << " True label: " << true_label
                    // << " Predicted label: " << predicted.data[0] << " " << std::endl;
                }
            }

            // Perform a single backward pass and update parameters after processing the entire batch
            backward(accumulated_grads);
            update(epoch);

            total_loss += batch_loss;

            std::cout << "Epoch: " << epoch + 1 << " | Batch: " << (batch_start / batch_size) + 1
                      << " | Batch Loss: " << batch_loss / batch_size
                      << " | Cumulative Loss: " << total_loss / ((batch_start / batch_size) + 1) << std::endl;
        }
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
        std::cout << "Saving weights: [";
        for (auto& shape : param.get().shape()) {
            std::cout << shape << " ";
        }
        std::cout << "]" << std::endl;
        Tensor<T>& tensor = param.get();
        outfile.write(reinterpret_cast<const char*>(tensor.data.data()), tensor.size() * sizeof(T));
        if (!outfile) {
            throw std::runtime_error("Error writing weights to file: " + filepath);
        }
    }

    outfile.close();
}

#endif //TRANSFORMER_TPP
