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
    positional_encoder_ = std::make_unique<Tokenizer<T>>(max_len); // Exclude <EOS> token for tgt and <SOS> token for src

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


// template <typename T>
// Tensor<T> Transformer<T>::forward(const Tensor<T>& src, const Tensor<T>& tgt) {
//     // Pass the input through the embedding layer
//     Tensor<T> src_embeded = embedding_->forward(src);
//     Tensor<T> tgt_embeded = embedding_->forward(tgt);
//
//     // Pass the encoded data through the encoder layers
//     for (auto& layer : encoder_layers_) {
//         src_embeded = layer->forward(src_embeded);
//     }
//
//     // Pass the shifted target through the ResidualBlock with MASKED MULTIHEAD ATTENTION
//     for (auto& layer : output_encoder_layers_) {
//         tgt_embeded = layer->forward(tgt_embeded, &src_embeded); // Use src_encoded as mask in case of cross-attention
//     }
//
//     // Pass the output through the decoder layers
//     for (auto& layer : decoder_layers_) {
//         tgt_embeded = layer->forward(tgt_embeded);
//     }
//
//     // Pass the output through the final softmax dense layer
//     tgt_embeded = output_layer_softmax_->forward(tgt_embeded);
//
//     return tgt_embeded; // Return the final output (processed tgt_encoded)
// }

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

template <typename T>
void Transformer<T>::train(const std::vector<std::vector<std::string>>& data, const int n_epochs, const int batch_size) {
    // Step 1: Tokenize the data once
    const int num_sentences = static_cast<int>(data.size());
    const int pad_token = positional_encoder_->textToIds({"<pad>"})[0];

    std::vector<std::vector<Tensor<T>>> processed_data = convert_to_tensor(data);

    // Step 2: Initialize variables for training
    std::vector<Tensor<T>> src = processed_data[0];
    std::vector<Tensor<T>> tgt = processed_data[1];
    std::vector<Tensor<T>> true_labels = processed_data[2];

    // Initialize accumulated gradients to zero
    Tensor<T> true_labels_tokens({vocab_size_});

    // Timer to control the batch processing time
    using namespace std::chrono;
    steady_clock::time_point start_time, end_time;
    duration<float> elapsed{};

    // Step 3: Training loop with batch processing
    for (int epoch = 0; epoch < 5; ++epoch) {
        for (int batch_start = 0; batch_start < num_sentences; batch_start += batch_size) {
            start_time = steady_clock::now();  // Start timing
            // Initialize the batch loss
            T batch_loss = T(0);
            #pragma omp parallel for reduction(+:batch_loss)
            for (int i = 0; i < batch_size; ++i) {
                auto start_process_sentence = steady_clock::now();
                // Get the current sentence index
                int idx = batch_start + i;
                if (idx >= num_sentences) break; // Prevent out-of-bounds access

                // Initialize the predicted and true labels tokens
                Tensor<T> masked_src = src[idx];
                Tensor<T> masked_tgt = tgt[idx];
                Tensor<T> true_label_subtensor = true_labels[idx];

                // Initialize the first token of the target tensor
                for (int j = 1; j < max_len_; j++) {
                    // Get the true label for the current token
                    T true_label = true_label_subtensor.data[j];

                    // Forward pass for the current token
                    Tensor<T> output = forward(masked_src, masked_tgt);

                    // Compute predicted token
                    Tensor<T> predicted = output.argmax();

                    // Update the predicted and true labels tokens for compute grad and loss
                    true_labels_tokens.data[true_label] += 1;

                    // Update src and tgt tensors for the next token
                    masked_src.data[j + 1] = true_label;
                    masked_tgt.data[j] = true_label;

                    T loss = loss_function_->forward(output, true_labels_tokens);
                    end_time = steady_clock::now();
                    elapsed = duration_cast<microseconds>(end_time - start_time);
                    // Perform a single backward pass and update parameters after processing the entire batch
                    Tensor<T> grad = loss_function_->backward(output, true_labels_tokens);
                    backward(grad);

                    true_labels_tokens.fill(0);
                    if (true_label == pad_token) break;
                }
            }

            update(epoch);

            // Print the batch loss
            std::cout << "Epoch: " << epoch + 1 << " | Batch: " << (batch_start / batch_size) + 1
                      << " | Batch Loss: " << std::endl;
        }
    }
}

template <typename T>
float Transformer<T>::evaluate(const std::vector<std::vector<std::string>>& val_data, int batch_size) {
    return 0.0;
}

template <typename T>
Tensor<T> Transformer<T>::predict(const std::vector<std::vector<std::string>>& src, const int max_len) {
    // Step 1: Tokenize the data once
    std::cout << "Predicting..." << std::endl;
    const int num_sentence = static_cast<int>(src.size());

    // Convert the input data to tokenized tensors
    std::vector<std::vector<Tensor<T>>> processed_data = convert_to_tensor(src);

    // Step 2: Initialize variables for prediction
    std::vector<Tensor<T>> src_tensors = processed_data[0];
    std::vector<Tensor<T>> tgt_tensors = processed_data[1];
    std::vector<Tensor<T>> true_labels_tensors = processed_data[2];

    // Initialize the predicted tokens tensor
    Tensor<T> predicted_tokens({max_len_});
    Tensor<T> true_labels_tokens({max_len_});

    // Predict the output for each sentence
    for (int i = 0; i < num_sentence; ++i) {
        // Initialize the predicted and true labels tokens
        Tensor<T> masked_src = src_tensors[i];
        Tensor<T> masked_tgt = tgt_tensors[i];
        Tensor<T> true_label_subtensor = true_labels_tensors[i];

        // Initialize the first token of the target tensor
        for (int j = 1; j < max_len_; j++) {
            // Get the true label for the current token
            T true_label = true_label_subtensor.data[j];

            // Forward pass for the current token
            Tensor<T> output = forward(masked_src, masked_tgt);

            // Compute predicted token
            Tensor<T> predicted = output.argmax();

            // Update the predicted and true labels tokens for compute grad and loss
            predicted_tokens.data[j-1] = predicted.data[0];
            true_labels_tokens.data[j-1] = true_label;

            // Update src and tgt tensors for the next token
            masked_src.data[j + 1] = true_label;
            masked_tgt.data[j] = true_label;
        }
    }

    // Compute the loss for the predicted tokens
    T loss = loss_function_->forward(predicted_tokens, true_labels_tokens);

    // Print the loss
    std::cout << "Predicted Loss: " << loss << std::endl;

    // Compute the accuracy
    T accuracy = predicted_tokens.data[predicted_tokens.argmax().data[0]] / true_labels_tokens.data[true_labels_tokens.argmax().data[0]];

    // Print the accuracy
    std::cout << "Predicted Accuracy: " << accuracy << std::endl;

    // Print the true labels
    std::cout << "True labels: " << std::endl;
    true_labels_tokens.print();

    // Print the predicted tokens
    std::cout << "Predicted tokens: " << std::endl;
    predicted_tokens.print();

    // Convert the predicted tokens to text
    std::vector<int> int_ids;
    std::transform(predicted_tokens.data.begin(), predicted_tokens.data.end(), std::back_inserter(int_ids),
                   [](float f) { return static_cast<int>(f); });
    std::vector<std::string> predicted_text = positional_encoder_->idsToText(int_ids);


    // Print the predicted text
    std::cout << "Predicted text: " << std::endl;
    for (const auto& token : predicted_text) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    return Tensor<T>();
}

template <typename T>
std::vector<std::vector<Tensor<T>>> Transformer<T>::convert_to_tensor(const std::vector<std::vector<std::string>>& data) {
    // Prepare special tokens
    const T pad_id = positional_encoder_->textToIds({"<pad>"})[0];
    const T sos_id = positional_encoder_->textToIds({"<sos>"})[0];
    const T eos_id = positional_encoder_->textToIds({"<eos>"})[0];

    // Pre-initialize tgt_initial Tensor
    std::vector<T> tgt_initial_vector(max_len_, pad_id);
    tgt_initial_vector[0] = sos_id;
    const Tensor<T> tgt_initial({max_len_}, tgt_initial_vector);

    // Initialize tensors for source, target, and true labels
    const int num_sentences = static_cast<int>(data.size());
    std::vector<Tensor<T>> src(num_sentences, tgt_initial);
    std::vector<Tensor<T>> tgt(num_sentences, tgt_initial);
    std::vector<Tensor<T>> true_labels(num_sentences);

    for (int i = 0; i < num_sentences; ++i) {
        const std::vector<std::string>& sentence_vector = data[i];
        const int sentence_len = static_cast<int>(sentence_vector.size());
        // Convert sentence to token IDs
        std::vector<int> sentence_ids = positional_encoder_->textToIds(sentence_vector);

        // Prepare src by copying tgt_initial and setting the first token
        src[i].data[1] = sentence_ids[0];

        // Prepare true_labels with eos and padding
        sentence_ids.insert(sentence_ids.begin(), sos_id);
        if (sentence_len >= max_len_) {
            sentence_ids[max_len_ - 1] = eos_id;
        } else {
            sentence_ids[sentence_len + 1] = eos_id;
        }
        sentence_ids.resize(max_len_, pad_id);
        true_labels[i] = Tensor<T>({max_len_}, sentence_ids);
    }

    return {src, tgt, true_labels};
}

// template <typename T>
// std::vector<std::vector<Tensor<T>>> Transformer<T>::convert_to_tensor(const std::vector<std::vector<std::string>>& train_data) {
//     const int num_sentences = static_cast<int>(train_data.size());
//     std::vector<Tensor<T>> src(num_sentences);
//     std::vector<Tensor<T>> tgt(num_sentences);
//     std::vector<Tensor<T>> true_labels(num_sentences);
//
//     // Convert the special tokens to strings if necessary
//     const std::string pad_str = "<pad>";
//     const std::string sos_str = "<sos>";
//     const std::string eos_str = "<eos>";
//
//     // Iterate over the sentences
//     for (int i = 0; i < num_sentences; ++i) {
//         std::vector<std::string> src_vector = train_data[i];
//         std::vector<std::string> tgt_vector = train_data[i];
//
//         // Add special symbols to the source sentence
//         src_vector.insert(src_vector.begin(), sos_str);
//         if (src_vector.size() >= max_len_) {
//             src_vector[max_len_ - 1] = eos_str;
//             src_vector.resize(max_len_);
//         } else {
//             src_vector.push_back(eos_str);
//             src_vector.resize(max_len_, pad_str);  // Pad the remaining tokens
//         }
//
//         // Add special symbols to the target sentence
//         tgt_vector.insert(tgt_vector.begin(), sos_str);
//         if (tgt_vector.size() >= max_len_) {
//             tgt_vector[max_len_ - 2] = eos_str;
//             tgt_vector[max_len_ - 1] = pad_str;
//             tgt_vector.resize(max_len_);
//         } else {
//             tgt_vector.push_back(eos_str);
//             tgt_vector.push_back(pad_str);
//             tgt_vector.resize(max_len_, pad_str);  // Pad the remaining tokens
//         }
//         true_labels[i] = Tensor<T>({max_len_}, positional_encoder_->textToIds(src_vector));
//
//         // Fill the tensors with the special tokens
//         std::fill(src_vector.begin() + 2, src_vector.end(), pad_str); // Fill unpredicted tokens with <pad>
//         std::fill(tgt_vector.begin() + 1, tgt_vector.end(), pad_str); // Fill unpredicted tokens with <pad>
//
//         // Convert the sentences to tokenized tensors
//         src[i] = Tensor<T>({max_len_}, positional_encoder_->textToIds(src_vector));
//         tgt[i] = Tensor<T>({max_len_}, positional_encoder_->textToIds(tgt_vector));
//     }
//
//     return {src, tgt, true_labels};
// }

template <typename T>
std::string Transformer<T>::generate_text(const std::vector<std::string>& input) {
    // Convert the input data to tokenized tensors
    std::vector<std::vector<Tensor<T>>> processed_data = convert_to_tensor({input}, 1);

    // loop through the input data
    for (int i = 0; i < 1; ++i) {
        // Initialize the predicted and true labels tokens
        Tensor<T> masked_src = processed_data[0][i];
        Tensor<T> masked_tgt = processed_data[1][i];
        Tensor<T> true_label_subtensor = processed_data[2][i];

        // Initialize the first token of the target tensor
        for (int j = 1; j < max_len_; j++) {
            // Get the true label for the current token
            T true_label = true_label_subtensor.data[j + 1];

            // Forward pass for the current token
            Tensor<T> output = forward(masked_src, masked_tgt);

            // Compute predicted token
            Tensor<T> predicted = output.argmax();

            // Update src and tgt tensors for the next token
            masked_src.data[j + 1] = predicted.data[0];
            masked_tgt.data[j] = predicted.data[0];

            if (true_label == masked_src.data[j + 1]) break;
        }
    }

    return "";
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
