#ifndef TRANSFORMER_TPP
#define TRANSFORMER_TPP

#include "Transformer.h"

template <typename T>
Transformer<T>::Transformer(LossFunction<T>* loss_function, Optimizer<T>* optimizer, std::vector<std::string> vocab,
        typename Optimizer<T>::LearningRateSchedule& learning_rate_schedule, int vocab_size, int d_model,
        int n_heads, int d_ff, int max_len, const float dropout, const float label_smoothing)
    : vocab_size_(vocab_size), d_model_(d_model), n_heads_(n_heads), d_ff_(d_ff),
      max_len_(max_len), dropout_(dropout), label_smoothing_(label_smoothing),
      learning_rate_schedule_(learning_rate_schedule), loss_function_(loss_function),
      optimizer_(optimizer), embedding_(), positional_encoder_(), encoder_layers_(),
      decoder_layers_(), output_encoder_layers_(), output_layer_softmax_() {
    // Initialize the ReLU activation function
    // Initialize the embedding layers

    this->embedding_ = std::make_unique<Embedding<T>>(vocab_size, d_model, learning_rate_schedule);

    // Initialize the positional encoder
    positional_encoder_ = std::make_unique<Tokenizer<T>>(max_len);

    // build vocabulary
    const std::unordered_map<std::string, int> vocab_map = Tokenizer<T>::buildVocabulary(vocab);
    positional_encoder_->setVocabulary(vocab_map);

    // Initialize the encoder layers
    std::cout << d_model / n_heads << std::endl;
    this->encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
    d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, new typename ActivationFunction<T>::Softmax))));
    this->encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new PositionalWiseDenseLayer<T>(d_model, d_ff, new typename ActivationFunction<T>::ReLU, 0.1))));

    // Initialize the decoder layers
    this->decoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, new typename ActivationFunction<T>::Softmax))));
    this->decoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new PositionalWiseDenseLayer<T>(d_model, d_ff, new typename ActivationFunction<T>::ReLU, 0.1))));

    // Initialize output encoder layers for shifted target (MASKED MULTIHEAD ATTENTION)
    this->output_encoder_layers_.emplace_back(std::move(std::make_unique<ResidualBlock<T, Layer<T>*>>(
        d_model, 1e-6, new MultiHeadAttention<T>(d_model, n_heads, d_model / n_heads, new typename ActivationFunction<T>::Softmax))));

    // Initialize the final dense layers
    this->output_layer_softmax_ = std::make_unique<DenseLayer<T>>(d_model, vocab_size_, new typename ActivationFunction<T>::Softmax, 0.1);

    // Initialize optimizer parameters
    optimizer_->initialize_params(parameters());
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
    // Update the model parameters using the optimizer
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

    // Step 3: Training loop with batch processing
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        for (int batch_start = 0; batch_start < num_sentences; batch_start += batch_size) {
            T loss = 0;
            for (int i = 0; i < batch_size; ++i) {
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

                    // Sample or predict the token (can use sampling techniques if needed)
                    Tensor<T> predicted = output.argmax(0);

                    // Update src and tgt tensors for the next token
                    masked_src.data[j + 1] = true_label;
                    masked_tgt.data[j] = true_label;

                    // Label smoothing: Apply smoothing to the true labels before computing the loss
                    float smoothing_value = label_smoothing_ / (vocab_size_ - 1);  // Adjust for (V - 1) non-true labels
                    true_labels_tokens.fill(smoothing_value);
                    true_labels_tokens.data[true_label] = 1 - label_smoothing_;

                    // Compute the loss with the smoothed true labels
                    loss += loss_function_->forward(output, true_labels_tokens);

                    // Compute gradients of the model parameters using the smoothed true labels
                    Tensor<T> grad_output = loss_function_->backward(output, true_labels_tokens);

                    backward(grad_output);

                    // Stop if the pad token is encountered
                    if (true_label == pad_token) break;
                }
            }
            // Update model parameters with the computed gradients
            update(epoch);

            // Zero the gradients for the next batch
            zero_grad();

            // Print the batch loss
            std::cout << "Epoch: " << epoch + 1 << " | Batch: " << (batch_start / batch_size) + 1
                      << " | Batch Loss: " << loss << std::endl;

            loss = 0;
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
    std::vector<std::string> predicted_text = positional_encoder_->idsToText(convert_to_int_tokens(predicted_tokens));

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
    std::vector<Tensor<T>> mama(num_sentences);

    for (int i = 0; i < num_sentences; ++i) {
        const std::vector<std::string>& sentence_vector = data[i];
        const int sentence_len = static_cast<int>(sentence_vector.size());
        // Convert sentence to token IDs
        std::vector<int> sentence_ids = positional_encoder_->textToIds(sentence_vector);

        mama[i] = Tensor<T>({max_len_}, sentence_ids);

        // Prepare src by copying tgt_initial and setting the first token
        src[i].data[1] = sentence_ids[0];

        // Prepare true_labels with eos and padding
        sentence_ids[0] = sos_id;
        if (sentence_len >= max_len_) {
            sentence_ids[max_len_ - 1] = eos_id;
        } else {
            sentence_ids[sentence_len + 1] = eos_id;
        }
        true_labels[i] = Tensor<T>({max_len_}, sentence_ids);
    }

    return {src, tgt, true_labels, mama};
}

template <typename T>
std::vector<std::vector<std::string>> Transformer<T>::generate(const std::vector<std::vector<std::string>>& input) {
    // Initialize the EOS token ID constant
    const int EOS_TOKEN_ID = positional_encoder_->textToIds({"<eos>"})[0];

    // Convert the input data to tokenized tensors
    std::vector<std::vector<Tensor<T>>> processed_data = convert_to_tensor(input);

    // Print size of processed data by zero dimension
    std::cout << "Processed data size by zero dimension: " << processed_data.size() << std::endl;

    // Print size of processed data by first dimenion
    std::cout << "Processed data size by first dimension: " << processed_data[0].size() << std::endl;

    // Initialize the generated sentences
    std::vector<std::vector<std::string>> generated_sentences;

    // Update processed data over that to add 3 additional context tokens
    for (int i = 0; i < processed_data[0].size(); ++i) {
        for (int j = 1; j < 3; ++j) {
            processed_data[0][i].data[j + 1] = processed_data[2][i].data[j]; // src
            processed_data[1][i].data[j] = processed_data[2][i].data[j]; // tgt
        }
    }

    // Loop through the processed data
    for (int i = 0; i < processed_data[0].size(); ++i) {
        // Initialize the source, target, and true label tensors
        Tensor<T> masked_src = processed_data[0][i];
        Tensor<T> masked_tgt = processed_data[1][i];
        Tensor<T> true_label_subtensor = processed_data[2][i];

        // Initialize the first token of the target tensor
        for (int j = 3; j < max_len_; j++) {
            // Forward pass for the current token
            Tensor<T> output = forward(masked_src, masked_tgt);

            // Compute predicted token
            Tensor<T> predicted = output.argmax(0);

            // Update src and tgt tensors for the next token
            masked_src.data[j + 1] = predicted.data[0];
            masked_tgt.data[j] = predicted.data[0];

            // Break if the EOS token is reached
            if (predicted.data[0] == EOS_TOKEN_ID) break;
        }
        // Convert the predicted tokens to text and add to the generated sentences
        generated_sentences.push_back(positional_encoder_->idsToText(convert_to_int_tokens(masked_tgt)));
    }
    return generated_sentences;
}

template <typename T>
void Transformer<T>::zero_grad() {
    for (auto& grad : this->gradients()) {
        grad.get().fill(0);
    }
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

// Convert float tokens to int tokens
template <typename T>
std::vector<int> Transformer<T>::convert_to_int_tokens(const Tensor<T>& tokens) {
    std::vector<int> int_ids;
    std::transform(tokens.data.begin(), tokens.data.end(), std::back_inserter(int_ids),
                   [](const float f) { return static_cast<int>(f); });
    return int_ids;
}

#endif //TRANSFORMER_TPP
