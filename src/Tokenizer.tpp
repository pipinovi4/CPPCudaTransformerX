#ifndef TOKENIZER_TPP
#define TOKENIZER_TPP

#include "../include/Tokenizer.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <utility>

// Static member initialization
template <typename T>
std::unordered_set<std::string> Tokenizer<T>::special_tokens = {"<PAD>", "<UNK>", "<SOS>", "<EOS>"};

// Constructor
/**
 * @brief Constructs a Tokenizer with specified settings.
 *
 * @param max_len The maximum length of tokenized sequences.
 * @param delimiters Delimiters for splitting the text into tokens.
 * @param pad Whether to pad sequences to max_len.
 * @param truncate Whether to truncate sequences to max_len.
 * @param to_lower Whether to convert text to lowercase.
 * @param strip_punctuation Whether to strip punctuation from the text.
 */
template <typename T>
Tokenizer<T>::Tokenizer(const int max_len, std::string delimiters, const bool pad, const bool truncate, const bool to_lower, const bool strip_punctuation)
    : max_len(max_len), delimiters(std::move(delimiters)), pad(pad), truncate(truncate), to_lower(to_lower), strip_punctuation(strip_punctuation) {
    if (max_len < 0) {
        throw std::invalid_argument("max_len must be non-negative");
    }
}

// Tokenize the text
/**
 * @brief Tokenizes the input text based on the configured settings.
 *
 * @param text The text to tokenize.
 * @return std::vector<std::string> A vector of tokens.
 */
template <typename T>
std::vector<std::string> Tokenizer<T>::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.length() / 2); // Pre-allocate memory

    std::string processed_text = text;

    // Convert to lowercase if enabled
    if (to_lower) {
        std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);
    }

    // Strip punctuation if enabled, while preserving special tokens
    if (strip_punctuation) {
        for (const auto& token : special_tokens) {
            size_t pos = 0;
            while ((pos = processed_text.find(token, pos)) != std::string::npos) {
                processed_text.replace(pos, token.length(), " " + token + " ");
                pos += token.length() + 2;
            }
        }
        processed_text.erase(
            std::remove_if(processed_text.begin(), processed_text.end(),
            [](unsigned char c) { return std::ispunct(c) && c != '<' && c != '>'; }),
            processed_text.end()
        );
    }

    // Split the text into tokens based on delimiters
    size_t start = processed_text.find_first_not_of(delimiters);
    while (start != std::string::npos) {
        const size_t end = processed_text.find_first_of(delimiters, start);
        tokens.emplace_back(processed_text.substr(start, end - start));
        start = processed_text.find_first_not_of(delimiters, end);
    }

    // Apply padding and truncation if necessary
    applyPadding(tokens, max_len);

    return tokens;
}

// Convert tokens to IDs using vocabulary
/**
 * @brief Converts a vector of tokens into their corresponding IDs using the vocabulary.
 *
 * @param tokens The tokens to convert.
 * @return std::vector<int> A vector of token IDs.
 */
template <typename T>
std::vector<int> Tokenizer<T>::textToIds(const std::vector<std::string>& tokens) const {
    std::vector<int> token_ids;
    token_ids.reserve(tokens.size());

    for (const auto& token : tokens) {
        if (auto it = vocab.find(token); it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(vocab.at("<UNK>"));
        }
    }

    // Apply truncation and padding if necessary
    if (truncate && max_len > 0 && token_ids.size() > max_len) {
        token_ids.resize(max_len);
    }
    if (pad && max_len > 0 && token_ids.size() < max_len) {
        token_ids.insert(token_ids.end(), max_len - token_ids.size(), vocab.at("<PAD>"));
    }

    return token_ids;
}

// Build vocabulary from a dataset
/**
 * @brief Builds a vocabulary from a dataset of tokenized sentences.
 *
 * @param dataset A vector of tokenized sentences.
 * @return std::unordered_map<std::string, int> A vocabulary map with tokens as keys and IDs as values.
 */
template <typename T>
std::unordered_map<std::string, int> Tokenizer<T>::buildVocabulary(const std::vector<std::vector<std::string>>& dataset) {
    std::unordered_map<std::string, int> vocab;
    int index = 0;

    // Add special tokens first
    for (const auto& token : special_tokens) {
        vocab.emplace(token, index++);
    }

    // Add tokens from the dataset
    for (const auto& sentence : dataset) {
        for (const auto& token : sentence) {
            if (vocab.find(token) == vocab.end()) {
                vocab.emplace(token, index++);
            }
        }
    }

    return vocab;
}

// Build inverse vocabulary (ID to token mapping)
/**
 * @brief Builds an inverse vocabulary mapping from ID to token.
 *
 * @param vocab The vocabulary map to invert.
 * @return std::unordered_map<int, std::string> The inverse vocabulary map.
 */
template <typename T>
std::unordered_map<int, std::string> Tokenizer<T>::buildInverseVocabulary(const std::unordered_map<std::string, int>& vocab) {
    std::unordered_map<int, std::string> inv_vocab;
    for (const auto& [token, id] : vocab) {
        inv_vocab.emplace(id, token);
    }
    return inv_vocab;
}

// Set vocabulary
/**
 * @brief Sets the vocabulary for the tokenizer.
 *
 * @param vocab The vocabulary map to set.
 */
template <typename T>
void Tokenizer<T>::setVocabulary(const std::unordered_map<std::string, int>& vocab) {
    this->vocab = vocab;
}

// Save vocabulary to a file
/**
 * @brief Saves the vocabulary to a file.
 *
 * @param filename The file to save the vocabulary to.
 */
template <typename T>
void Tokenizer<T>::saveVocabulary(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to open file for saving vocabulary");
    }
    for (const auto& [token, id] : vocab) {
        out << token << " " << id << "\n";
    }
    out.close();
}

// Load vocabulary from a file
/**
 * @brief Loads the vocabulary from a file.
 *
 * @param filename The file to load the vocabulary from.
 */
template <typename T>
void Tokenizer<T>::loadVocabulary(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open file for loading vocabulary");
    }
    std::string token;
    int id;
    while (in >> token >> id) {
        vocab[token] = id;
    }
    in.close();
}

// Apply padding and truncation
/**
 * @brief Pads or truncates the token sequence to the specified maximum length.
 *
 * @param tokens The tokens to pad or truncate.
 * @param max_len The maximum length to pad or truncate to.
 */
template <typename T>
void Tokenizer<T>::applyPadding(std::vector<std::string>& tokens, const int max_len) {
    if (max_len > 0) {
        if (tokens.size() > max_len) {
            tokens.resize(max_len);  // Truncate if too many tokens
        } else if (tokens.size() < max_len) {
            tokens.insert(tokens.end(), max_len - tokens.size(), "<PAD>");  // Pad if too few tokens
        }
    }
}

#endif // TOKENIZER_TPP
