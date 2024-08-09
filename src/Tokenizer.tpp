#ifndef TOKENIZER_TPP
#define TOKENIZER_TPP

#include <utility>
#include "../include/Tokenizer.h"

// Static member initialization
template <typename T>
std::unordered_set<std::string> Tokenizer<T>::special_tokens = {"<PAD>", "<UNK>", "<SOS>", "<EOS>"};

// Constructor
template <typename T>
Tokenizer<T>::Tokenizer(std::string text, std::string delimiters, const bool to_lower, const bool strip_punctuation)
    : text(std::move(text)), delimiters(std::move(delimiters)), to_lower(to_lower), strip_punctuation(strip_punctuation) {}

// Tokenize the text
template <typename T>
std::vector<std::string> Tokenizer<T>::tokenize() {
    std::vector<std::string> tokens;
    std::string processed_text = text;

    // Convert to lowercase if the option is enabled
    if (to_lower) {
        std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);
    }

    // Strip punctuation if the option is enabled, but keep special tokens intact
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
            [](const unsigned char c) { return std::ispunct(c) && c != '<' && c != '>'; }),
            processed_text.end()
        );
    }

    // Tokenization logic
    size_t start = processed_text.find_first_not_of(delimiters);
    while (start != std::string::npos) {
        size_t end = processed_text.find_first_of(delimiters, start);
        tokens.emplace_back(processed_text.substr(start, end - start));
        start = processed_text.find_first_not_of(delimiters, end);
    }

    return tokens;
}

// Convert tokens to IDs using vocabulary
template <typename T>
std::vector<int> Tokenizer<T>::textToIds(const std::vector<std::string>& tokens) const {
    std::vector<int> token_ids;
    token_ids.reserve(tokens.size()); // Reserve space in advance

    for (const auto& token : tokens) {
        auto it = vocab.find(token);
        if (it != vocab.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(vocab.at("<UNK>"));
        }
    }
    return token_ids;
}

// Build vocabulary from a dataset
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
template <typename T>
std::unordered_map<int, std::string> Tokenizer<T>::buildInverseVocabulary(const std::unordered_map<std::string, int>& vocab) {
    std::unordered_map<int, std::string> inv_vocab;
    for (const auto& pair : vocab) {
        inv_vocab.emplace(pair.second, pair.first);
    }
    return inv_vocab;
}

#endif // TOKENIZER_TPP
