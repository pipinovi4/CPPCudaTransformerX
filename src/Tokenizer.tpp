#ifndef TOKENIZER_TPP
#define TOKENIZER_TPP

#include "../include/Tokenizer.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <utility>

template <typename T>
std::unordered_set<std::string> Tokenizer<T>::special_tokens = {"<pad>", "<unk>", "<sos>", "<eos>"};

template <typename T>
Tokenizer<T>::Tokenizer(const int max_len, std::string delimiters, const bool pad, const bool truncate, const bool to_lower, const bool strip_punctuation)
    : max_len(max_len), delimiters(std::move(delimiters)), pad(pad), truncate(truncate), to_lower(to_lower), strip_punctuation(strip_punctuation) {
    if (max_len < 0) {
        throw std::invalid_argument("max_len must be non-negative");
    }
}

template <typename T>
std::vector<std::string> Tokenizer<T>::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    tokens.reserve(text.length() / 2); // Pre-allocate memory

    std::string processed_text = text;

    // Convert to lowercase if enabled
    std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);

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

template <typename T>
std::vector<int> Tokenizer<T>::textToIds(const std::vector<std::string>& tokens) const {
    std::vector<int> token_ids;
    token_ids.reserve(tokens.size());

    const int unk_id = vocab.at("<unk>");
    const int pad_id = vocab.at("<pad>");

    for (const auto& token : tokens) {
        // Directly transform the token to lowercase while searching
        std::string lower_token(token.size(), '\0');
        std::transform(token.begin(), token.end(), lower_token.begin(), ::tolower);

        // Use find() and insert the result directly
        auto it = vocab.find(lower_token);
        token_ids.push_back(it != vocab.end() ? it->second : unk_id);
    }

    // Apply truncation if necessary
    if (truncate && max_len > 0 && token_ids.size() > max_len) {
        token_ids.resize(max_len);
    }

    // Apply padding if necessary
    if (pad && max_len > 0 && token_ids.size() < max_len) {
        token_ids.insert(token_ids.end(), max_len - token_ids.size(), pad_id);
    }

    return token_ids;
}


// template <typename T>
// std::vector<int> Tokenizer<T>::textToIds(const std::vector<std::string>& tokens) const {
//     std::vector<int> token_ids;
//     token_ids.reserve(tokens.size());
//
//     for (const auto& token : tokens) {
//         std::string lower_token = token;
//         std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
//
//         auto it = vocab.find(lower_token);
//         if (it != vocab.end()) {
//             token_ids.push_back(it->second);
//         } else {
//             token_ids.push_back(vocab.at("<unk>"));
//         }
//     }
//
//     // Apply truncation and padding if necessary
//     if (truncate && max_len > 0 && token_ids.size() > max_len) {
//         token_ids.resize(max_len);
//     }
//     if (pad && max_len > 0 && token_ids.size() < max_len) {
//         token_ids.insert(token_ids.end(), max_len - token_ids.size(), vocab.at("<pad>"));
//     }
//
//     return token_ids;
// }

template <typename T>
std::vector<std::string> Tokenizer<T>::idsToText(const std::vector<int>& ids) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());

    for (const auto& id : ids) {
        // Find the token corresponding to the given ID
        for (const auto& pair : vocab) {
            if (pair.second == id) {
                tokens.push_back(pair.first);
                break;
            }
        }
    }

    return tokens;
}

template <typename T>
std::unordered_map<std::string, int> Tokenizer<T>::buildVocabulary(const std::vector<std::string>& vocab) {
    std::unordered_map<std::string, int> vocab_;
    int index = 0;

    // Add special tokens first
    for (const auto& token : special_tokens) {
        // Convert to lowercase and add to the vocab
        vocab_.emplace(toLower(token), index++);
    }

    // Add tokens from the vocab
    for (const auto& token : vocab) {
        // Convert to lowercase and add to the vocab
        vocab_.emplace(toLower(token), index++);
    }

    return vocab_;
}

template <typename T>
void Tokenizer<T>::setVocabulary(const std::unordered_map<std::string, int>& vocab) {
    this->vocab = vocab;
}

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

template <typename T>
std::string Tokenizer<T>::toLower(const std::string& text) {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    return lower_text;
}

#endif // TOKENIZER_TPP