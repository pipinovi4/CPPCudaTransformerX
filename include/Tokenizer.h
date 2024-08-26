#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T>
class Tokenizer {
public:
    // Constructor: Initializes the tokenizer with various options
    explicit Tokenizer(int max_len = 0, std::string delimiters = " \t\n", bool pad = true,
        bool truncate = true, bool to_lower = true, bool strip_punctuation = true);

    // Tokenizes the input text into a vector of tokens
    std::vector<std::string> tokenize(const std::string& text);

    // Converts a vector of tokens into a vector of token IDs using the vocabulary
    std::vector<int> textToIds(const std::vector<std::string>& tokens) const;

    // Converts a vector of token IDs into a vector of tokens using the vocabulary
    std::vector<std::string> idsToText(const std::vector<T>& ids) const;

    // Builds a vocabulary from a dataset of tokenized sentences
    static std::unordered_map<std::string, int> buildVocabulary(const std::vector<std::string>& vocab);

    // Builds an inverse vocabulary (ID to token mapping) from a given vocabulary
    static std::unordered_map<int, std::string> buildInverseVocabulary(const std::unordered_map<std::string, int>& vocab);

    // Sets the vocabulary for the tokenizer
    void setVocabulary(const std::unordered_map<std::string, int>& vocab);

    // Saves the vocabulary to a file
    void saveVocabulary(const std::string& filename) const;

    // Loads the vocabulary from a file
    void loadVocabulary(const std::string& filename);

private:
    // Pads or truncates the token sequence to the specified maximum length
    static void applyPadding(std::vector<std::string>& tokens, int max_len);

    int max_len;  // Maximum length for tokenized sequences
    std::string delimiters;  // Characters used to delimit tokens in the text
    bool pad;  // Whether to pad sequences to max_len
    bool truncate;  // Whether to truncate sequences to max_len
    bool to_lower;  // Whether to convert text to lowercase
    bool strip_punctuation;  // Whether to remove punctuation from the text
    std::unordered_map<std::string, int> vocab;  // Vocabulary mapping tokens to IDs
    static std::unordered_set<std::string> special_tokens;  // Set of special tokens like <PAD>, <UNK>
};

#include "../src/Tokenizer.tpp"

#endif // TOKENIZER_H