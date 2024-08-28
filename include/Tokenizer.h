#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/// @brief A generic Tokenizer class that provides functionality for text tokenization,
/// conversion between text and token IDs, and vocabulary management.
template <typename T>
class Tokenizer {
public:
    /**
     * @brief Constructor to initialize the tokenizer with specific options.
     * @param max_len Maximum length for tokenized sequences (0 means no limit).
     * @param delimiters A string containing characters that act as token delimiters.
     * @param pad If true, sequences will be padded to max_len.
     * @param truncate If true, sequences will be truncated to max_len.
     * @param to_lower If true, all text will be converted to lowercase.
     * @param strip_punctuation If true, punctuation will be removed from the text.
     */
    explicit Tokenizer(int max_len = 0, std::string delimiters = " \t\n", bool pad = true,
        bool truncate = true, bool to_lower = true, bool strip_punctuation = true);

    /**
     * @brief Tokenizes the input text into a vector of tokens.
     * @param text The input string to be tokenized.
     * @return A vector of tokenized strings.
     */
    std::vector<std::string> tokenize(const std::string& text);

    /**
     * @brief Converts a vector of tokens into a vector of token IDs using the vocabulary.
     * @param tokens The vector of tokens to be converted.
     * @return A vector of integer token IDs.
     */
    std::vector<int> textToIds(const std::vector<std::string>& tokens) const;

    /**
     * @brief Converts a vector of token IDs back into a vector of tokens using the vocabulary.
     * @param ids The vector of token IDs to be converted.
     * @return A vector of tokens corresponding to the provided token IDs.
     */
    std::vector<std::string> idsToText(const std::vector<int>& ids) const;

    /**
     * @brief Builds a vocabulary from a dataset of tokenized sentences.
     * @param vocab A vector of tokens from which to build the vocabulary.
     * @return A map associating each token with a unique integer ID.
     */
    static std::unordered_map<std::string, int> buildVocabulary(const std::vector<std::string>& vocab);

    /**
     * @brief Sets the vocabulary for the tokenizer.
     * @param vocab A map of tokens to integer IDs to be used by the tokenizer.
     */
    void setVocabulary(const std::unordered_map<std::string, int>& vocab);

    /**
     * @brief Saves the current vocabulary to a file.
     * @param filename The path to the file where the vocabulary will be saved.
     */
    void saveVocabulary(const std::string& filename) const;

    /**
     * @brief Loads a vocabulary from a file.
     * @param filename The path to the file from which the vocabulary will be loaded.
     */
    void loadVocabulary(const std::string& filename);

private:
    /**
     * @brief Pads or truncates the token sequence to the specified maximum length.
     * @param tokens The vector of tokens to be padded or truncated.
     * @param max_len The maximum length for the tokenized sequence.
     */
    static void applyPadding(std::vector<std::string>& tokens, int max_len);

    /**
     * @brief Converts the input text to lowercase in a non-in-place operation.
     * This method is a more optimized and memory-efficient alternative to std::tolower.
     * @param text The input string to be converted to lowercase.
     * @return A new string that is the lowercase version of the input.
     */
    static std::string toLower(const std::string& text);

    int max_len;  ///< Maximum length for tokenized sequences
    std::string delimiters;  ///< Characters used to delimit tokens in the text
    bool pad;  ///< Whether to pad sequences to max_len
    bool truncate;  ///< Whether to truncate sequences to max_len
    bool to_lower;  ///< Whether to convert text to lowercase
    bool strip_punctuation;  ///< Whether to remove punctuation from the text
    std::unordered_map<std::string, int> vocab;  ///< Vocabulary mapping tokens to IDs
    static std::unordered_set<std::string> special_tokens;  ///< Set of special tokens like <PAD>, <UNK>
};

#include "../src/Tokenizer.tpp"

#endif // TOKENIZER_H
