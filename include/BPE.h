#ifndef BPE_H
#define BPE_H

#pragma once
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

/**
 * @class BPE
 * @brief A class for Byte-Pair Encoding (BPE) tokenization.
 *
 * The BPE class provides a method to apply byte-pair encoding to a given word,
 * allowing for subword tokenization based on a set of predefined merge rules.
 * This is commonly used in natural language processing to reduce vocabulary size
 * while maintaining the ability to represent rare words effectively.
 */
class BPE {
public:
    /**
     * @brief Constructs a BPE tokenizer.
     *
     * @param merges A map of merge rules where each pair of strings represents
     *               a possible merge and the associated integer is the priority.
     * @param vocab  A vocabulary mapping where each string token is associated with an integer ID.
     */
    inline BPE(const std::map<std::pair<std::string, std::string>, int>& merges, const std::unordered_map<std::string, int>& vocab);

    /**
     * @brief Applies the BPE algorithm to the given word.
     *
     * The method iteratively merges pairs of subwords in the given word according to the
     * BPE merge rules until no more merges can be applied.
     *
     * @param word The input word to which BPE is applied.
     * @return A vector of strings representing the subwords after applying BPE.
     */
    inline std::vector<std::string> apply(const std::string& word) const;

private:
    std::map<std::pair<std::string, std::string>, int> merges_; ///< Stores the BPE merge rules.
    std::unordered_map<std::string, int> vocab_; ///< Stores the vocabulary mapping.

    /**
     * @brief Merges subwords in the given vector according to the specified pair.
     *
     * This static function takes a vector of subwords and a specific pair of subwords to be merged,
     * and then it combines the pair into a single subword within the vector.
     *
     * @param subwords The vector of subwords that will be merged.
     * @param pair The pair of subwords to merge into a single subword.
     */
    static inline void mergeSubwords(std::vector<std::string>& subwords, const std::pair<std::string, std::string>& pair);
};

#include "../src/BPE.tpp"

#endif // BPE_H
