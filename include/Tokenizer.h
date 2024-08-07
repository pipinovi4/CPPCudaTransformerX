#ifndef TOKENIZER_H
#define TOKENIZER_H

#pragma once
#include "../include/Tokenizer.h"
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
class Tokenizer {
public:
    explicit Tokenizer(std::string  text, std::string  delimiters = " \t\n", bool to_lower = true, bool strip_punctuation = true);

    std::vector<std::string> tokenize();
    std::vector<int> textToIds(const std::vector<std::string>& tokens) const;

    static std::unordered_map<std::string, int> buildVocabulary(const std::vector<std::vector<std::string>>& dataset);
    static std::unordered_map<int, std::string> buildInverseVocabulary(const std::unordered_map<std::string, int>& vocab);

    void setVocabulary(const std::unordered_map<std::string, int>& vocab) {
        this->vocab = vocab;
    }

    static std::unordered_set<std::string> special_tokens;

private:
    std::string text;
    std::string delimiters;
    bool to_lower;
    bool strip_punctuation;

    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> inv_vocab;
};

#include "../src/Tokenizer.tpp"

#endif // TOKENIZER_H
