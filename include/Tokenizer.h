#ifndef TOKENIZER_H
#define TOKENIZER_H

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

template <typename T>
class Tokenizer {
public:
    Tokenizer() = default;

    Tokenizer(const std::string& text, const std::string& delimiters);

    std::vector<std::string> tokenize();

private:
    std::string text;
    std::string delimiters;
};

#include "../src/Tokenizer.tpp"

#endif // TOKENIZER_H
