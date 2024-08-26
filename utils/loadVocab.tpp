#ifndef LOADVOCAB_TPP
#define LOADVOCAB_TPP

#include "loadVocab.h"

std::vector<std::string> load_vocab(const std::string& filepath) {
    std::vector<std::string> vocab;
    std::ifstream file(filepath);

    if (file.is_open()) {
        std::string word;
        while (std::getline(file, word)) {
            // Only add non-empty words
            if (!word.empty()) {
                vocab.push_back(word);
            }
        }
        file.close();
    } else {
        throw std::runtime_error("Could not open the file: " + filepath);
    }

    return vocab;
}

#endif //LOADVOCAB_TPP
