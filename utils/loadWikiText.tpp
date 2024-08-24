#ifndef LOADWIKITEXT_TPP
#define LOADWIKITEXT_TPP

#include "loadWikiText.h"

// Function to remove empty sentences from a dataset
inline void removeEmptySentences(std::vector<std::vector<std::string>>& dataset) {
    dataset.erase(
        std::remove_if(dataset.begin(), dataset.end(),
            [](const std::vector<std::string>& sentence) {
                return sentence.empty();  // Check if the sentence is empty
            }),
        dataset.end()
    );
}

std::vector<std::vector<std::vector<std::string>>> loadWikiText(const std::string& dataset_dir) {
    std::vector<std::vector<std::string>> train_data;
    std::vector<std::vector<std::string>> val_data;
    std::vector<std::vector<std::string>> test_data;

    // Load the training data
    std::ifstream train_file(dataset_dir + "/train.txt");
    if (!train_file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + dataset_dir + "/train.txt");
    }

    std::string line;
    while (std::getline(train_file, line)) {
        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;

        while (iss >> word) {
            words.push_back(word);
        }

        train_data.push_back(words);
    }

    // Load the validation data
    std::ifstream val_file(dataset_dir + "/valid.txt");
    if (!val_file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + dataset_dir + "/valid.txt");
    }

    while (std::getline(val_file, line)) {
        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;

        while (iss >> word) {
            words.push_back(word);
        }

        val_data.push_back(words);
    }

    // Load the test data
    std::ifstream test_file(dataset_dir + "/test.txt");
    if (!test_file.is_open()) {
        throw std::runtime_error("Failed to open the file: " + dataset_dir + "/test.txt");
    }

    while (std::getline(test_file, line)) {
        std::vector<std::string> words;
        std::istringstream iss(line);
        std::string word;

        while (iss >> word) {
            words.push_back(word);
        }

        test_data.push_back(words);
    }

    removeEmptySentences(train_data);
    removeEmptySentences(val_data);
    removeEmptySentences(test_data);

    return {train_data, val_data, test_data};
}

#endif // LOADWIKITEXT_TPP
