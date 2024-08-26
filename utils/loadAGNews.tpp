#ifndef LOADAGNEWS_TPP
#define LOADAGNEWS_TPP

#include "loadAGNews.h"

std::vector<std::vector<std::vector<std::string>>> loadAGNews(const std::string& dataset_dir) {
    std::vector<std::vector<std::string>> train_data;
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

    // Load the test data
    std::ifstream test_file(dataset_dir + "/test.txt");
    if (!test_file.is_open()) {
        throw std::runtime_error("Failed to open the file: data/ag_news/test.txt");
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

    // Return both train and test datasets
    return {train_data, test_data};
}

#endif //LOADAGNEWS_TPP
