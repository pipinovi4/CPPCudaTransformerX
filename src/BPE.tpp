#ifndef BPE_TPP
#define BPE_TPP

#include "../include/BPE.h"
#include <iostream>

// Constructor for the BPE tokenizer
BPE::BPE(const std::map<std::pair<std::string, std::string>, int>& merges, const std::unordered_map<std::string, int>& vocab)
    : merges_(merges), vocab_(vocab) {}

std::vector<std::string> BPE::apply(const std::string& word) const {
    std::vector<std::string> subwords;

    // Start by splitting the word into individual characters
    for (char c : word) {
        subwords.emplace_back(1, c);
    }

    // Continuously apply the best merge until no more merges are found
    while (true) {
        std::pair<std::string, std::string> best_pair;
        int best_count = 0;

        // Find the best pair of subwords to merge
        for (size_t i = 0; i < subwords.size() - 1; i++) {
            const auto pair = std::make_pair(subwords[i], subwords[i + 1]);
            if (merges_.find(pair) != merges_.end()) {
                // Update best_pair if the current pair has a higher priority (lower count)
                if (best_count == 0 || merges_.at(pair) < best_count) {
                    best_count = merges_.at(pair);
                    best_pair = pair;
                }
            }
        }

        // If no valid merge is found, break out of the loop
        if (best_count == 0) {
            break;
        }

        // Merge the best pair of subwords
        mergeSubwords(subwords, best_pair);
    }

    return subwords;
}

std::map<std::pair<std::string, std::string>, int> BPE::computeMerges(const std::vector<std::vector<std::string>>& dataset, const int num_merges) {
    std::map<std::pair<std::string, std::string>, int> merges;
    int merge_count = 0;

    // Iterate through the dataset a fixed number of times to create merges
    for (int merge_iteration = 0; merge_iteration < num_merges; ++merge_iteration) {
        for (const auto& sentence : dataset) {
            for (size_t i = 0; i < sentence.size() - 2; ++i) {
                // Only add new pairs to the merges map
                if (merges.find(std::make_pair(sentence[i], sentence[i + 1])) == merges.end()) {
                    merges[std::make_pair(sentence[i], sentence[i + 1])] = merge_count++;

                    // Break early if we've reached the desired number of merges
                    if (merge_count >= num_merges) {
                        return merges;
                    }
                }
            }
        }
    }

    return merges;  // Return the computed merges
}

// Merge subwords in the given vector according to the specified pair
void BPE::mergeSubwords(std::vector<std::string>& subwords, const std::pair<std::string, std::string>& pair) {
    std::vector<std::string> new_subwords;
    bool merged = false;

    // Iterate through the subwords and merge the specified pair
    for (size_t i = 0; i < subwords.size(); i++) {
        if (i < subwords.size() - 1 && subwords[i] == pair.first && subwords[i + 1] == pair.second) {
            // Merge the pair of subwords
            new_subwords.push_back(pair.first + pair.second);
            i++; // Skip the next subword since it was merged
            merged = true;
        } else {
            new_subwords.push_back(subwords[i]);
        }
    }

    // Update the subwords vector if a merge was performed
    if (merged) {
        subwords = new_subwords;
    }
}

#endif // BPE_TPP
