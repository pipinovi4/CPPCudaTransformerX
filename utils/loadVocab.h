#ifndef LOADVOCAB_H
#define LOADVOCAB_H

#pragma once
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Loads a vocabulary from a file where each line represents a word.
 *
 * This function reads the file specified by the given filepath and loads each word 
 * (each line in the file) into a vector of strings. The file is expected to have 
 * one word per line. Empty lines are ignored.
 *
 * @param filepath The path to the file containing the vocabulary.
 * @return A vector of strings, each representing a word from the vocabulary file.
 * @throws std::runtime_error If the file cannot be opened.
 */
inline std::vector<std::string> load_vocab(const std::string& filepath);

#include "loadVocab.tpp"

#endif // LOADVOCAB_H
