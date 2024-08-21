#ifndef LOADMNIST_H
#define LOADMNIST_H

#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include "readUint32.tpp"

inline std::vector<std::vector<std::uint8_t>> loadMNISTImages(const std::string& filename);

inline std::vector<std::uint8_t> loadMNISTLabels(const std::string& filename);

#include "loadMNIST.tpp"

#endif //LOADMNIST_H

