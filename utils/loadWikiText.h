#ifndef LOADWIKITEXT_H
#define LOADWIKITEXT_H

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

inline std::vector<std::vector<std::vector<std::string>>> loadWikiText(const std::string& dataset_dir);

#include "loadWikiText.tpp"

#endif //LOADWIKITEXT_H
