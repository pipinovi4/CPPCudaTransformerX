#ifndef LOADWIKITEXT_H
#define LOADWIKITEXT_H

#pragma once
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <tuple>

inline std::vector<std::vector<std::vector<std::string>>> loadWikiText(const std::string& dataset_dir);

#include "loadWikiText.tpp"

#endif // LOADWIKITEXT_H
