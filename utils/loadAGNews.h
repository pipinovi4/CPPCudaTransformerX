#ifndef LOADAGNEWS_H
#define LOADAGNEWS_H

#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

inline std::vector<std::vector<std::vector<std::string>>> loadAGNews(const std::string& dataset_dir);

#include "loadAGNews.tpp"

#endif //LOADAGNEWS_H
