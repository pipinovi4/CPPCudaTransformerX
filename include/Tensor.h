#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include "MixedPrecisionFloat16.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

template <typename T>
class Tensor {
private:
    std::vector<int> dimensions;
    std::vector<T> data;
    std::string dtype;

    [[nodiscard]] int calculateIndex(const std::vector<int>& indices) const;
    T &at(const std::vector<int> &indices);

public:
    explicit Tensor(const std::vector<int>& dims);

    [[maybe_unused]] [[nodiscard]] const std::vector<int>& shape() const;

    [[maybe_unused]] [[nodiscard]] int size() const;

    [[maybe_unused]] void set(const std::vector<int>& indices, T value);

    [[maybe_unused]] T get(const std::vector<int>& indices) const;

    void fill(T value);

    [[maybe_unused]] [[nodiscard]] Tensor<T> slice(int axis, int start, int end, int step) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> slice(int axis, int start, int end) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> slice(int axis, int start) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> slice(int axis) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> concatenate(const Tensor<T>& other) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> concatenate(const Tensor<T>& other, int axis) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> expandDims(int axis) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> squeeze() const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> reshape(int newShape) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> reshape(const std::vector<int>& newDimensions) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> transpose(const std::vector<int>& permutation = std::vector<int>()) const;

    [[maybe_unused]] [[nodiscard]] static Tensor<T> zeros(const std::vector<int>& dims);

    [[maybe_unused]] [[nodiscard]] static Tensor<T> ones(const std::vector<int>& dims);

    [[maybe_unused]] [[nodiscard]] Tensor<T> strip(const int& axis) const;

    Tensor<T> operator+(const Tensor<T>& other) const;

    Tensor<T> operator-(const Tensor<T>& other) const;

    Tensor<T> operator*(const Tensor<T>& other) const;

    Tensor<T> operator/(const Tensor<T>& other) const;

    Tensor<T> operator+(T scalar) const;

    Tensor<T> operator-(T scalar) const;

    Tensor<T> operator*(T scalar) const;

    Tensor<T> operator/(T scalar) const;

    Tensor<T> operator[](int index) const;

    Tensor<T> operator[](const std::vector<int>& indices) const;

    void print() const;
};

#include "../src/Tensor.cpp"

#endif // TENSOR_H
