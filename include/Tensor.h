#ifndef TENSOR_H
#define TENSOR_H

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <numeric>
#include <cassert>
#include <cmath>
#include <functional>
#include "MixedPrecisionFloat16.h"
#include <random>

template <typename T>
class Tensor {
public:
    std::vector<T> data;
    std::vector<int> dimensions;

    Tensor() = default;

    // Constructor to initialize the tensor with dimensions and optionally with data
    template<typename D>
    Tensor(const std::vector<int>& dims, const D& data);

    // Constructor to initialize the tensor with data
    template<typename D>
    explicit Tensor(const D& data);

    // Constructor to initialize the tensor with dimensions only
    explicit Tensor(const std::vector<int>& dims);

    // Constructor to initialize the tensor with dimensions from an initializer list
    Tensor(std::initializer_list<int> dims);

    // Default constructor to initialize an empty tensor
    Tensor(const std::vector<int>& dims, const int& newSize);


    // Helper functions to get the dimensions and size of the tensor

    [[nodiscard]] const std::vector<int>& shape() const;

    [[nodiscard]] int size() const;

    void print() const;

    // Manipulation structure functions

    void set(const std::vector<int>& indices, T value);

    T get(const std::vector<int>& indices) const;

    void fill(T value);

    Tensor<T> sqrt();

    Tensor<T> apply(std::function<T(T)> func) const;

    void add(const Tensor<T>& other);

    Tensor<T> sum(int axis) const;

    Tensor<T> slice(int axis, int start, int end, int step) const;

    Tensor<T> slice(int axis, int start, int end) const;

    Tensor<T> slice(int axis, int start) const;

    Tensor<T> slice(int axis) const;

    Tensor<T> concatenate(const Tensor<T>& other) const;

    Tensor<T> concatenate(const Tensor<T>& other, int axis) const;

    Tensor<T> expandDims(int axis) const;

    Tensor<T> squeeze() const;

    Tensor<T> reshape(int newShape) const;

    Tensor<T> reshape(const std::vector<int>& newDimensions) const;

    Tensor<T> transpose(const std::vector<int>& permutation = std::vector<int>()) const;

    static Tensor<T> zeros(const std::vector<int>& dims);

    static Tensor<T> ones(const std::vector<int>& dims);

    static Tensor<T> uniform(const std::vector<int>& dims, T lower = 0.0, T upper = 1.0);

    Tensor<T> tril(const int& axis = 0);

    Tensor<T> triu(const int& axis = 0);

    Tensor<T> dot(const Tensor<T>& other) const ;

    Tensor<T> operator+(const Tensor<T>& other) const;

    Tensor<T> operator-(const Tensor<T>& other) const;

    Tensor<T> operator*(const Tensor<T>& other) const;

    Tensor<T> operator/(const Tensor<T>& other) const;

    Tensor<T> operator+(T scalar) const;

    Tensor<T> operator-(T scalar) const;

    Tensor<T> operator*(T scalar) const;

    Tensor<T> operator/(T scalar) const;

    Tensor<T>& operator-=(const Tensor<T>& other);

    Tensor<T>& operator+=(const Tensor<T>& other);

    Tensor<T>& operator*=(const Tensor<T>& other);

    Tensor<T>& operator/=(const Tensor<T>& other);

    Tensor<T>& operator-=(const T& scalar);

    Tensor<T>& operator+=(const T& scalar);

    Tensor<T>& operator*=(const T& scalar);

    Tensor<T>& operator/=(const T& scalar);

    Tensor<T> operator[](const std::vector<int>& indices);

    Tensor<T> operator[](const std::vector<int>& indices) const;

    T& operator()(int indices);

    T& operator()(const std::vector<int>& indices);

    bool operator==(const Tensor<T>& other) const;

    bool operator!=(const Tensor<T>& other) const;

    // Method to serialize the tensor to a stream
    void serialize(std::ostream& os) const;

    // Method to deserialize the tensor from a stream
    void deserialize(std::istream& is);

private:
    std::vector<int> strides;

    [[nodiscard]] int calculateIndex(const std::vector<int>& indices) const {
        if (indices.size() != dimensions.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions");
        }

        int index = 0;
        int stride = 1;
        for (int i = dimensions.size() - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= dimensions[i];
        }
        return index;
    }

    [[nodiscard]] std::vector<int> calculateStrides() const {
        std::vector<int> localStrides(dimensions.size());
        int stride = 1;
        for (int i = dimensions.size() - 1; i >= 0; --i) {
            localStrides[i] = stride;
            stride *= dimensions[i];
        }
        return localStrides;
    }

    static int getTotalSize(const std::vector<int>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
    }

    // Primary template for is_vector
    template <typename D>
    struct is_vector : std::false_type {};

    // Specialization for std::vector
    template <typename D, typename Allocator>
    struct is_vector<std::vector<D, Allocator>> : std::true_type {};

    // Base case: handle the innermost type
    template <typename D>
    struct ExtractType {
        using Type = T; // This is the innermost type
    };

    // Recursive case: handle the outer dimensions
    template <typename D>
    struct ExtractType<std::vector<D>> {
        using Type = typename ExtractType<D>::Type; // Recursively extract inner type
    };

    // Recursive traversal to the innermost type
    template <typename D>
    void flatten(const D& vec, std::vector<T>& result) {
        if constexpr (is_vector<D>::value) {
            for (const auto& elem : vec) {
                flatten(elem, result);
            }
        } else {
            result.push_back(vec);
        }
    }

    template <typename D>
     std::vector<int> compute_shape(const D& vec) {
        if constexpr (is_vector<D>::value) {
            if (vec.empty()) {
                return {0};
            }
            std::vector<int> shape;
            shape.push_back(vec.size());
            auto inner_shape = compute_shape(vec[0]);
            shape.insert(shape.end(), inner_shape.begin(), inner_shape.end());
            return shape;
        } else {
            return {};
        }
    }

    static std::vector<int> combineIndices(const std::vector<int>& this_indices, const std::vector<int>& other_indices, const int this_rank, const int other_rank) {
        std::vector<int> result_indices(this_rank + (other_rank - 1), 0);

        // Copy dimensions from this_indices
        for (int i = 0; i < this_rank - 1; ++i) {
            result_indices[i] = this_indices[i];
        }

        // Insert dimensions from other_indices
        for (int i = 0; i < other_rank - 1; ++i) {
            result_indices[this_rank - 1 + i] = other_indices[i + 1];
        }

        return result_indices;
    }

    [[nodiscard]] int toFlatIndex(const std::vector<int>& indices) const {
        size_t flatIndex = 0;
        size_t product = 1;
        for (size_t i = indices.size(); i > 0; --i) {
            const auto index = static_cast<size_t>(indices[i - 1]);
            flatIndex += index * product;
            product *= dimensions[i - 1];
        }
        return static_cast<int>(flatIndex);
    }
};

#include "../src/Tensor.tpp"

template class Tensor<float>;
template class Tensor<int>;
template class Tensor<double>;

#endif // TENSOR_H
