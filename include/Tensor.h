#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <type_traits>
#include <cassert>
#include <cmath>
#include "MixedPrecisionFloat16.h"

template <typename T>
class Tensor {
public:
    // Constructor to initialize the tensor with dimensions and optionally with data
    template<typename D>
    explicit Tensor(const std::vector<int>& dims, const D& data);

    // Constructor to initialize the tensor with dimensions only
    explicit Tensor(const std::vector<int>& dims);

    // Constructor to initialize the tensor with dimensions from an initializer list
    Tensor(std::initializer_list<int> dims);

    // Constructor to initialize the tensor with data
    template<typename D>
    explicit Tensor(const D& data);

    // Helper functions to get the dimensions and size of the tensor

    [[nodiscard]] const std::vector<int>& shape() const;

    [[nodiscard]] int size() const;

    void print() const;

    // Manipulation structure functions

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

    [[maybe_unused]] [[nodiscard]] Tensor<T> tril(const int& axis = 0) const;

    [[maybe_unused]] [[nodiscard]] Tensor<T> triu(const int& axis = 0) const;

    Tensor<T> operator+(const Tensor<T>& other) const;

    Tensor<T> operator-(const Tensor<T>& other) const;

    Tensor<T> operator*(const Tensor<T>& other) const;

    Tensor<T> operator/(const Tensor<T>& other) const;

    Tensor<T> operator+(T scalar) const;

    Tensor<T> operator-(T scalar) const;

    Tensor<T> operator*(T scalar) const;

    Tensor<T> operator/(T scalar) const;

    T& operator[](int index);

    Tensor<T> operator[](const std::vector<int>& indices) const;

    T& operator()(int indices);

    const T& operator()(const std::vector<int>& indices) const;

private:
    std::vector<int> dimensions;
    std::vector<T> data;
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

//    std::vector<int> combineIndices(const std::vector<int>& this_indices, const std::vector<int>& other_indices, int this_rank, int other_rank) const {
//        std::vector<int> result_indices(this_rank + (other_rank - 1), 0);
//
//        // Copy dimensions from this_indices
//        for (int i = 0; i < this_rank - 1; ++i) {
//            result_indices[i] = this_indices[i];
//        }
//
//        // Insert dimensions from other_indices
//        for (int i = 0; i < other_rank - 1; ++i) {
//            result_indices[this_rank - 1 + i] = other_indices[i + 1];
//        }
//
//        return result_indices;
//    }
//
//    int toFlatIndex(const std::vector<int>& indices) const {
//        size_t flatIndex = 0;
//        size_t product = 1;
//        for (size_t i = indices.size(); i > 0; --i) {
//            auto index = static_cast<size_t>(indices[i - 1]);
//            flatIndex += index * product;
//            product *= dimensions[i - 1];
//        }
//        return static_cast<int>(flatIndex);
//    }

};

template class Tensor<float>;
template class Tensor<int>;
template class Tensor<double>;

#endif // TENSOR_H
