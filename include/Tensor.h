#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include "MixedPrecisionFloat16.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <type_traits>

template <typename T>
class Tensor {
private:
    std::vector<int> dimensions;
    std::vector<T> data;

    [[nodiscard]] int calculateIndex(const std::vector<int>& indices) const;

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

public:
    // Constructor to initialize the tensor with a vector of vectors

    template<typename D>
    explicit Tensor(const std::vector<int>& dims, const D& data);

    explicit Tensor(const std::vector<int>& dims);

    template<typename D>
    explicit Tensor(const D& data);

    // Helper functions to get the dimensions and size of the tensor

    [[maybe_unused]] [[nodiscard]] const std::vector<int>& shape() const;

    [[maybe_unused]] [[nodiscard]] int size() const;

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

    Tensor<T> operator[](int index) const;

    Tensor<T> operator[](const std::vector<int>& indices) const;

    void print() const;
};

template class Tensor<float>;
template class Tensor<int>;
template class Tensor<double>;

#include "../src/Tensor.cpp"

#endif // TENSOR_H
