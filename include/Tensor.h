#ifndef TENSOR_H
#define TENSOR_H

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <numeric>
#include <cassert>
#include <cmath>
#include <functional>
#include <random>

/**
 * @brief The Tensor class represents a multi-dimensional array (tensor) and provides various operations.
 *
 * This class allows for the creation and manipulation of multi-dimensional arrays (tensors) of various data types.
 * It supports operations such as element-wise addition, subtraction, multiplication, division, reshaping,
 * slicing, and broadcasting.
 *
 * @tparam T The data type of the elements in the tensor (e.g., int, float, double).
 */
template <typename T>
class Tensor {
public:
    std::vector<T> data;
    std::vector<int> dimensions;

    /**
     * @brief Default constructor to create an empty tensor.
     *
     * This constructor creates an empty tensor with zero dimensions and no data.
     */
    Tensor() = default;

    /**
     * @brief Constructs a tensor with specified dimensions and data.
     *
     * This constructor initializes a tensor using the provided dimensions and data. The data is expected
     * to be a vector or nested vectors (depending on the dimensions). The constructor flattens the input data
     * and stores it in the tensor's internal storage. It also checks that the size of the provided data matches
     * the total number of elements implied by the dimensions.
     *
     * @tparam D Type of the input data. This should be a vector or nested vectors that correspond to the tensor's dimensions.
     * @param dims The dimensions of the tensor. It should be a vector of integers where each integer represents the size of that dimension.
     * @param data The data to initialize the tensor with. The size of the data must match the product of the dimensions.
     *
     * @throws std::invalid_argument if the data is not a vector or if data size doesn't match dimensions.
     */
    template<typename D>
    Tensor(const std::vector<int>& dims, const D& data);

    /**
     * @brief Constructs a tensor by inferring dimensions from the provided data.
     *
     * This constructor creates a tensor by inferring its dimensions from the structure of the provided data.
     * The data should be a vector or nested vectors, and the dimensions will be derived automatically based on
     * the depth and size of these vectors.
     *
     * @tparam D Type of the input data. This should be a vector or nested vectors.
     * @param data The data to initialize the tensor with.
     *
     * @throws std::invalid_argument if the data is not a vector.
     *
     * The constructor will flatten the input data and store it in the tensor, calculating the dimensions based
     * on the structure of the input data.
     */
    template<typename D>
    explicit Tensor(const D& data);

    /**
     * @brief Constructs a tensor with specified dimensions, initializing data to zero.
     *
     * This constructor creates a tensor with the specified dimensions and initializes all elements to zero.
     * This is useful when you need a tensor of a specific size but don't have data to initialize it with.
     *
     * @param dims The dimensions of the tensor. Each element in the vector represents the size of that dimension.
     *
     * This constructor will automatically allocate memory for the tensor's data and set all elements to zero.
     */
    explicit Tensor(const std::vector<int>& dims);

    /**
     * @brief Constructs a tensor with specified dimensions using an initializer list, initializing data to zero.
     *
     * This constructor is similar to the one that takes a vector of dimensions but allows for initialization using
     * an initializer list. It is useful when you want to specify the dimensions inline in a compact way.
     *
     * @param dims The dimensions of the tensor as an initializer list.
     *
     * Just like the vector-based constructor, this one also initializes all tensor elements to zero.
     */
    Tensor(std::initializer_list<int> dims);

    /**
     * @brief Constructs a tensor with specified dimensions and reserves space for newSize elements.
     *
     * This constructor creates a tensor with the specified dimensions and reserves space for a specified number
     * of elements. This can be useful when you plan to manually add elements to the tensor after construction.
     *
     * @param dims The dimensions of the tensor.
     * @param newSize The size to reserve for the tensor's data.
     *
     * The tensor's data will be uninitialized after construction, and you will need to manually assign values to it.
     */
    Tensor(const std::vector<int>& dims, const int& newSize);


    /**
     * @brief Returns the shape (dimensions) of the tensor.
     *
     * This method returns the dimensions of the tensor as a vector of integers. Each element in the vector
     * represents the size of the tensor along that dimension.
     *
     * @return const std::vector<int>& The dimensions of the tensor.
     *
     * This method is useful when you need to know the size of the tensor in each dimension for operations
     * like reshaping or broadcasting.
     */
    [[nodiscard]] const std::vector<int>& shape() const;

    /**
     * @brief Returns the total number of elements in the tensor.
     *
     * This method returns the total number of elements in the tensor, which is the product of its dimensions.
     *
     * @return int The number of elements in the tensor.
     *
     * This is useful for operations that require knowledge of the total number of elements, such as when
     * allocating memory or when iterating over all elements.
     */
    [[nodiscard]] int size() const;

    /**
     * @brief Prints the tensor's contents, shape, size, and data type.
     *
     * This method outputs the tensor's contents to the standard output. It displays the data in a structured
     * format that reflects the tensor's dimensions. It also prints the shape, total size, and data type of the tensor.
     *
     * This is a convenient method for debugging or for visually inspecting the tensor's contents.
     */
    void print() const;

    /**
     * @brief Sets the value at the specified indices in the tensor.
     *
     * This method sets the value of the tensor at the specified indices. The indices should be provided as a
     * vector of integers, where each integer corresponds to a dimension of the tensor.
     *
     * @param indices The indices where the value should be set.
     * @param value The value to set at the specified indices.
     *
     * @throws std::invalid_argument if the indices are out of bounds.
     *
     * This method calculates the correct position in the flattened data array based on the provided indices
     * and assigns the given value to that position.
     */
    void set(const std::vector<int>& indices, T value);

    /**
     * @brief Gets the value at the specified indices in the tensor.
     *
     * This method retrieves the value of the tensor at the specified indices. The indices should be provided as a
     * vector of integers, where each integer corresponds to a dimension of the tensor.
     *
     * @param indices The indices from which to get the value.
     * @return T The value at the specified indices.
     *
     * @throws std::invalid_argument if the indices are out of bounds.
     *
     * The method calculates the correct position in the flattened data array based on the provided indices
     * and returns the value stored at that position.
     */
    T get(const std::vector<int>& indices) const;

    /**
     * @brief Fills the tensor with the specified value.
     *
     * This method sets every element in the tensor to the specified value. This is useful when you need to
     * initialize or reset a tensor to a particular value.
     *
     * @param value The value to fill the tensor with.
     *
     * The method iterates over the entire data array and assigns the provided value to each element.
     */
    void fill(T value);

    /**
     * @brief Computes the element-wise square root of the tensor.
     *
     * This method returns a new tensor where each element is the square root of the corresponding element
     * in the original tensor. This is useful for mathematical operations where the square root is needed.
     *
     * @return Tensor<T> A tensor containing the square root of each element.
     *
     * The method creates a new tensor with the same dimensions as the original, and then computes the square root
     * of each element, storing the results in the new tensor.
     */
    Tensor<T> sqrt();

    /**
     * @brief Raises each element of the tensor to the specified power.
     *
     * This method applies element-wise exponentiation to the tensor.
     * Each element in the tensor is raised to the power of the provided exponent.
     *
     * @tparam T Data type of the tensor elements (e.g., float, double).
     * @param exponent The exponent to which each element of the tensor will be raised.
     * @return Tensor<T> A new tensor with the same shape as the original, containing the results of the element-wise exponentiation.
     */
    Tensor<T> pow(T exponent) const;

    /**
     * @brief Applies a function to each element of the tensor.
     *
     * This method applies a given function to each element of the tensor and returns a new tensor with the results.
     * The function should take a single argument of type T and return a value of type T.
     *
     * @param func The function to apply to each element.
     * @return Tensor<T> A tensor with the function applied to each element.
     *
     * The method creates a new tensor with the same dimensions as the original and then applies the function
     * to each element in the original tensor, storing the results in the new tensor.
     */
    Tensor<T> apply(std::function<T(T)> func) const;

    /**
     * @brief Adds another tensor to this tensor element-wise.
     *
     * This method performs element-wise addition of the given tensor `other` to the current tensor.
     * The dimensions of both tensors must match for the addition to be performed.
     *
     * @param other The tensor to add to this tensor.
     *
     * @throws std::invalid_argument if the dimensions of the tensors do not match.
     *
     * This method modifies the current tensor in place by adding the corresponding elements from `other`.
     */
    void add(const Tensor<T>& other);

    /**
     * @brief Computes the sum along the specified axis.
     *
     * This method returns a new tensor that is the sum of the elements along the specified axis.
     * If the axis is not specified, the sum is computed across the last axis by default.
     *
     * @param axis The axis along which to compute the sum. If axis is -1, the last axis is used.
     * @return Tensor<T> A new tensor with the sum computed along the specified axis.
     *
     * @throws std::invalid_argument if the specified axis is out of bounds.
     *
     * This method reduces the dimensionality of the tensor by one along the specified axis.
     */
    Tensor<T> sum(int axis = 0) const;

    /**
     * @brief Computes the mean along the specified axis.
     *
     * This method returns a new tensor that contains the mean of the elements along the specified axis.
     * If the axis is not specified, the mean is computed across the last axis by default.
     *
     * @param axis The axis along which to compute the mean. If axis is -1, the last axis is used.
     * @return Tensor<T> A new tensor with the mean computed along the specified axis.
     *
     * @throws std::invalid_argument if the specified axis is out of bounds.
     *
     * The method computes the mean by summing the elements along the axis and dividing by the size of the axis.
     */
    Tensor<T> mean(int axis = 0) const;

    /**
     * @brief Returns the indices of the maximum values along a specified axis.
     *
     * The `argmax` method computes the index of the maximum value along a specified axis of the tensor.
     * It returns a tensor containing the indices of the maximum values.
     *
     * @param axis The axis along which to compute the indices of the maximum values.
     *             If `axis` is negative, it counts from the last to the first axis.
     *
     * @return A tensor of indices corresponding to the maximum values along the specified axis.
     *         The returned tensor will have the same shape as the original tensor, except along the
     *         specified axis, where it will be reduced to 1 (the dimension corresponding to the indices).
     *
     * @throws std::invalid_argument if the specified axis is out of range for the tensor's dimensions.
     */
    Tensor<T> argmax(int axis = 0) const;

    /**
     * @brief Applies the Softmax function along the specified axis of the tensor.
     *
     * The Softmax function is applied to each element along the specified axis,
     * converting the elements into probabilities that sum to 1. This is useful
     * in classification problems where the output needs to represent probabilities
     * for each class.
     *
     * @param axis The axis along which to apply the Softmax function.
     *             A negative axis value counts from the last dimension backwards
     *             (e.g., -1 for the last axis).
     * @return A new tensor with the same shape as the input tensor, but with
     *         the elements normalized using the Softmax function along the specified axis.
     * @throws std::invalid_argument if the axis is out of range for tensor dimensions.
     */
    Tensor<T> softmax(int axis = 0) const;

    /**
     * @brief Extracts a slice of the tensor along the specified axis.
     *
     * This method returns a new tensor that represents a slice of the original tensor along the specified axis.
     * The slice is defined by the start, end, and step parameters.
     *
     * @param axis The axis along which to slice the tensor.
     * @param start The starting index for the slice.
     * @param end The ending index for the slice.
     * @param step The step size for the slice.
     * @return Tensor<T> A new tensor representing the specified slice.
     *
     * @throws std::invalid_argument if the axis or indices are out of bounds.
     *
     * This method creates a view of the tensor data from `start` to `end` with the specified `step`.
     * It does not modify the original tensor.
     */
    Tensor<T> slice(int axis, int start, int end, int step) const;

    /**
    * @brief Extracts a slice of the tensor along the specified axis with default step of 1.
    *
    * This method returns a new tensor that represents a slice of the original tensor along the specified axis.
    * The slice is defined by the start and end parameters, with a default step size of 1.
    *
    * @param axis The axis along which to slice the tensor.
    * @param start The starting index for the slice.
    * @param end The ending index for the slice.
    * @return Tensor<T> A new tensor representing the specified slice.
    *
    * @throws std::invalid_argument if the axis or indices are out of bounds.
    *
    * This method is a convenience overload that assumes a step size of 1.
    */
    Tensor<T> slice(int axis, int start, int end) const;

    /**
     * @brief Extracts a slice of the tensor along the specified axis starting from the given index to the end.
     *
     * This method returns a new tensor that represents a slice of the original tensor along the specified axis.
     * The slice starts from the specified index and continues to the end of the axis.
     *
     * @param axis The axis along which to slice the tensor.
     * @param start The starting index for the slice.
     * @return Tensor<T> A new tensor representing the specified slice.
     *
     * @throws std::invalid_argument if the axis or start index are out of bounds.
     *
     * This method is a convenience overload that slices from `start` to the end of the axis.
     */
    Tensor<T> slice(int axis, int start) const;

    /**
     * @brief Extracts a slice of the tensor along the entire axis.
     *
     * This method returns a new tensor that represents the entire data along the specified axis.
     *
     * @param axis The axis along which to slice the tensor.
     * @return Tensor<T> A new tensor representing the entire slice along the specified axis.
     *
     * @throws std::invalid_argument if the axis is out of bounds.
     *
     * This method is a convenience overload that slices the entire axis.
     */
    Tensor<T> slice(int axis) const;

    /**
     * @brief Concatenates this tensor with another tensor along the specified axis.
     *
     * This method returns a new tensor that is the result of concatenating the current tensor with another tensor
     * along the specified axis. Both tensors must have matching dimensions except along the concatenation axis.
     *
     * @param other The tensor to concatenate with this tensor.
     * @param axis The axis along which to concatenate the tensors.
     * @return Tensor<T> A new tensor resulting from the concatenation.
     *
     * @throws std::invalid_argument if the tensors are not compatible for concatenation along the specified axis.
     *
     * This method expands the size of the specified axis by adding the size of the corresponding axis in `other`.
     */
    Tensor<T> concatenate(const Tensor<T>& other, int axis) const;

    /**
     * @brief Concatenates this tensor with another tensor along the first axis.
     *
     * This method returns a new tensor that is the result of concatenating the current tensor with another tensor
     * along the first axis. Both tensors must have matching dimensions except along the concatenation axis.
     *
     * @param other The tensor to concatenate with this tensor.
     * @return Tensor<T> A new tensor resulting from the concatenation.
     *
     * @throws std::invalid_argument if the tensors are not compatible for concatenation along the first axis.
     *
     * This method is a convenience overload that concatenates along the first axis.
     */
    Tensor<T> concatenate(const Tensor<T>& other) const;

    /**
     * @brief Expands the dimensions of the tensor along the specified axis.
     *
     * This method returns a new tensor with an expanded dimension along the specified axis. This is useful
     * for preparing the tensor for operations that require a specific dimensionality.
     *
     * @param axis The axis along which to expand the tensor dimensions.
     * @return Tensor<T> A new tensor with expanded dimensions.
     *
     * @throws std::invalid_argument if the axis is out of bounds.
     *
     * The method inserts a new dimension of size 1 along the specified axis.
     */
    Tensor<T> expandDims(int axis) const;

    /**
     * @brief Expands the dimensions of the tensor to match the specified dimensions.
     *
     * This method returns a new tensor whose dimensions are expanded to match the provided `other_dimensions`.
     * This is useful for broadcasting operations where the tensor needs to be expanded to match another tensor.
     *
     * @param other_dimensions The dimensions that the tensor should expand to match.
     * @return Tensor<T> A new tensor with expanded dimensions.
     *
     * @throws std::invalid_argument if the resulting shape is incompatible with the original tensor's shape.
     *
     * This method repeats the tensor's data to match the size of the target dimensions.
     */
    Tensor<T> expandDimsAs(const std::vector<int>& other_dimensions) const;

    /**
     * @brief Removes singleton dimensions from the tensor.
     *
     * This method returns a new tensor with all dimensions of size 1 removed. This is useful for reducing
     * the dimensionality of a tensor after operations like broadcasting.
     *
     * @return Tensor<T> A new tensor with singleton dimensions removed.
     *
     * This method preserves the data of the original tensor and only adjusts the dimensions.
     */
    Tensor<T> squeeze() const;

    /**
     * @brief Reshapes the tensor to the specified shape.
     *
     * This method returns a new tensor that is a reshaped version of the current tensor.
     * The total number of elements must remain the same, but the dimensions can be rearranged.
     *
     * @param newShape The new shape of the tensor as a vector of integers.
     * @return Tensor<T> A new tensor with the specified shape.
     *
     * @throws std::invalid_argument if the new shape is incompatible with the total number of elements.
     *
     * The method preserves the data of the original tensor and only rearranges it to match the new shape.
     */
    Tensor<T> reshape(const std::vector<int>& newShape) const;

    /**
     * @brief Reshapes the tensor to a one-dimensional tensor.
     *
     * This method returns a new tensor that is a reshaped version of the current tensor with a single dimension.
     *
     * @param newShape The new size of the tensor in a single dimension.
     * @return Tensor<T> A new tensor with the specified size in one dimension.
     *
     * @throws std::invalid_argument if the new shape is incompatible with the total number of elements.
     *
     * This method is a convenience overload that flattens the tensor into a one-dimensional vector.
     */
    Tensor<T> reshape(int newShape) const;

    /**
     * @brief Transposes the tensor by permuting its dimensions.
     *
     * This method returns a new tensor that is a transposed version of the current tensor.
     * The transposition is done according to the specified permutation of axes.
     *
     * @param permutation The order in which to permute the tensor's axes. If not specified, the axes are reversed.
     * @return Tensor<T> A new tensor with transposed dimensions.
     *
     * @throws std::invalid_argument if the permutation is invalid for the tensor's shape.
     *
     * The method rearranges the dimensions of the tensor according to the specified order.
     */
    Tensor<T> transpose(const std::vector<int>& permutation = std::vector<int>()) const;

    /**
     * @brief Creates a tensor filled with zeros.
     *
     * This static method creates a new tensor with the specified dimensions and fills it with zeros.
     *
     * @param dims The dimensions of the tensor.
     * @return Tensor<T> A tensor filled with zeros.
     *
     * This method initializes a tensor with the specified shape and all elements set to zero.
     */
    static Tensor<T> zeros(const std::vector<int>& dims);

    /**
     * @brief Creates a tensor filled with ones.
     *
     * This static method creates a new tensor with the specified dimensions and fills it with ones.
     *
     * @param dims The dimensions of the tensor.
     * @return Tensor<T> A tensor filled with ones.
     *
     * This method initializes a tensor with the specified shape and all elements set to one.
     */
    static Tensor<T> ones(const std::vector<int>& dims);

    /**
     * @brief Creates a tensor with uniformly distributed random values.
     *
     * This static method creates a new tensor with the specified dimensions and fills it with random values
     * sampled from a uniform distribution within the specified range.
     *
     * @param dims The dimensions of the tensor.
     * @param lower The lower bound of the uniform distribution. Default is 0.0.
     * @param upper The upper bound of the uniform distribution. Default is 1.0.
     * @return Tensor<T> A tensor filled with uniformly distributed random values.
     *
     * @throws std::invalid_argument if the data type does not support the specified range.
     *
     * This method uses a random number generator to fill the tensor with values.
     */
    static Tensor<T> uniform(const std::vector<int>& dims, T lower = 0.0, T upper = 1.0);

    /**
     * @brief Returns the lower triangular part of the tensor.
     *
     * This method returns a new tensor that contains the lower triangular part of the original tensor.
     * Elements above the diagonal are set to zero.
     *
     * @param axis The diagonal offset from which the lower triangle is taken. Default is 0 (main diagonal).
     * @return Tensor<T> A tensor containing the lower triangular part.
     *
     * @throws std::invalid_argument if the tensor does not have at least two dimensions.
     *
     * This method is useful for extracting the lower triangle of a matrix or batch of matrices.
     */
    Tensor<T> tril(const int& axis = 0);

   /**
    * @brief Returns the upper triangular part of the tensor.
    *
    * This method returns a new tensor that contains the upper triangular part of the original tensor.
    * Elements below the diagonal are set to zero.
    *
    * @param axis The diagonal offset from which the upper triangle is taken. Default is 0 (main diagonal).
    * @return Tensor<T> A tensor containing the upper triangular part.
    *
    * @throws std::invalid_argument if the tensor does not have at least two dimensions.
    *
    * This method is useful for extracting the upper triangle of a matrix or batch of matrices.
    */
    Tensor<T> triu(const int& axis = 0);

    /**
     * @brief Computes the dot product of this tensor with another tensor.
     *
     * This method returns a new tensor that is the result of the dot product between this tensor and another tensor.
     * The dot product is computed along the last dimension of the first tensor and the second-to-last dimension of the second tensor.
     *
     * @param other The tensor to perform the dot product with.
     * @return Tensor<T> A tensor resulting from the dot product operation.
     *
     * @throws std::invalid_argument if the dimensions of the tensors are incompatible for dot product.
     *
     * This method performs a matrix multiplication for higher-dimensional tensors.
     */
    Tensor<T> dot(const Tensor<T>& other) const;

    /**
     * @brief Adds two tensors element-wise.
     *
     * This method returns a new tensor that is the result of element-wise addition of this tensor with another tensor.
     * The dimensions of the tensors must be compatible for broadcasting.
     *
     * @param other The tensor to add.
     * @return Tensor<T> A tensor resulting from the element-wise addition.
     *
     * @throws std::invalid_argument if the tensors are not broadcast-compatible.
     *
     * This method supports broadcasting, allowing tensors of different shapes to be added.
     */
    Tensor<T> operator+(const Tensor<T>& other) const;

    /**
     * @brief Subtracts one tensor from another element-wise.
     *
     * This method returns a new tensor that is the result of element-wise subtraction of another tensor from this tensor.
     * The dimensions of the tensors must be compatible for broadcasting.
     *
     * @param other The tensor to subtract.
     * @return Tensor<T> A tensor resulting from the element-wise subtraction.
     *
     * @throws std::invalid_argument if the tensors are not broadcast-compatible.
     *
     * This method supports broadcasting, allowing tensors of different shapes to be subtracted.
     */
    Tensor<T> operator-(const Tensor<T>& other) const;

    /**
     * @brief Multiplies two tensors element-wise.
     *
     * This method returns a new tensor that is the result of element-wise multiplication of this tensor with another tensor.
     * The dimensions of the tensors must be compatible for broadcasting.
     *
     * @param other The tensor to multiply.
     * @return Tensor<T> A tensor resulting from the element-wise multiplication.
     *
     * @throws std::invalid_argument if the tensors are not broadcast-compatible.
     *
     * This method supports broadcasting, allowing tensors of different shapes to be multiplied.
     */
    Tensor<T> operator*(const Tensor<T>& other) const;
    /**
     * @brief Divides one tensor by another element-wise.
     *
     * This method returns a new tensor that is the result of element-wise division of this tensor by another tensor.
     * The dimensions of the tensors must be compatible for broadcasting.
     *
     * @param other The tensor to divide by.
     * @return Tensor<T> A tensor resulting from the element-wise division.
     *
     * @throws std::invalid_argument if the tensors are not broadcast-compatible.
     *
     * This method supports broadcasting, allowing tensors of different shapes to be divided.
     */
    Tensor<T> operator/(const Tensor<T>& other) const;

   /**
    * @brief Adds a scalar to each element of the tensor.
    *
    * This method returns a new tensor where the specified scalar has been added to each element of the tensor.
    *
    * @param scalar The scalar value to add to each element.
    * @return Tensor<T> A tensor resulting from the addition of the scalar.
    *
    * This method does not modify the original tensor but returns a new tensor with the scalar added.
    */
    Tensor<T> operator+(T scalar) const;

    /**
     * @brief Subtracts a scalar from each element of the tensor.
     *
     * This method returns a new tensor where the specified scalar has been subtracted from each element of the tensor.
     *
     * @param scalar The scalar value to subtract from each element.
     * @return Tensor<T> A tensor resulting from the subtraction of the scalar.
     *
     * This method does not modify the original tensor but returns a new tensor with the scalar subtracted.
     */
    Tensor<T> operator-(T scalar) const;

    /**
     * @brief Multiplies each element of the tensor by a scalar.
     *
     * This method returns a new tensor where each element of the tensor has been multiplied by the specified scalar.
     *
     * @param scalar The scalar value to multiply each element by.
     * @return Tensor<T> A tensor resulting from the multiplication by the scalar.
     *
     * This method does not modify the original tensor but returns a new tensor with each element multiplied by the scalar.
     */
    Tensor<T> operator*(T scalar) const;

    /**
     * @brief Divides each element of the tensor by a scalar.
     *
     * This method returns a new tensor where each element of the tensor has been divided by the specified scalar.
     *
     * @param scalar The scalar value to divide each element by.
     * @return Tensor<T> A tensor resulting from the division by the scalar.
     *
     * This method does not modify the original tensor but returns a new tensor with each element divided by the scalar.
     */
    Tensor<T> operator/(T scalar) const;

    /**
     * @brief Adds another tensor to this tensor element-wise and modifies this tensor.
     *
     * This method performs element-wise addition of the given tensor `other` to the current tensor and modifies this tensor.
     * The dimensions of both tensors must match for the addition to be performed.
     *
     * @param other The tensor to add to this tensor.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * @throws std::invalid_argument if the dimensions of the tensors do not match.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator-=(const Tensor<T>& other);

    /**
     * @brief Subtracts another tensor from this tensor element-wise and modifies this tensor.
     *
     * This method performs element-wise subtraction of the given tensor `other` from the current tensor and modifies this tensor.
     * The dimensions of both tensors must match for the subtraction to be performed.
     *
     * @param other The tensor to subtract from this tensor.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * @throws std::invalid_argument if the dimensions of the tensors do not match.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator+=(const Tensor<T>& other);

    /**
     * @brief Multiplies this tensor by another tensor element-wise and modifies this tensor.
     *
     * This method performs element-wise multiplication of the given tensor `other` with the current tensor and modifies this tensor.
     * The dimensions of both tensors must match for the multiplication to be performed.
     *
     * @param other The tensor to multiply with this tensor.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * @throws std::invalid_argument if the dimensions of the tensors do not match.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator*=(const Tensor<T>& other);

    /**
     * @brief Divides this tensor by another tensor element-wise and modifies this tensor.
     *
     * This method performs element-wise division of the current tensor by the given tensor `other` and modifies this tensor.
     * The dimensions of both tensors must match for the division to be performed.
     *
     * @param other The tensor to divide this tensor by.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * @throws std::invalid_argument if the dimensions of the tensors do not match.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator/=(const Tensor<T>& other);

    /**
     * @brief Adds a scalar to each element of the tensor and modifies this tensor.
     *
     * This method performs element-wise addition of the given scalar to the current tensor and modifies this tensor.
     *
     * @param scalar The scalar value to add to each element.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator+=(const T& scalar);

    /**
     * @brief Subtracts a scalar from each element of the tensor and modifies this tensor.
     *
     * This method performs element-wise subtraction of the given scalar from the current tensor and modifies this tensor.
     *
     * @param scalar The scalar value to subtract from each element.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator-=(const T& scalar);

    /**
     * @brief Multiplies each element of the tensor by a scalar and modifies this tensor.
     *
     * This method performs element-wise multiplication of the current tensor by the given scalar and modifies this tensor.
     *
     * @param scalar The scalar value to multiply each element by.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator*=(const T& scalar);

    /**
     * @brief Divides each element of the tensor by a scalar and modifies this tensor.
     *
     * This method performs element-wise division of the current tensor by the given scalar and modifies this tensor.
     *
     * @param scalar The scalar value to divide each element by.
     * @return Tensor<T>& A reference to the modified tensor.
     *
     * This method modifies the current tensor in place.
     */
    Tensor<T>& operator/=(const T& scalar);

    /**
     * @brief Accesses a sub-tensor or element by indices.
     *
     * This method returns a sub-tensor or element from the tensor based on the provided indices.
     *
     * @param indices The indices to access.
     * @return Tensor<T> The sub-tensor or element at the specified indices.
     *
     * @throws std::invalid_argument if the indices are out of bounds.
     *
     * The method supports both positive and negative indexing.
     */
    Tensor<T> operator[](const std::vector<int>& indices);

    /**
     * @brief Accesses a sub-tensor or element by indices.
     *
     * This method returns a sub-tensor or element from the tensor based on the provided indices.
     * This is a const version of the operator, meaning the tensor cannot be modified.
     *
     * @param indices The indices to access.
     * @return Tensor<T> The sub-tensor or element at the specified indices.
     *
     * @throws std::invalid_argument if the indices are out of bounds.
     *
     * The method supports both positive and negative indexing.
     */
    Tensor<T> operator[](const std::vector<int>& indices) const;

    /**
     * @brief Accesses an element by its flat index.
     *
     * This method returns a reference to the element at the specified flat index.
     *
     * @param index The flat index of the element.
     * @return T& A reference to the element at the specified index.
     *
     * @throws std::invalid_argument if the index is out of bounds.
     *
     * This method does not create a sub-tensor, it directly accesses the element at the given flat index.
     */
    T& operator()(int index);

    /**
     * @brief Accesses an element by its multi-dimensional indices.
     *
     * This method returns a reference to the element at the specified multi-dimensional indices.
     *
     * @param indices The multi-dimensional indices of the element.
     * @return T& A reference to the element at the specified indices.
     *
     * @throws std::invalid_argument if the indices are out of bounds.
     *
     * This method flattens the multi-dimensional indices and accesses the corresponding element in the tensor.
     */
    T& operator()(const std::vector<int>& indices);

    /**
     * @brief Checks if two tensors are equal.
     *
     * This method compares two tensors element-wise and returns true if all corresponding elements are equal.
     *
     * @param other The tensor to compare with.
     * @return bool True if the tensors are equal, false otherwise.
     *
     * This method does not consider floating-point tolerance; it requires exact equality.
     */
    bool operator==(const Tensor<T>& other) const;

    /**
     * @brief Checks if two tensors are not equal.
     *
     * This method compares two tensors element-wise and returns true if any corresponding elements are not equal.
     *
     * @param other The tensor to compare with.
     * @return bool True if the tensors are not equal, false otherwise.
     *
     * This method does not consider floating-point tolerance; it requires exact inequality.
     */
    bool operator!=(const Tensor<T>& other) const;

    /**
     * @brief Serializes the tensor to an output stream.
     *
     * This method writes the tensor's shape and data to the specified output stream in a human-readable format.
     *
     * @param os The output stream to write to.
     *
     * This method is useful for saving the tensor's state to a file or sending it over a network.
     */
    void serialize(std::ostream& os) const;

    /**
     * @brief Deserializes the tensor from an input stream.
     *
     * This method reads the tensor's shape and data from the specified input stream.
     *
     * @param is The input stream to read from.
     *
     * This method is useful for loading the tensor's state from a file or receiving it over a network.
     */
    void deserialize(std::istream& is);

private:
    /**
     * @brief Stores the stride values for each dimension of the tensor.
     *
     * The strides are used to efficiently calculate the flat index of a multi-dimensional index.
     * Each stride represents the number of steps needed to move one unit along a particular dimension.
     */
    std::vector<int> strides;

    /**
     * @brief Calculates the flat index for a given multi-dimensional index.
     *
     * This method takes a multi-dimensional index and calculates the corresponding flat index in the underlying data storage.
     *
     * @param indices The multi-dimensional index to convert.
     * @return int The corresponding flat index.
     *
     * @throws std::invalid_argument if the number of indices does not match the number of dimensions.
     *
     * This method is essential for accessing elements in the tensor's underlying data array.
     * It multiplies each index by its corresponding stride and sums the results to obtain the flat index.
     */
    [[nodiscard]] int calculateIndex(const std::vector<int>& indices) const;

    /**
     * @brief Calculates the strides for the tensor's dimensions.
     *
     * This method computes the stride for each dimension of the tensor, which is used to calculate flat indices from multi-dimensional indices.
     *
     * @return std::vector<int> A vector of strides corresponding to each dimension.
     *
     * The stride for a dimension is calculated as the product of the sizes of all dimensions that come after it.
     */
    [[nodiscard]] std::vector<int> calculateStrides() const;

    /**
     * @brief Calculates the total size of a tensor given its dimensions.
     *
     * This method calculates the total number of elements in a tensor by multiplying all its dimension sizes together.
     *
     * @param dims The dimensions of the tensor.
     * @return int The total number of elements in the tensor.
     *
     * This method is used to allocate the correct amount of memory for the tensor's data storage.
     */
    static int getTotalSize(const std::vector<int>& dims);

    /**
     * @brief Checks if a given type is a vector.
     *
     * This is a helper template struct used to determine whether a given type is a `std::vector`.
     * The base case returns `false`.
     */
    template <typename D>
    struct is_vector : std::false_type {};

    /**
     * @brief Specialization of is_vector for `std::vector` types.
     *
     * This specialization returns `true` if the given type is a `std::vector`.
     */
    template <typename D, typename Allocator>
    struct is_vector<std::vector<D, Allocator>> : std::true_type {};

    /**
     * @brief Extracts the innermost type from a nested vector structure.
     *
     * This template struct recursively traverses nested vectors to extract the innermost type.
     *
     * @tparam D The type to extract from.
     */
    template <typename D>
    struct ExtractType {
        using Type = T; // This is the innermost type
    };

    /**
     * @brief Specialization of ExtractType for nested vectors.
     *
     * This specialization recursively extracts the innermost type from nested vectors.
     */
    template <typename D>
    struct ExtractType<std::vector<D>> {
        using Type = typename ExtractType<D>::Type; // Recursively extract inner type
    };

    /**
     * @brief Flattens a nested vector into a single-dimensional vector.
     *
     * This method recursively flattens a nested vector structure into a single-dimensional vector.
     *
     * @param vec The nested vector to flatten.
     * @param result The flattened vector to store the result.
     *
     * This method is useful for converting multi-dimensional data into a format that can be stored in the tensor's data array.
     */
    template <typename D>
    void flatten(const D& vec, std::vector<T>& result);

    /**
     * @brief Computes the shape of a nested vector.
     *
     * This method recursively computes the shape (dimensions) of a nested vector structure.
     *
     * @param vec The nested vector whose shape is to be computed.
     * @return std::vector<int> A vector representing the shape of the nested structure.
     *
     * This method is useful for determining the dimensions of a tensor when it is constructed from a nested vector.
     */
    template <typename D>
     std::vector<int> compute_shape(const D& vec);

    /**
     * @brief Combines indices from two tensors for operations like matrix multiplication.
     *
     * This method combines the indices of two tensors based on their ranks (number of dimensions) to form a single set of indices.
     *
     * @param this_indices The indices from the first tensor.
     * @param other_indices The indices from the second tensor.
     * @param this_rank The rank (number of dimensions) of the first tensor.
     * @param other_rank The rank (number of dimensions) of the second tensor.
     * @return std::vector<int> The combined indices.
     *
     * This method is useful for operations that involve combining two tensors, such as dot products or concatenations.
     */
    static std::vector<int> combineIndices(const std::vector<int>& this_indices,
     const std::vector<int>& other_indices, int this_rank, int other_rank);

    /**
     * @brief Converts multi-dimensional indices to a flat index.
     *
     * This method converts a multi-dimensional index into a flat index for accessing elements in the tensor's data array.
     *
     * @param indices The multi-dimensional indices to convert.
     * @return int The corresponding flat index.
     *
     * This method is similar to `calculateIndex` but optimized for certain use cases.
     */
    [[nodiscard]] int toFlatIndex(const std::vector<int>& indices) const;
};

#include "../src/Tensor.tpp"

template class Tensor<float>;
template class Tensor<int>;
template class Tensor<double>;

#endif // TENSOR_H
