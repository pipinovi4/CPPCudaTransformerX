#ifndef TENSOR_TPP
#define TENSOR_TPP

#include "../include/Tensor.h"

template<typename T>
template<typename D>
Tensor<T>::Tensor(const std::vector<int>& dims, const D& data) {
    /*
     * Header:
     * \brief Constructor for a tensor with specified dimensions and data
     *
     * Params:
     * \param dims The dimensions of the tensor
     * \param data The data to be stored in the tensor
     *
     * TParams:
     * \tparam D The type of the data
     *
     * Exceptions:
     * \throws std::invalid_argument If the data is not a vector
     * \throws std::invalid_argument If the data size does not match the specified dimensions
     * \throws std::invalid_argument If the number of indices does not match the number of dimensions
     *
     * Notes:
     * The constructor initializes the tensor with the specified data and dimensions.
     * For the data, the constructor flattens the input data and stores it in the tensor.
     *
     * Example cases:
     * In the case when specified data and dimensions, the data size must match the specified dimensions.
     * In the case when aren't specified data and specified dimensions, the data will be initialized with zeros.
     * In the case when specified data and aren't specified dimensions, the dimensions will be taken from the source data.
     * In the case when aren't specified data and dimensions, will create an empty Tensor.
     *
     * Rules for the data:
     * Data must be a vector.
     * The number of indices must match the vector of number dimensions.
    */

    if (!is_vector<D>::value) {
        throw std::invalid_argument("Data must be a vector");
    }

    int dims_size = getTotalSize(dims);

    if (!dims.empty()) {
        this->dimensions = dims;
        std::vector<int> computed_shape = compute_shape(data);

        if (data.empty()) {
            this->data.resize(dims_size, T(0));
        } else {
            if (dims_size != std::accumulate(computed_shape.begin(), computed_shape.end(), 1, std::multiplies<>())) {
                throw std::invalid_argument("Data size does not match the specified dimensions");
            }
            std::vector<typename ExtractType<D>::Type> flattened_data;
            flatten(data, flattened_data);
            this->data = flattened_data;
            this->dimensions = dims;
        }
    } else if (!data.empty()) {
        std::vector<typename ExtractType<D>::Type> flattened_data;
        flatten(data, flattened_data);
        this->data = flattened_data;
        this->dimensions = dims;
    }

    this->strides = calculateStrides();
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& dims) : Tensor<T>(dims, std::vector<T>()) {
    int dims_size = getTotalSize(dims);
    this->dimensions = dims;
    this->data.resize(dims_size, T(0));

    if (dims_size != this->data.size()) {
        throw std::invalid_argument("Data size does not match the specified dimensions");
    }

    this->strides = calculateStrides();
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<int> dims) : Tensor<T>(std::vector<int>(dims), std::vector<T>()) {
    int dims_size = getTotalSize(dims);
    std::vector<int> dims_vec(dims);
    this->data.resize(dims_size, T(0));

    if (dims_size != this->data.size()) {
        throw std::invalid_argument("Data size does not match the specified dimensions");
    }

    this->strides = calculateStrides();
}

template<typename T>
template<typename D>
Tensor<T>::Tensor(const D &data) : Tensor<T>(std::vector<int>(), data) {
    std::vector<int> computed_shape = compute_shape(data);
    if (!is_vector<D>::value) {
        throw std::invalid_argument("Data must be a vector");
    }

    std::vector<typename ExtractType<D>::Type> flattened_data;
    flatten(data, flattened_data);
    this->data = flattened_data;
    this->dimensions = compute_shape(data);
    this->strides = calculateStrides();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int>& dims, const int& newSize) : dimensions(dims){
    this->data.reserve(newSize);
    this->strides = calculateStrides();
}


template<typename T>
const std::vector<int>& Tensor<T>::shape() const {
    return dimensions;
}

template<typename T>
int Tensor<T>::size() const {
    return data.size();
}

template<typename T>
void Tensor<T>::set(const std::vector<int>& indices, T value) {
    int index = calculateIndex(indices);
    data[index] = value;
}

template<typename T>
T Tensor<T>::get(const std::vector<int>& indices) const {
    int index = calculateIndex(indices);
    return data[index];
}

template<typename T>
void Tensor<T>::print() const {
    if (dimensions.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }

    std::vector<size_t> indices(dimensions.size(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        // Print indices
        if (indices.back() == 0) {
            for (size_t j = dimensions.size() - 1; j < dimensions.size(); --j) {
                if (indices[j] == 0) {
                    std::cout << "[";
                }
            }
        }

        // Print data
        std::cout << data[i];

        // Close brackets
        if (indices.back() == dimensions.back() - 1) {
            for (size_t j = dimensions.size() - 1; j < dimensions.size(); --j) {
                if (indices[j] == dimensions[j] - 1) {
                    std::cout << "]";
                }
            }
        } else {
            std::cout << ", ";
        }

        // Newline between rows
        if (indices.back() == dimensions.back() - 1 && i != data.size() - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << "\n\n";
    std::cout << "Shape: ";
    for (size_t dim = 0; dim < dimensions.size(); ++dim) {
        if (dim == 0) {
            std::cout << "(";
        }
        std::cout << dimensions[dim];
        if (dim < dimensions.size() - 1) {
            std::cout << ", ";
        } else {
            std::cout << ")";
        }
    }
    std::cout << " | ";
    std::cout << "Size: " << data.size() << " | ";
    std::cout << "Dtype: ";

    if (std::is_same<T, int>::value) {
        std::cout << "int";
    } else if (std::is_same<T, float>::value) {
        std::cout << "float";
    } else if (std::is_same<T, double>::value) {
        std::cout << "double";
    } else {
        std::cout << typeid(T).name();
    }

    std::cout << "\n" << std::endl;
}

template<typename T>
void Tensor<T>::fill(T value) {
    for (T& element : data) {
        element = value;
    }
}

template<typename T>
Tensor<T> Tensor<T>::sqrt() {
    Tensor<T> result(dimensions);
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = std::sqrt(data[i]);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::sum(int axis) const {
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    std::vector<int> new_dims = dimensions;
    new_dims.erase(new_dims.begin() + axis);

    Tensor<T> result(new_dims);

    std::vector<int> indices(dimensions.size(), 0);
    std::vector<int> result_indices(new_dims.size(), 0);

    for (size_t i = 0; i < data.size(); ++i) {
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        for (size_t j = 0; j < indices.size(); ++j) {
            if (j < axis) {
                result_indices[j] = indices[j];
            } else if (j > axis) {
                result_indices[j - 1] = indices[j];
            }
        }

        int result_index = result.calculateIndex(result_indices);
        result.data[result_index] += data[i];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start, const int end, const int step) const {
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Calculate effective start, end and step
    const int actual_start = (start < 0) ? (dimensions[axis] + start) : start;
    const int actual_end = (end < 0) ? (dimensions[axis] + end) : end;
    const int actual_step = (step == 0) ? 1 : step;

    // Validate indices
    if (actual_start < 0 || actual_start >= dimensions[axis]) {
        throw std::invalid_argument("Invalid start index");
    }
    if (actual_end < 0 || actual_end > dimensions[axis]) {
        throw std::invalid_argument("Invalid end index");
    }

    // Compute new dimensions for the sliced tensor
    std::vector<int> new_dims = dimensions;
    new_dims[axis] = (actual_end - actual_start + actual_step - 1) / actual_step;

    // Create a new tensor with the computed dimensions
    Tensor<T> sliced_tensor(new_dims);

    // Copy data from the original tensor based on the slice parameters
    std::vector<int> indices(dimensions.size(), 0);
    std::vector<int> sliced_indices(dimensions.size(), 0);

    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices in the original tensor
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        // Check if the current index falls within the slice range along the specified axis
        if (indices[axis] >= actual_start && indices[axis] < actual_end && (indices[axis] - actual_start) % actual_step == 0) {
            // Calculate the corresponding index in the sliced tensor
            for (size_t j = 0; j < indices.size(); ++j) {
                sliced_indices[j] = (j == axis) ? (indices[j] - actual_start) / actual_step : indices[j];
            }

            // Calculate the index in the sliced tensor
            int sliced_index = sliced_tensor.calculateIndex(sliced_indices);

            // Copy the data from the original tensor to the sliced tensor
            sliced_tensor.data[sliced_index] = data[i];
        }
    }

    // Optional: Handle squeezing of dimensions
    for (const int& dim : new_dims) {
        if (dim < 1) {
            return sliced_tensor.squeeze(); // Assuming squeeze returns by value
        }
    }

    return sliced_tensor; // Return by value
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start, const int end) const {
    return slice(axis, start, end, 1);
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start) const {
    return slice(axis, start, dimensions[axis], 1);
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis) const {
    return slice(axis, 0, dimensions[axis], 1);
}

template<typename T>
Tensor<T> Tensor<T>::concatenate(const Tensor<T>& other, int axis) const {
    // Ensure axis is valid
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Check dimensions for concatenation axis
    if (dimensions[axis] != other.shape()[axis]) {
        throw std::invalid_argument("Dimensions mismatch along concatenation axis");
    }

    // Determine new dimensions for the concatenated tensor
    std::vector<int> newDimensions = dimensions;
    newDimensions[axis] += other.shape()[axis];

    // Create a new tensor for concatenation result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(dimensions.size(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    // Copy data from the other tensor
    indices.assign(dimensions.size(), 0);
    for (size_t i = 0; i < other.size(); ++i) {
        // Calculate current indices for the other tensor
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % other.shape()[j];
            temp /= other.shape()[j];
        }
        // Retrieve the value from the other tensor
        T value = other.get(indices);

        // Adjust the axis index for concatenation
        indices[axis] += other.shape()[axis];

        // Copy data into the result tensor
        result.set(indices, value);
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::concatenate(const Tensor<T> &other) const {
    return concatenate(other, 0);
}

template<typename T>
Tensor<T> Tensor<T>::expandDims(int axis) const {
    // Ensure axis is valid
    if (axis < 0 || axis > dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Create new dimensions for the expanded tensor
    std::vector<int> newDimensions = dimensions;
    newDimensions.insert(newDimensions.begin() + axis, 1);

    // Create a new tensor for the expanded result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(dimensions.size() + 1, 0);  // Increase indices size for new axis
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices
        size_t temp = i;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (j < axis) {
                indices[j] = temp % dimensions[j];
                temp /= dimensions[j];
            } else if (j > axis) {
                indices[j] = temp % dimensions[j - 1];  // Adjust indices for new axis
                temp /= dimensions[j - 1];
            }
        }

        // Set the new axis index to 0
        indices[axis] = 0;

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::squeeze() const {
    // Step 1: Calculate the number of dimensions to be removed and the new dimensions.
    std::vector<int> newDimensions;
    newDimensions.reserve(dimensions.size());

    // Use size_t for indices and iterators
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (dimensions[i] != 1) {
            newDimensions.push_back(dimensions[i]);
        }
    }

    // Step 2: Create a new tensor for the squeezed result.
    Tensor<T> result(newDimensions);

    // Step 3: Efficiently copy data to the new tensor.
    if (newDimensions.size() == dimensions.size()) {
        // If no dimensions were removed, directly copy the data
        result.data = data;
        return result;
    }

    size_t oldStride = 1;
    std::vector<size_t> oldStrides(dimensions.size());
    std::vector<size_t> newStrides(newDimensions.size());

    // Calculate strides for old dimensions
    for (int i = static_cast<int>(dimensions.size()) - 1; i >= 0; --i) {
        oldStrides[i] = oldStride;
        oldStride *= dimensions[i];
    }

    // Calculate strides for new dimensions
    size_t newStride = 1;
    for (int i = static_cast<int>(newDimensions.size()) - 1; i >= 0; --i) {
        newStrides[i] = newStride;
        newStride *= newDimensions[i];
    }

    // Use a single loop with accumulated strides to copy data efficiently
    for (size_t i = 0; i < data.size(); ++i) {
        size_t newIndex = 0;
        size_t oldIndex = i;
        for (size_t j = 0, newDimIndex = 0; j < dimensions.size(); ++j) {
            if (dimensions[j] != 1) {
                newIndex += (oldIndex / oldStrides[j]) * newStrides[newDimIndex++];
            }
            oldIndex %= oldStrides[j];
        }
        result.data[newIndex] = data[i];
    }

    return result;
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& newDimensions) const {
    // Calculate the total size of the new dimensions
    int newTotalSize = 1;
    for (const int dim : newDimensions) {
        newTotalSize *= dim;
    }

    // Create a new tensor for the reshaped result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    result.data = data;

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const int newShape) const {
    return reshape(std::vector<int>{newShape});
}

template<typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<int> &dims) {
    Tensor<T> tensor(dims);
    tensor.fill(T(0));
    return tensor;
}

template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<int>& permutation) const {
    // Create new dimensions for the transposed tensor
    std::vector<int> newDimensions;
    if (permutation.empty()) {
        newDimensions = std::vector<int>(dimensions.rbegin(), dimensions.rend());
    } else {
        for (size_t i = 0; i < permutation.size(); ++i) {
            newDimensions.push_back(dimensions[permutation[i]]);
        }
    }

    // Create a new tensor for the transposed result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(dimensions.size(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices in the original tensor
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        // Calculate transposed indices
        std::vector<int> transposedIndices(indices.size());
        if (permutation.empty()) {
            for (size_t j = 0; j < indices.size(); ++j) {
                transposedIndices[j] = indices[indices.size() - 1 - j];
            }
        } else {
            for (size_t j = 0; j < permutation.size(); ++j) {
                transposedIndices[j] = indices[permutation[j]];
            }
        }

        // Copy data into the result tensor
        result.set(transposedIndices, data[i]);
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::ones(const std::vector<int>& dims) {
    Tensor<T> tensor(dims);
    tensor.fill(T(1.0));
    return tensor;
}

template<typename T>
Tensor<T> Tensor<T>::uniform(const std::vector<int>& dims, T lower, T upper) {
    Tensor<T> result(dims);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<T> dis(lower, upper);
        for (T& element : result.data) {
            element = dis(gen);
        }
    } else if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dis(lower, upper);
        for (T& element : result.data) {
            element = dis(gen);
        }
    } else {
        throw std::invalid_argument("Unsupported type for uniform distribution");
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::tril(const int& axis) {
    const int dimSize = dimensions.size();
    if (dimSize < 2) {
        throw std::invalid_argument("Tensor must have at least 2 dimensions for tril operation.");
    }

    // Get the shape of the matrix and batch dimensions
    const int rows = dimensions[dimSize - 2];
    const int cols = dimensions[dimSize - 1];
    std::vector<int> batchDimensions(dimSize - 2);
    std::copy(dimensions.begin(), dimensions.end() - 2, batchDimensions.begin());

    // Compute the total size for batch dimensions
    int batchSize = 1;
    for (const int dim : batchDimensions) {
        batchSize *= dim;
    }

    // Create new data vector for the result
    std::vector<T> newData(data.size());

    // Iterate over each element in the matrix dimensions
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int index = batchIndex * rows * cols + i * cols + j;
                if (j > i + axis) {
                    newData[index] = static_cast<T>(0);  // Set elements above the diagonal to zero
                } else {
                    newData[index] = data[index];  // Copy elements below or on the diagonal
                }
            }
        }
    }

    // Return the new tensor with the same dimensions and the modified data
    return Tensor<T>(dimensions, newData);
}

template<typename T>
Tensor<T> Tensor<T>::triu(const int& axis) {
    const int dimSize = dimensions.size();
    if (dimSize < 2) {
        throw std::invalid_argument("Tensor must have at least 2 dimensions for triu operation.");
    }

    // Get the shape of the matrix and batch dimensions
    const int rows = dimensions[dimSize - 2];
    const int cols = dimensions[dimSize - 1];
    std::vector<int> batchDimensions(dimSize - 2);
    std::copy(dimensions.begin(), dimensions.end() - 2, batchDimensions.begin());

    // Compute the total size for batch dimensions
    int batchSize = 1;
    for (const int dim : batchDimensions) {
        batchSize *= dim;
    }

    // Create new data vector for the result
    std::vector<T> newData(data.size());

    // Iterate over each element in the matrix dimensions
    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int index = batchIndex * rows * cols + i * cols + j;
                if (i > j + axis) {
                    newData[index] = static_cast<T>(0);  // Set elements below the diagonal to zero
                } else {
                    newData[index] = data[index];  // Copy elements above or on the diagonal
                }
            }
        }
    }

    // Return the new tensor with the same dimensions and the modified data
    return Tensor<T>(dimensions, newData);
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T>& other) const {
    // Copy dimensions to handle broadcasting
    std::vector<int> this_dims = dimensions;
    std::vector<int> other_dims = other.dimensions;

    // Handle broadcasting dimensions
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Ensure compatible dimensions for dot product
    if (this_dims.back() != other_dims[other_dims.size() - 2]) {
        throw std::invalid_argument("The last dimension of the first tensor must match the second-to-last dimension of the second tensor.");
    }

    // Compute result dimensions
    std::vector<int> resultDimensions(this_dims.begin(), this_dims.end() - 1);
    resultDimensions.insert(resultDimensions.end(), other_dims.begin(), other_dims.end() - 2);
    resultDimensions.push_back(other_dims.back());

    // Initialize result tensor
    Tensor<T> result(resultDimensions);
    T* result_data = result.data.data();

    // Perform the dot product
    const int M = this_dims[0]; // Batch size or outer dimension
    const int N = other_dims.back(); // The final output dimension
    const int K = this_dims.back(); // The shared dimension for the dot product

    // Access data pointers for faster access
    const T* A = data.data();
    const T* B = other.data.data();

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = static_cast<T>(0);
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            result_data[i * N + j] = sum;
        }
    }

    // Return the squeezed result tensor
    return result.squeeze();
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for addition");
        }
    }

    // Determine resulting dimensions
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create result tensor
    Tensor<T> result(result_dims);
    size_t result_size = result.data.size();

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform element-wise addition with broadcasting
    for (size_t i = 0; i < result_size; ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        for (size_t j = result_dims.size(); j-- > 0;) {
            int result_idx = temp % result_dims[j];
            temp /= result_dims[j];

            this_index += (this_dims[j] == 1 ? 0 : result_idx) * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        result.data[i] = this->data[this_index] + other.data[other_index];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const
{
    // Check if dimensions are compatible for broadcasting
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for subtraction");
        }
    }

    // Calculate resulting dimensions
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create a new tensor to hold the result
    Tensor<T> result(result_dims);

    // Perform element-wise subtraction with broadcasting
    for (size_t i = 0; i < result.data.size(); ++i) {
        std::vector<int> result_indices(result_dims.size());
        size_t temp = i;
        for (size_t j = result_indices.size(); j-- > 0;) {
            result_indices[j] = temp % result_dims[j];
            temp /= result_dims[j];
        }

        std::vector<int> this_indices(this_dims.size());
        std::vector<int> other_indices(other_dims.size());
        for (size_t j = 0; j < result_indices.size(); ++j) {
            this_indices[j] = (this_dims[j] == 1) ? 0 : result_indices[j];
            other_indices[j] = (other_dims[j] == 1) ? 0 : result_indices[j];
        }

        int this_index = 0;
        int this_stride = 1;
        int other_index = 0;
        int other_stride = 1;
        for (int j = this_dims.size() - 1; j >= 0; --j) {
            this_index += this_indices[j] * this_stride;
            this_stride *= this_dims[j];
            other_index += other_indices[j] * other_stride;
            other_stride *= other_dims[j];
        }

        result.data[i] = this->data[this_index] - other.data[other_index];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    // Assuming tensors are of the same shape for simplicity
    Tensor<T> result(dimensions, std::vector<T>(data.size(), T()));

    if (dimensions != other.shape()) {
        throw std::invalid_argument("Dimensions mismatch");
    }

    // Perform element-wise multiplication
    const size_t size = Tensor<T>::getTotalSize(dimensions);
    for (size_t i = 0; i < size; ++i) {
        std::vector<int> indices(dimensions.size());
        size_t temp = i;
        for (size_t d = dimensions.size(); d > 0; --d) {
            indices[d - 1] = temp % dimensions[d - 1];
            temp /= dimensions[d - 1];
        }
        result(indices) = data[i] * other.get(indices);
    }
    return result;
}


template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
    // Assuming tensors are of the same shape for simplicity
    Tensor<T> result(dimensions, std::vector<T>(data.size(), T()));

    if (dimensions != other.shape()) {
        throw std::invalid_argument("Dimensions mismatch");
    }

    // Perform element-wise multiplication
    const size_t size = Tensor<T>::getTotalSize(dimensions);
    for (size_t i = 0; i < size; ++i) {
        std::vector<int> indices(dimensions.size());
        size_t temp = i;
        for (size_t d = dimensions.size(); d > 0; --d) {
            indices[d - 1] = temp % dimensions[d - 1];
            temp /= dimensions[d - 1];
        }
        result(indices) = data[i] / other.get(indices);
    }
    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
    // Create a new tensor for the result
    Tensor<T> result(dimensions);

    // Perform element-wise addition with scalar
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + scalar;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(T scalar) const {
    // Create a new tensor for the result
    Tensor<T> result(dimensions);

    // Perform element-wise subtraction with scalar
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - scalar;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T scalar) const {
    // Create a new tensor for the result
    Tensor<T> result(dimensions);

    // Perform element-wise multiplication with scalar
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T scalar) const {
    // Create a new tensor for the result
    Tensor<T> result(dimensions);

    // Perform element-wise division with scalar
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] / scalar;
    }

    return result;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
    *this = *this - other;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
    *this = *this + other;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
    *this = *this / other;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
    *this = *this * other;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(T& scalar) {
    *this = *this - scalar;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(T& scalar) {
    *this = *this + scalar;
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(T& scalar) {
    *this = *this / scalar;
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::operator[](const std::vector<int>& indices) {
    // Check for index out of bounds and adjust for negative indices
    std::vector<int> positiveIndices = indices;
    for (size_t i = 0; i < indices.size(); ++i) {
        int dimSize = dimensions[i];
        if (std::abs(indices[i]) >= dimSize) {
            throw std::invalid_argument("Index out of bounds");
        }
        if (indices[i] < 0) {
            positiveIndices[i] = dimSize + indices[i];
        }
    }

    // Calculate the shape of the resulting tensor
    std::vector<int> resultDimensions;
    for (size_t i = indices.size(); i < dimensions.size(); ++i) {
        resultDimensions.push_back(dimensions[i]);
    }

    // Calculate strides for efficient indexing
    std::vector<int> strides(dimensions.size(), 1);
    for (int i = dimensions.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * dimensions[i + 1];
    }

    // Calculate the starting offset in the flattened data array
    int offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += positiveIndices[i] * strides[i];
    }

    // Extract the required data from the original data array
    int resultSize = 1;
    for (int dim : resultDimensions) {
        resultSize *= dim;
    }

    std::vector<T> newData(resultSize);
    for (int i = 0; i < resultSize; ++i) {
        int linearIndex = offset;
        int temp = i;
        for (size_t j = 0; j < resultDimensions.size(); ++j) {
            linearIndex += (temp % resultDimensions[j]) * strides[j + indices.size()];
            temp /= resultDimensions[j];
        }
        newData[i] = data[linearIndex];
    }

    // Return the new Tensor by value
    return Tensor<T>(resultDimensions, newData);
}

template<typename T>
Tensor<T> Tensor<T>::operator[](const std::vector<int>& indices) const {
    return const_cast<Tensor<T>&>(*this)[indices];
}

template<typename T>
T& Tensor<T>::operator()(int indices) {
    int flatIndex = toFlatIndex({indices});
    return data[flatIndex];
}

template<typename T>
T& Tensor<T>::operator()(const std::vector<int>& indices) {
    int flatIndex = toFlatIndex(indices);
    return data[flatIndex];
}

template<typename T>
bool Tensor<T>::operator==(const Tensor<T>& other) const {
    // Check if dimensions match
    if (this->dimensions != other.dimensions) {
        return false;
    }

    // Check if data matches
    for (size_t i = 0; i < this->data.size(); ++i) {
        if (this->data[i] != other.data[i]) {
            return false;
        }
    }

    return true;
}

template<typename T>
bool Tensor<T>::operator!=(const Tensor<T>& other) const {
    return !(*this == other);
}

template <typename T>
void Tensor<T>::serialize(std::ostream& os) const {
    // Write the shape
    os << "{";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        os << dimensions[i];
        if (i < dimensions.size() - 1) {
            os << ", ";
        }
    }
    os << "}, ";

    // Write the data
    os << "{";
    for (size_t i = 0; i < data.size(); ++i) {
        os << data[i];
        if (i < data.size() - 1) {
            os << ", ";
        }
    }
    os << "}";
}

template <typename T>
void Tensor<T>::deserialize(std::istream& is) {
    char ch;
    // Read the shape
    is >> ch; // Read '{'
    dimensions.clear();
    int dim;
    while (is >> dim) {
        dimensions.push_back(dim);
        is >> ch; // Read ',' or '}'
        if (ch == '}') break;
    }
    is >> ch; // Read ','

    // Read the data
    is >> ch; // Read '{'
    data.clear();
    data.reserve(getTotalSize(dimensions));
    T value;
    while (is >> value) {
        data.push_back(value);
        is >> ch; // Read ',' or '}'
        if (ch == '}') break;
    }
    is >> ch; // Read ','
}

#endif // TENSOR_TPP