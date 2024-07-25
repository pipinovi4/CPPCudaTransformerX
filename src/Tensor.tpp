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
    } else if (std::is_same<T, float_16>::value) {
        std::cout << "float_16";
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

    return sliced_tensor.squeeze();
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
    // Find the number of dimensions to be removed
    int numDimsToRemove = 0;
    for (int dim : dimensions) {
        if (dim == 1) {
            ++numDimsToRemove;
        }
    }

    // Create new dimensions for the squeezed tensor
    std::vector<int> newDimensions;
    for (int dim : dimensions) {
        if (dim != 1) {
            newDimensions.push_back(dim);
        }
    }

    // Create a new tensor for the squeezed result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(newDimensions.size(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = static_cast<int>(temp) % newDimensions[j];
            temp /= newDimensions[j];
        }

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& newDimensions) const {
    // Calculate the total size of the new dimensions
    int newTotalSize = 1;
    for (int dim : newDimensions) {
        newTotalSize *= dim;
    }

    // Ensure the total size remains the same
    if (newTotalSize != size()) {
        throw std::invalid_argument("Total size of new dimensions must remain the same");
    }

    // Create a new tensor for the reshaped result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(newDimensions.size(), 0);
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = static_cast<int>(temp) % newDimensions[j];
            temp /= newDimensions[j];
        }

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    return result;

}

template<typename T>
Tensor<T> Tensor<T>::reshape(int newShape) const {
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
    std::vector<int> perm = permutation;

    // Check if the permutation is empty
    if (perm.empty()) {
        if (dimensions.size() == 2) {
            perm = {1, 0};
        } else {
            perm.resize(dimensions.size());
            std::iota(perm.rbegin(), perm.rend(), 0);
        }
    }

    if (perm.size() != dimensions.size()) {
        throw std::invalid_argument("Invalid permutation size");
    }

    // Validate permutation
    std::vector<int> check_perm(perm.size(), 0);
    for (const int& p : perm) {
        if (p < 0 || p >= perm.size()) {
            throw std::invalid_argument("Invalid permutation index");
        }
        check_perm[p]++;
    }

    for (const int& p : check_perm) {
        if (p != 1) {
            throw std::invalid_argument("Invalid permutation");
        }
    }

    // Create new dimensions on the permutation
    std::vector<int> newDimensions(dimensions.size());
    for (size_t i = 0; i < perm.size(); ++i) {
        newDimensions[i] = dimensions[perm[i]];
    }
    // Assuming transpose operation on data (not implemented here)
    // This is just a placeholder for actual transpose logic
    Tensor<T> transposedTensor(newDimensions);
    // Transpose data (requires actual implementation)

    return transposedTensor;
}

template<typename T>
Tensor<T> Tensor<T>::ones(const std::vector<int>& dims) {
    Tensor<T> tensor(dims);
    tensor.fill(T(1.0));
    return tensor;
}

// TODO: Implement the tril and triu
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

// //TODO: Implement the dot product
// template <typename T>
// Tensor<T> Tensor<T>::dot(Tensor<T>& other) {
//
// }

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    // Ensure that tensors have the same shape
    if (this->dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same shape for addition");
    }

    // Create a new tensor to hold the result
    Tensor<T> result(this->dimensions);

    // Perform element-wise addition
    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] + other.data[i];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    // Ensure dimensions match
    if (this->dimensions != other.dimensions) {
        throw std::invalid_argument("Dimension mismatch for subtraction");
    }

    // Create result tensor
    Tensor<T> result(this->dimensions);

    // Perform element-wise subtraction
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = this->data[i] - other.data[i];
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

template<typename T>
Tensor<T> Tensor<T>::operator[](const std::vector<int>& indices) {
    for (int i = 0; i < indices.size(); ++i){
        if (std::abs(indices[i]) >= dimensions[i]) {
            throw std::invalid_argument("Index out of bounds");
        }
    }

    /*
        * Explanation:
        *
        * This function is designed to index a multi-dimensional tensor using a vector of indices.
        *
        * Parameters:
        * - indices: A vector of integers specifying the indices along each dimension of the tensor.
        *            Indices can be positive or negative. Negative indices are converted to their
        *            corresponding positive indices by adding the dimension size.
        *
        * Internal Concepts:
        *
        * 1. newIndices:
        *    - Represents the indices for the new tensor with specified dimensions.
        *    - If the number of provided indices is less than the total number of dimensions,
        *      the remaining dimensions are assumed to include all possible indices.
        *    - The unspecified dimensions are filled with -1, indicating that the entire slice along
        *      those dimensions should be included in the resulting tensor.
        *
        * 2. coreDimensions:
        *    - These dimensions remain unchanged and are not directly indexed.
        *    - They represent the base dimensions of the tensor that are not affected by the
        *      current indexing operation.
        *
        * 3. targetDimensions:
        *    - These dimensions are affected by the indexing operation.
        *    - They correspond to the dimensions where indices are provided.
        *
        * Pseudocode Example:
        *
        * Given a tensor with dimensions [4, 3, 3], and indices [2, 1]:
        *
        * - If the indices are [2, 1], this means we are selecting a slice of the tensor where:
        *   - The first dimension is fixed at index 2.
        *   - The second dimension is fixed at index 1.
        *   - The resulting tensor has dimensions [3], corresponding to the last remaining dimension.
        *
        * - If the indices are not fully specified (e.g., [2]), the new tensor will include all
        *   indices along the unspecified dimensions.
        *
        * - Example Iteration:
        *   - For core dimension [0, 0, 0, 0] and target dimension [3, 3]:
        *     - core - [0, 0, 0, 0], target - [3, 3]
        *     - core - [0, 0, 0, 1], target - [3, 3]
        *     - core - [0, 0, 1, 0], target - [3, 3]
        *     - core - [1, 1, 1, 1], target - [3, 3]
        *   - This iteration explains how the tensor's core dimensions interact with the target dimensions
        *     and how the resulting size is computed.
        *
        * The final size of the resulting tensor should match the expected dimensions after
        * applying the indexing operation.
    */
    std::vector<int> newIndices;
    std::vector<int> coreDimensions(dimensions.size() - 2, 0);
    std::vector<int> targetDimensions = {dimensions[dimensions.size() - 2], dimensions[dimensions.size() - 1]};
    int linear_index = indices[0];

    if (indices.size() != dimensions.size()) {
        newIndices = std::vector<int>(dimensions.size(), -1);
        for (int i = 0; i < indices.size(); ++i) {
            newIndices[i] = indices[i];
        }
    }
    for (int i = 0; i < dimensions.size() - 1; ++i) {
        if (dimensions.size() - 2 >= i) {
            coreDimensions[i] = dimensions[i];
        }
    }

    for (int i = 1; i < dimensions.size(); ++i) {
        if (newIndices[i] == -1) {
            linear_index *= dimensions[i];
        } else {
            linear_index *= indices[i];
        }
    }

    std::vector<int> mama{};
    for (int i = 0; i < dimensions.size(); ++i) {
        if (i >= indices.size()) {
            mama.push_back(dimensions[i]);
        }
    }
    int dataSize = getTotalSize(mama);

    std::vector<int> newData(dataSize, T(0));
    for (int i = 0; i < dataSize; ++i) {
        newData[i] = data[i];
    }

    std::vector<int> resultDimensions{};
    for (size_t i = 0; i < dimensions.size(); ++i) {
        if (newIndices[i] == -1) {
            resultDimensions.push_back(dimensions[i]);
        }
    }

    Tensor<T> result(resultDimensions, newData);
    return result;
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

#endif // TENSOR_TPP