#include "../include/Tensor.h"

template<typename T>
template<typename D>
Tensor<T>::Tensor(const std::vector<int> &dims, const D& data) {
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

    int dims_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());

    if (!dims.empty()) {
        this->dimensions = dims;
        if (data.empty()) {
            for (int i = 0; i < dims_size; ++i) {
                this->data.push_back(T(0));
            }
            this->dimensions = dims;
        } else {
            std::vector<typename ExtractType<D>::Type> flattened_data;
            flatten(data, flattened_data);
            this->data = flattened_data;
            std::vector<int> computed_shape = compute_shape(data);
            if (dims_size != std::accumulate(computed_shape.begin(), computed_shape.end(), 1, std::multiplies<>())) {
                throw std::invalid_argument("Data size does not match the specified dimensions");
            }
            this->dimensions = dims;
        }
    } else if (!data.empty()) {
        std::vector<typename ExtractType<D>::Type> flattened_data;
        flatten(data, flattened_data);
        this->data = flattened_data;
        this->dimensions = compute_shape(data);
    }

    if (dims_size != this->data.size()) {
        throw std::invalid_argument("Data size does not match the specified dimensions");
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int> &dims) {
    /*
     * Header:
     * \brief Constructor for a tensor with specified dimensions
     *
     * Params:
     * \param dims The dimensions of the tensor
     *
     * Exceptions:
     * \throws std::invalid_argument If the number of indices does not match the number of dimensions
     *
    */
    int dims_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
    for (int i = 0; i < dims_size; ++i) {
        this->data.push_back(T(0));
    }
    this->dimensions = dims;
}

template<typename T>
template<typename D>
Tensor<T>::Tensor(const D& data) {
    /*
     * Header:
     * \brief Constructor for a tensor with data
     *
     * Params:
     * \param data The data to be stored in the tensor
     *
     * TParams:
     * \tparam D The type of the data
     *
     * Exceptions:
     * \throws std::invalid_argument If the data is not a vector
     * \throws std::invalid_argument If the number of indices does not match the number of dimensions
     *
    */
    if (!is_vector<D>::value) {
        throw std::invalid_argument("Data must be a vector");
    }

    std::vector<typename ExtractType<D>::Type> flattened_data;
    flatten(data, flattened_data);
    this->data = flattened_data;
    this->dimensions = compute_shape(data);
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
int Tensor<T>::calculateIndex(const std::vector<int>& indices) const {
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
Tensor<T> Tensor<T>::slice(int axis, int start, int end, int step) const {
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Calculate effective start, end and step
    int actual_start = (start < 0) ? (dimensions[axis] + start) : start;
    int actual_end = (end < 0) ? (dimensions[axis] + end) : end;
    int actual_step = (step == 0) ? 1 : step;

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

    return sliced_tensor;
}

template<typename T>
Tensor<T> Tensor<T>::slice(int axis, int start, int end) const {
    return slice(axis, start, end, 1);
}

template<typename T>
Tensor<T> Tensor<T>::slice(int axis, int start) const {
    return slice(axis, start, dimensions[axis], 1);
}

template<typename T>
Tensor<T> Tensor<T>::slice(int axis) const {
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

template<typename T>
Tensor<T> Tensor<T>::tril(const int& axis) const {
    std::vector<int> new_dims = dimensions;
    std::vector<T> result(data);

    int axis_to_use = (axis == -1) ? dimensions.size() - 2 : axis;

    for (int i = 0; i < dimensions[axis_to_use]; ++i) {
        for (int j = i + 1; j < dimensions[axis_to_use + 1]; ++j) {
            std::vector<int> idx1(dimensions.size(), 0);
            std::vector<int> idx2(dimensions.size(), 0);

            idx1[axis_to_use] = i;
            idx1[axis_to_use + 1] = j;
            idx2[axis_to_use] = j;
            idx2[axis_to_use + 1] = i;

            int idx1_flat = calculateIndex(idx1);
            int idx2_flat = calculateIndex(idx2);

            result[idx1_flat] = 0;
            result[idx2_flat] = 0;
        }
    }

    Tensor<T> triuTensor(new_dims);

    return Tensor<T>(new_dims, result);
}

template<typename T>
Tensor<T> Tensor<T>::triu(const int& axis) const {
    std::vector<int> new_dims = dimensions;
    std::vector<T> result(data);

    int axis_to_use = (axis == -1) ? dimensions.size() - 2 : axis;

    for (int i = 0; i < dimensions[axis_to_use]; ++i) {
        for (int j = 0; j < i; ++j) {
            std::vector<int> idx1(dimensions.size(), 0);
            std::vector<int> idx2(dimensions.size(), 0);

            idx1[axis_to_use] = i;
            idx1[axis_to_use + 1] = j;
            idx2[axis_to_use] = j;
            idx2[axis_to_use + 1] = i;

            int idx1_flat = calculateIndex(idx1);
            int idx2_flat = calculateIndex(idx2);

            result[idx1_flat] = 0;
            result[idx2_flat] = 0;
        }
    }

    return Tensor<T>(new_dims, result);
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) const {
    // Implementation methode operator+ for other
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
    // Implementation methode operator- for other
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
    // Implementation methode operator* for other
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
    // Implementation methode operator/ for other
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
    // Implementation methode operator+ for scalar
}

template<typename T>
Tensor<T> Tensor<T>::operator-(T scalar) const {
    // Implementation methode operator- for scalar
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T scalar) const {
    // Implementation methode operator* for scalar
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T scalar) const {
    // Implementation methode operator/ for scalar
}

template<typename T>
Tensor<T> Tensor<T>::operator[](int index) const {
    // Implementation methode operator[] for index
}

template<typename T>
Tensor<T> Tensor<T>::operator[](const std::vector<int>& indices) const {
    // Implementation methode operator[] for indices
}

