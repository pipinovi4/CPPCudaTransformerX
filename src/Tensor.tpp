#ifndef TENSOR_TPP
#define TENSOR_TPP

#include "../include/Tensor.h"

template<typename T>
template<typename D>
Tensor<T>::Tensor(const std::vector<int>& dims, const D& data) {
    if (!is_vector<D>::value) { // Ensure that the provided data is a vector
        throw std::invalid_argument("Data must be a vector");
    }

    int dims_size = getTotalSize(dims); // Calculate the total size based on the dimensions

    if (!dims.empty()) { // Check if dimensions are specified
        this->dimensions = dims; // Set the dimensions
        std::vector<int> computed_shape = compute_shape(data); // Compute the shape of the input data

        if (data.empty()) { // If data is empty, initialize with zeros
            this->data.resize(dims_size, T(0));
        } else {
            if (dims_size != std::accumulate(computed_shape.begin(), computed_shape.end(), 1, std::multiplies<>())) {
                // Validate that the data size matches the specified dimensions
                throw std::invalid_argument("Data size does not match the specified dimensions");
            }
            std::vector<typename ExtractType<D>::Type> flattened_data; // Flatten the data
            flatten(data, flattened_data); // Flatten the input data recursively
            this->data = flattened_data; // Assign the flattened data to the tensor
            this->dimensions = dims; // Set the dimensions
        }
    } else if (!data.empty()) { // If dimensions are not specified but data is provided
        std::vector<typename ExtractType<D>::Type> flattened_data;
        flatten(data, flattened_data); // Flatten the data
        this->data = flattened_data; // Assign the flattened data to the tensor
        this->dimensions = dims; // Set the dimensions
    }

    this->strides = calculateStrides(); // Calculate strides for efficient indexing
}

template<typename T>
Tensor<T>::Tensor(const std::vector<int>& dims) : Tensor<T>(dims, std::vector<T>()) {
    int dims_size = getTotalSize(dims); // Calculate the total size based on the dimensions
    this->dimensions = dims; // Set the dimensions
    this->data.resize(dims_size, T(0)); // Initialize the data with zeros

    if (dims_size != this->data.size()) { // Validate the size of the data
        throw std::invalid_argument("Data size does not match the specified dimensions");
    }

    this->strides = calculateStrides(); // Calculate strides for efficient indexing
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<int> dims) : Tensor<T>(std::vector<int>(dims), std::vector<T>()) {
    int dims_size = getTotalSize(dims); // Calculate the total size based on the dimensions
    std::vector<int> dims_vec(dims); // Convert the initializer list to a vector
    this->data.resize(dims_size, T(0)); // Initialize the data with zeros

    if (dims_size != this->data.size()) { // Validate the size of the data
        throw std::invalid_argument("Data size does not match the specified dimensions");
    }

    this->strides = calculateStrides(); // Calculate strides for efficient indexing
}

template<typename T>
template<typename D>
Tensor<T>::Tensor(const D &data) : Tensor<T>(std::vector<int>(), data) {
    std::vector<int> computed_shape = compute_shape(data); // Compute the shape of the input data
    if (!is_vector<D>::value) { // Ensure that the provided data is a vector
        throw std::invalid_argument("Data must be a vector");
    }

    std::vector<typename ExtractType<D>::Type> flattened_data; // Flatten the data
    flatten(data, flattened_data); // Flatten the input data recursively
    this->data = flattened_data; // Assign the flattened data to the tensor
    this->dimensions = compute_shape(data); // Set the dimensions based on the data shape
    this->strides = calculateStrides(); // Calculate strides for efficient indexing
}

template <typename T>
Tensor<T>::Tensor(const std::vector<int>& dims, const int& newSize) : dimensions(dims){
    this->data.reserve(newSize); // Reserve space for the data with the given size
    this->strides = calculateStrides(); // Calculate strides for efficient indexing
}

template<typename T>
const std::vector<int>& Tensor<T>::shape() const {
    return dimensions; // Return the shape of the tensor as a vector of dimensions
}

template<typename T>
int Tensor<T>::size() const {
    return data.size(); // Return the total number of elements in the tensor
}

template<typename T>
void Tensor<T>::set(const std::vector<int>& indices, T value) {
    int index = calculateIndex(indices); // Calculate the flattened index based on the provided indices
    data[index] = value; // Set the value at the calculated index
}

template<typename T>
T Tensor<T>::get(const std::vector<int>& indices) const {
    int index = calculateIndex(indices); // Calculate the flattened index based on the provided indices
    return data[index]; // Return the value at the calculated index
}

template<typename T>
void Tensor<T>::print() const {
    if (dimensions.empty()) { // Check if the tensor is empty (no dimensions)
        std::cout << "[]" << std::endl; // Print an empty tensor representation
        return;
    }

    std::vector<size_t> indices(dimensions.size(), 0); // Initialize indices vector for tracking positions
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices for each dimension
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices[j] = temp % dimensions[j]; // Calculate the index for dimension j
            temp /= dimensions[j]; // Update temp for the next dimension
        }

        // Print opening brackets for nested dimensions
        if (indices.back() == 0) {
            for (size_t j = dimensions.size() - 1; j < dimensions.size(); --j) {
                if (indices[j] == 0) {
                    std::cout << "[";
                }
            }
        }

        // Print the data value at the current index
        std::cout << data[i];

        // Close brackets for nested dimensions
        if (indices.back() == dimensions.back() - 1) {
            for (size_t j = dimensions.size() - 1; j < dimensions.size(); --j) {
                if (indices[j] == dimensions[j] - 1) {
                    std::cout << "]";
                }
            }
        } else {
            std::cout << ", "; // Separate elements with a comma
        }

        // Print a newline after each row
        if (indices.back() == dimensions.back() - 1 && i != data.size() - 1) {
            std::cout << std::endl;
        }
    }
    std::cout << "\n\n"; // Extra newline for formatting
    std::cout << "Shape: "; // Print the shape of the tensor
    for (size_t dim = 0; dim < dimensions.size(); ++dim) {
        if (dim == 0) {
            std::cout << "("; // Opening parenthesis for shape
        }
        std::cout << dimensions[dim]; // Print each dimension
        if (dim < dimensions.size() - 1) {
            std::cout << ", "; // Separate dimensions with a comma
        } else {
            std::cout << ")"; // Closing parenthesis for shape
        }
    }
    std::cout << " | "; // Separator for additional information
    std::cout << "Size: " << data.size() << " | "; // Print the total size of the tensor
    std::cout << "Dtype: "; // Print the data type of the tensor

    // Determine and print the data type
    if (std::is_same<T, int>::value) {
        std::cout << "int";
    } else if (std::is_same<T, float>::value) {
        std::cout << "float";
    } else if (std::is_same<T, double>::value) {
        std::cout << "double";
    } else {
        std::cout << typeid(T).name(); // Print the type name if it's not a common type
    }

    std::cout << "\n" << std::endl; // Final newline for formatting
}

template<typename T>
void Tensor<T>::fill(T value) {
    for (T& element : data) { // Iterate over all elements in the tensor
        element = value; // Set each element to the specified value
    }
}

template<typename T>
Tensor<T> Tensor<T>::sqrt() {
    Tensor<T> result(dimensions); // Create a new tensor with the same dimensions
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = std::sqrt(data[i]); // Apply the square root function to each element
    }
    return result; // Return the resulting tensor
}

template <typename T>
Tensor<T> Tensor<T>::pow(T exponent) const {
    // Create a result tensor with the same dimensions
    Tensor<T> result(dimensions);

    // Use direct pointer access for optimization
    const T* src_data = data.data(); // Pointer to the source data
    T* dest_data = result.data.data(); // Pointer to the destination data

    // Perform element-wise exponentiation
    for (size_t i = 0; i < data.size(); ++i) {
        dest_data[i] = std::pow(src_data[i], exponent); // Raise each element to the specified power
    }

    return result; // Return the resulting tensor
}

template <typename T>
Tensor<T> Tensor<T>::apply(std::function<T(T)> func) const {
    Tensor<T> result(dimensions); // Create a new tensor with the same dimensions
    for (int i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]); // Apply the provided function to each element
    }
    return result; // Return the resulting tensor
}

template <typename T>
void Tensor<T>::add(const Tensor<T>& other) {
    // Check if dimensions match for element-wise addition
    if (dimensions == other.dimensions) {
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] += other.data[i]; // Add corresponding elements
        }
    } else {
        // Handle broadcasting manually for dimensions that don't match
        for (int i = 0; i < dimensions[0]; ++i) {
            for (int j = 0; j < dimensions[1]; ++j) {
                data[i * dimensions[1] + j] += other.data[j]; // Broadcast and add elements
            }
        }
    }
}

template<typename T>
Tensor<T> Tensor<T>::sum(int axis) const {
    if (axis == -1) {
        axis = dimensions.size() - 1; // Default to the last axis if -1 is specified
    }

    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis"); // Check for valid axis
    }

    std::vector<int> new_dims = dimensions; // Copy the current dimensions
    new_dims.erase(new_dims.begin() + axis); // Remove the specified axis

    Tensor<T> result(new_dims); // Create a new tensor with the reduced dimensions

    const  std::vector<int> strides = calculateStrides(); // Precompute strides for the original tensor
    const std::vector<int> result_strides = result.calculateStrides(); // Precompute strides for the result tensor

#pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        int temp = static_cast<int>(i);
        int result_index = 0;
        for (size_t j = 0; j < dimensions.size(); ++j) {
            const int index = temp / strides[j];
            temp %= strides[j];
            if (j < axis) {
                result_index += index * result_strides[j];
            } else if (j > axis) {
                result_index += index * result_strides[j - 1];
            }
        }
#pragma omp atomic
        result.data[result_index] += data[i]; // Accumulate the sum along the specified axis
    }

    return result; // Return the resulting tensor
}

template <typename T>
Tensor<T> Tensor<T>::mean(int axis) const {
    if (axis == -1) {
        axis = dimensions.size() - 1; // Default to the last axis if -1 is specified
    }
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis"); // Check for valid axis
    }
    Tensor<T> sum_tensor = sum(axis); // Sum the tensor along the specified axis
    return sum_tensor / dimensions[axis]; // Divide by the size of the axis to compute the mean
}

template <typename T>
Tensor<T> Tensor<T>::argmax(int axis) const {
    // Get the shape of the tensor
    const std::vector<int> shape = this->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Cannot perform argmax on an empty tensor.");
    }

    if (axis < 0) {
        axis += static_cast<int>(shape.size());
    }

    if (axis >= static_cast<int>(shape.size()) || axis < 0) {
        throw std::invalid_argument("Axis out of range for tensor dimensions.");
    }

    // Create a vector to hold the shape of the output tensor
    std::vector<int> output_shape = shape;

    // Remove the specified axis from the output shape
    if (shape.size() > 1) {
        output_shape.erase(output_shape.begin() + axis);
    } else {
        output_shape[axis] = 1;  // Set the output shape to 1 if the tensor is 1D
    }

    // Calculate the total number of elements in the tensor
    const int total_elements = this->size();
    const int axis_dim = shape[axis];
    const int num_iterations = total_elements / axis_dim;

    // Initialize a vector to hold the result indices
    std::vector<T> result_indices(num_iterations);

    // Iterate over the tensor to find the indices of the maximum values along the specified axis
    #pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        T max_value = std::numeric_limits<T>::lowest();
        int max_index = 0;

        for (int j = 0; j < axis_dim; ++j) {
            int index = i * axis_dim + j;
            if (this->data[index] > max_value) {
                max_value = this->data[index];
                max_index = j;
            }
        }

        result_indices[i] = max_index;
    }

    // Create a new tensor to store the result and return it
    return Tensor<T>(output_shape, result_indices);
}

template <typename T>
Tensor<T> Tensor<T>::softmax(int axis) const {
    // Get the shape of the tensor
    const std::vector<int> shape = this->shape();

    if (shape.empty()) {
        throw std::invalid_argument("Cannot perform softmax on an empty tensor.");
    }

    // Adjust axis if it's negative (e.g., -1 refers to the last axis)
    if (axis < 0) {
        axis += static_cast<int>(shape.size());
    }

    // Ensure the axis is within the valid range
    if (axis >= static_cast<int>(shape.size()) || axis < 0) {
        throw std::invalid_argument("Axis out of range for tensor dimensions.");
    }

    // Create a tensor to hold the result
    Tensor<T> result = *this;

    // Calculate the total number of elements and the size of the axis dimension
    const int total_elements = this->size();
    const int axis_dim = shape[axis];
    const int num_iterations = total_elements / axis_dim;

    // Iterate over the tensor to apply the Softmax function along the specified axis
    #pragma omp parallel for
    for (int i = 0; i < num_iterations; ++i) {
        // Calculate the start and end indices for this slice
        const int start_idx = i * axis_dim;
        const int end_idx = start_idx + axis_dim;

        // Find the maximum value in the current slice for numerical stability
        T max_value = std::numeric_limits<T>::lowest();
        for (int j = start_idx; j < end_idx; ++j) {
            if (this->data[j] > max_value) {
                max_value = this->data[j];
            }
        }

        // Compute the exponentials and their sum
        T sum = 0;
        for (int j = start_idx; j < end_idx; ++j) {
            result.data[j] = std::exp(this->data[j] - max_value);
            sum += result.data[j];
        }

        // Normalize the values to get the Softmax probabilities
        for (int j = start_idx; j < end_idx; ++j) {
            result.data[j] /= sum;
        }
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start, const int end, const int step) const {
    if (axis < 0 || axis >= dimensions.size()) { // Check if the axis is valid
        throw std::invalid_argument("Invalid axis");
    }

    // Calculate effective start, end, and step values
    const int actual_start = (start < 0) ? (dimensions[axis] + start) : start;
    const int actual_end = (end < 0) ? (dimensions[axis] + end) : end;
    const int actual_step = (step == 0) ? 1 : step;

    // Validate the start index
    if (actual_start < 0 || actual_start >= dimensions[axis]) {
        throw std::invalid_argument("Invalid start index");
    }
    // Validate the end index
    if (actual_end < 0 || actual_end > dimensions[axis]) {
        throw std::invalid_argument("Invalid end index");
    }

    // Compute new dimensions for the sliced tensor
    std::vector<int> new_dims = dimensions;
    new_dims[axis] = (actual_end - actual_start + actual_step - 1) / actual_step;

    // Create a new tensor with the computed dimensions
    Tensor<T> sliced_tensor(new_dims);

    // Initialize indices for the original and sliced tensors
    std::vector<int> indices(dimensions.size(), 0);
    std::vector<int> sliced_indices(dimensions.size(), 0);

    // Access data pointers for faster access
    auto* data_access = data.data();
    auto* indices_data_access = indices.data();
    auto* sliced_indices_data_access = sliced_indices.data();
    auto* dimensions_data_access = dimensions.data();
    auto* sliced_tensor_data_access = sliced_tensor.data.data();

    // Iterate over the data and copy the sliced data
    for (int i = 0; i < data.size(); ++i) {
        // Calculate current indices in the original tensor
        int temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices_data_access[j] = temp % dimensions_data_access[j];
            temp /= dimensions_data_access[j];
        }

        // Check if the current index falls within the slice range along the specified axis
        if (indices_data_access[axis] >= actual_start && indices_data_access[axis] < actual_end && (indices_data_access[axis] - actual_start) % actual_step == 0) {
            // Calculate the corresponding index in the sliced tensor
            for (size_t j = 0; j < indices.size(); ++j) {
                sliced_indices_data_access[j] = (j == axis) ? (indices_data_access[j] - actual_start) / actual_step : indices_data_access[j];
            }

            // Calculate the index in the sliced tensor
            int sliced_index = sliced_tensor.calculateIndex(sliced_indices);

            // Copy the data from the original tensor to the sliced tensor
            sliced_tensor_data_access[sliced_index] = data_access[i];
        }
    }

    // Handle squeezing of dimensions if necessary
    for (const int& dim : new_dims) {
        if (dim < 1) {
            return sliced_tensor.squeeze(); // Squeeze the tensor if any dimension is less than 1
        }
    }

    return sliced_tensor; // Return the sliced tensor
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start, const int end) const {
    return slice(axis, start, end, 1); // Call the full slice method with a default step of 1
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis, const int start) const {
    return slice(axis, start, dimensions[axis], 1); // Slice from start to the end of the dimension with step 1
}

template<typename T>
Tensor<T> Tensor<T>::slice(const int axis) const {
    return slice(axis, 0, dimensions[axis], 1); // Slice the entire axis with a default step of 1
}

template<typename T>
Tensor<T> Tensor<T>::concatenate(const Tensor<T>& other, int axis) const {
    // Ensure axis is valid
    if (axis < 0 || axis >= dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Check dimensions for concatenation along the specified axis
    if (dimensions[axis] != other.shape()[axis]) {
        throw std::invalid_argument("Dimensions mismatch along concatenation axis");
    }

    // Determine new dimensions for the concatenated tensor
    std::vector<int> newDimensions = dimensions;
    newDimensions[axis] += other.shape()[axis];

    // Create a new tensor for the concatenated result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(dimensions.size(), 0);
    auto indices_data_access = indices.data();
    for (size_t i = 0; i < data.size(); ++i) {
        // Calculate current indices for each element
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices_data_access[j] = temp % dimensions[j];
            temp /= dimensions[j];
        }

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    // Copy data from the other tensor
    indices.assign(dimensions.size(), 0);
    indices_data_access = indices.data();
    for (size_t i = 0; i < other.size(); ++i) {
        // Calculate current indices for the other tensor
        size_t temp = i;
        for (size_t j = indices.size(); j-- > 0;) {
            indices_data_access[j] = temp % other.shape()[j];
            temp /= other.shape()[j];
        }
        // Retrieve the value from the other tensor
        T value = other.get(indices);

        // Adjust the axis index for concatenation
        indices_data_access[axis] += other.shape()[axis];

        // Copy data into the result tensor
        result.set(indices, value);
    }

    return result; // Return the concatenated tensor
}

template<typename T>
Tensor<T> Tensor<T>::concatenate(const Tensor<T> &other) const {
    return concatenate(other, 0); // Concatenate along the first axis by default
}

template<typename T>
Tensor<T> Tensor<T>::expandDims(int axis) const {
    if (axis == 1) axis = dimensions.size() + 1; // Adjust axis if specified as 1
    // Ensure axis is valid
    if (axis < 0 || axis > dimensions.size()) {
        throw std::invalid_argument("Invalid axis");
    }

    // Create new dimensions for the expanded tensor
    std::vector<int> newDimensions = dimensions;
    newDimensions.insert(newDimensions.begin() + axis, 1); // Insert a new dimension of size 1

    // Create a new tensor for the expanded result
    Tensor<T> result(newDimensions);

    // Copy data from the current tensor
    std::vector<int> indices(dimensions.size() + 1, 0);  // Increase indices size for new axis
    const auto indices_data_access = indices.data();
    const auto dimensions_data_access = dimensions.data();
    for (int i = 0; i < data.size(); ++i) {
        // Calculate current indices
        int temp = i;
        for (size_t j = 0; j < indices.size(); ++j) {
            if (j < axis) {
                indices_data_access[j] = temp % dimensions_data_access[j];
                temp /= dimensions_data_access[j];
            } else if (j > axis) {
                indices_data_access[j] = temp % dimensions_data_access[j - 1];
                temp /= dimensions_data_access[j - 1];
            }
        }

        // Set the new axis index to 0
        indices[axis] = 0;

        // Copy data into the result tensor
        result.set(indices, data[i]);
    }

    return result; // Return the expanded tensor
}

template <typename T>
Tensor<T> Tensor<T>::expandDimsAs(const std::vector<int>& other_dimensions) const {
    Tensor<T> expanded_tensor(other_dimensions); // Create a tensor with the target shape

    // Assuming data is stored in a flat vector
    const T* src_data = data.data(); // Pointer to the source data
    T* dest_data = expanded_tensor.data.data(); // Pointer to the destination data

    // Expand the tensor data to match the target shape
    for (size_t i = 0; i < expanded_tensor.data.size(); ++i) {
        dest_data[i] = src_data[i % data.size()]; // Copy and wrap around data if necessary
    }

    return expanded_tensor; // Return the expanded tensor
}

template<typename T>
Tensor<T> Tensor<T>::squeeze() const {
    // Calculate the number of dimensions to be removed and the new dimensions
    std::vector<int> newDimensions;
    newDimensions.reserve(dimensions.size());

    // Add non-unit dimensions to the new shape
    for (int dimension : dimensions) {
        if (dimension != 1) {
            newDimensions.emplace_back(dimension);
        }
    }

    // Create a new tensor for the squeezed result
    Tensor<T> result(newDimensions);

    // Copy data directly if no dimensions were removed
    if (newDimensions.size() == dimensions.size()) {
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

    // Copy data efficiently using strides
    auto result_data_access = result.data.data();

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
        result_data_access[newIndex] = data[i];
    }

    return result; // Return the squeezed tensor
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& newShape) const {
    // Calculate the total size of the new dimensions
    int newTotalSize = 1;
    for (const int dim : newShape) {
        newTotalSize *= dim;
    }

    // Create a new tensor for the reshaped result
    Tensor<T> result(newShape);

    // Copy data from the current tensor
    result.data = data;

    return result; // Return the reshaped tensor
}

template<typename T>
Tensor<T> Tensor<T>::reshape(const int newShape) const {
    return reshape(std::vector<int>{newShape}); // Reshape to a single dimension
}

template<typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<int> &dims) {
    Tensor<T> tensor(dims); // Create a tensor with the specified dimensions
    tensor.fill(T(0)); // Fill the tensor with zeros
    return tensor; // Return the zero-filled tensor
}

template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<int>& permutation) const {
    // Validate the permutation vector
    if (permutation.size() != dimensions.size()) {
        throw std::invalid_argument("Permutation size does not match tensor dimensions");
    }

    for (const int i : permutation) {
        if (i < 0 || i >= static_cast<int>(dimensions.size())) {
            throw std::invalid_argument("Invalid permutation index");
        }
    }

    // Calculate the new shape based on the permutation
    std::vector<int> newShape(dimensions.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        newShape[i] = dimensions[permutation[i]];
    }

    // Calculate the original strides
    std::vector<int> originalStrides(dimensions.size(), 1);
    for (int i = dimensions.size() - 2; i >= 0; --i) {
        originalStrides[i] = originalStrides[i + 1] * dimensions[i + 1];
    }

    // Calculate the new strides based on the permutation
    std::vector<int> newStrides(dimensions.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
        newStrides[i] = originalStrides[permutation[i]];
    }

    // Create a new tensor with the new shape
    Tensor<T> result(newShape);

    // Map the data from the original tensor to the new tensor
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        int oldIndex = 0;
        int newIndex = 0;
        int temp = static_cast<int>(i);

        for (int j = dimensions.size() - 1; j >= 0; --j) {
            int index = temp % dimensions[j];
            temp /= dimensions[j];

            oldIndex += index * originalStrides[j];
            newIndex += index * newStrides[permutation[j]];
        }

        result.data[newIndex] = data[oldIndex];
    }

    return result;
}


template<typename T>
Tensor<T> Tensor<T>::ones(const std::vector<int>& dims) {
    Tensor<T> tensor(dims); // Create a tensor with the specified dimensions
    tensor.fill(T(1.0)); // Fill the tensor with ones
    return tensor; // Return the one-filled tensor
}

template<typename T>
Tensor<T> Tensor<T>::uniform(const std::vector<int>& dims, T lower, T upper) {
    Tensor<T> result(dims); // Create a tensor with the specified dimensions
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_floating_point<T>::value) { // Check if T is a floating-point type
        std::uniform_real_distribution<T> dis(lower, upper); // Create a distribution for floating-point numbers
        for (T& element : result.data) {
            element = dis(gen); // Fill the tensor with random floating-point numbers
        }
    } else if constexpr (std::is_integral<T>::value) { // Check if T is an integral type
        std::uniform_int_distribution<T> dis(lower, upper); // Create a distribution for integers
        for (T& element : result.data) {
            element = dis(gen); // Fill the tensor with random integers
        }
    } else {
        throw std::invalid_argument("Unsupported type for uniform distribution"); // Throw an exception for unsupported types
    }

    return result; // Return the tensor filled with uniformly distributed random values
}

template<typename T>
Tensor<T> Tensor<T>::tril(const int& axis) {
    const int dimSize = dimensions.size();
    if (dimSize < 2) { // Ensure the tensor has at least 2 dimensions
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
    if (dimSize < 2) { // Ensure the tensor has at least 2 dimensions
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

    // Ensure the tensors have compatible dimensions for dot product
    if (this_dims.empty() || other_dims.empty()) {
        throw std::invalid_argument("Tensors must have at least one dimension for dot product");
    }

    // Ensure the inner dimensions match for matrix multiplication
    if (this_dims.back() != other_dims[other_dims.size() - 2]) {
        throw std::invalid_argument("Inner dimensions do not match for dot product");
    }

    // Handle broadcasting dimensions
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Compute result dimensions
    std::vector<int> resultDimensions;
    for (int i = 0; i < this_dims.size() - 2; ++i) {
        resultDimensions.emplace_back(this_dims[i]);
    }
    resultDimensions.emplace_back(this_dims[this_dims.size() - 2]);
    resultDimensions.emplace_back(other_dims.back());

    // Initialize result tensor (or resize if needed)
    Tensor<T> result(resultDimensions);

    const int M = this_dims[this_dims.size() - 2]; // Outer dimension for the first tensor
    const int K = this_dims.back();                // Inner dimension for both tensors
    const int N = other_dims.back();               // Outer dimension for the second tensor

    // Access data pointers for faster access
    T* result_data = result.data.data();
    const T* A = data.data();
    const T* B = other.data.data();

    // Precompute batch size
    const int batch_size = result.data.size() / (M * N);

    // Use Eigen to perform the matrix multiplication
    #pragma omp parallel for // Parallelize over batches
    for (int b = 0; b < batch_size; ++b) {
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matA(A + b * M * K, M, K);
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matB(B + b * K * N, K, N);
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> matC(result_data + b * M * N, M, N);

        matC.noalias() = matA * matB;  // Perform matrix multiplication using Eigen
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Handle broadcasting for mismatched dimension sizes
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Check compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for addition");
        }
    }

    // Calculate resulting dimensions after broadcasting
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create result tensor
    Tensor<T> result(result_dims);

    // Perform the addition with broadcasting manually
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matA(this->data.data(), this->data.size(), 1);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matB(other.data.data(), other.data.size(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matC(result.data.data(), result.data.size(), 1);

    if (this->data.size() == other.data.size()) {
        // If both tensors have the same size, perform element-wise addition
        matC = matA.array() + matB.array();
    } else {
        // Manually broadcast one of the tensors
        for (int i = 0; i < matC.size(); ++i) {
            int this_idx = i % matA.size();
            int other_idx = i % matB.size();
            matC(i) = matA(this_idx) + matB(other_idx);
        }
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Handle broadcasting for mismatched dimension sizes
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Check compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for subtraction");
        }
    }

    // Calculate resulting dimensions after broadcasting
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create result tensor
    Tensor<T> result(result_dims);

    // Perform the subtraction with broadcasting manually
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matA(this->data.data(), this->data.size(), 1);
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matB(other.data.data(), other.data.size(), 1);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matC(result.data.data(), result.data.size(), 1);

    if (this->data.size() == other.data.size()) {
        // If both tensors have the same size, perform element-wise subtraction
        matC = matA.array() - matB.array();
    } else {
        // Manually broadcast one of the tensors
        for (int i = 0; i < matC.size(); ++i) {
            int this_idx = i % matA.size();
            int other_idx = i % matB.size();
            matC(i) = matA(this_idx) - matB(other_idx);
        }
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    // Determine broadcast-compatible dimensions
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Handle broadcasting for mismatched dimension sizes
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Check compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for multiplication");
        }
    }

    // Calculate resulting dimensions
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create a new tensor to hold the result
    Tensor<T> result(result_dims);

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform element-wise multiplication with broadcasting
    #pragma omp parallel for
    for (size_t i = 0; i < result.data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = result_dims.size(); j-- > 0;) {
            int result_idx = static_cast<int>(temp) % result_dims[j];
            temp /= result_dims[j];

            this_index += (this_dims[j] == 1 ? 0 : result_idx) * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        result.data[i] = this->data[this_index] * other.data[other_index];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
    // Determine broadcast-compatible dimensions
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Handle broadcasting for mismatched dimension sizes
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Check compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for division");
        }
    }

    // Calculate resulting dimensions
    std::vector<int> result_dims(this_dims.size());
    for (size_t i = 0; i < this_dims.size(); ++i) {
        result_dims[i] = std::max(this_dims[i], other_dims[i]);
    }

    // Create a new tensor to hold the result
    Tensor<T> result(result_dims);

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform element-wise division with broadcasting
    #pragma omp parallel for
    for (size_t i = 0; i < result.data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = result_dims.size(); j-- > 0;) {
            int result_idx = static_cast<int>(temp) % result_dims[j];
            temp /= result_dims[j];

            this_index += (this_dims[j] == 1 ? 0 : result_idx) * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        result.data[i] = this->data[this_index] / other.data[other_index];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::operator+(T scalar) const {
    // Create a new tensor for the result
    Tensor<T> result(dimensions);

    // Perform element-wise addition with scalar
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
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
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] / scalar;
    }

    return result;
}

template<typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other) {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Adjust dimensions for broadcasting
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Verify compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for addition");
        }
    }

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform in-place element-wise addition with broadcasting using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = this_dims.size(); j-- > 0;) {
            int result_idx = temp % this_dims[j];
            temp /= this_dims[j];

            this_index += result_idx * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        this->data[this_index] += other.data[other_index];
    }

    return *this;
}

template<typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other) {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Adjust dimensions for broadcasting
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Verify compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for subtraction");
        }
    }

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform in-place element-wise subtraction with broadcasting using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = this_dims.size(); j-- > 0;) {
            int result_idx = temp % this_dims[j];
            temp /= this_dims[j];

            this_index += result_idx * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        this->data[this_index] -= other.data[other_index];
    }

    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other) {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Adjust dimensions for broadcasting
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Verify compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for multiplication");
        }
    }

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform in-place element-wise multiplication with broadcasting using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = this_dims.size(); j-- > 0;) {
            int result_idx = static_cast<int>(temp) % this_dims[j];
            temp /= this_dims[j];

            this_index += (this_dims[j] == 1 ? 0 : result_idx) * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        this->data[this_index] *= other.data[other_index];
    }

    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other) {
    // Ensure dimensions are broadcast-compatible
    auto this_dims = this->dimensions;
    auto other_dims = other.dimensions;

    // Adjust dimensions for broadcasting
    if (this_dims.size() < other_dims.size()) {
        this_dims.insert(this_dims.begin(), other_dims.size() - this_dims.size(), 1);
    } else if (other_dims.size() < this_dims.size()) {
        other_dims.insert(other_dims.begin(), this_dims.size() - other_dims.size(), 1);
    }

    // Verify compatibility for broadcasting
    for (size_t i = 0; i < this_dims.size(); ++i) {
        if (this_dims[i] != other_dims[i] && this_dims[i] != 1 && other_dims[i] != 1) {
            throw std::invalid_argument("Tensors are not broadcastable for division");
        }
    }

    // Precompute strides for efficient indexing
    std::vector<int> this_strides(this_dims.size(), 1);
    std::vector<int> other_strides(other_dims.size(), 1);

    for (int i = this_dims.size() - 2; i >= 0; --i) {
        this_strides[i] = this_strides[i + 1] * this_dims[i + 1];
        other_strides[i] = other_strides[i + 1] * other_dims[i + 1];
    }

    // Perform in-place element-wise division with broadcasting using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        int this_index = 0;
        int other_index = 0;
        size_t temp = i;

        // Calculate index in each tensor based on broadcasted dimensions
        for (size_t j = this_dims.size(); j-- > 0;) {
            int result_idx = static_cast<int>(temp) % this_dims[j];
            temp /= this_dims[j];

            this_index += (this_dims[j] == 1 ? 0 : result_idx) * this_strides[j];
            other_index += (other_dims[j] == 1 ? 0 : result_idx) * other_strides[j];
        }

        this->data[this_index] /= other.data[other_index];
    }

    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const T& scalar) {
    // Perform in-place addition with a scalar using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] += scalar;
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const T& scalar) {
    // Perform in-place subtraction with a scalar using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] -= scalar;
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const T& scalar) {
    // Perform in-place multiplication with a scalar using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] *= scalar;
    }
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const T& scalar) {
    // Perform in-place division with a scalar using parallel processing
    #pragma omp parallel for
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] /= scalar;
    }
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
T& Tensor<T>::operator()(int index) {
    // Convert a single index into a flat index and return a reference to the corresponding data
    int flatIndex = toFlatIndex({index});
    return data[flatIndex];
}

template<typename T>
T& Tensor<T>::operator()(const std::vector<int>& indices) {
    // Convert multi-dimensional indices into a flat index and return a reference to the corresponding data
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
    // Check if tensors are not equal by leveraging the equality operator
    return !(*this == other);
}

template <typename T>
void Tensor<T>::serialize(std::ostream& os) const {
    // Write the shape of the tensor to the output stream
    os << "{";
    for (size_t i = 0; i < dimensions.size(); ++i) {
        os << dimensions[i];
        if (i < dimensions.size() - 1) {
            os << ", ";
        }
    }
    os << "}, ";

    // Write the data of the tensor to the output stream
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
    // Read the shape of the tensor from the input stream
    is >> ch; // Read '{'
    dimensions.clear();
    int dim;
    while (is >> dim) {
        dimensions.push_back(dim);
        is >> ch; // Read ',' or '}'
        if (ch == '}') break;
    }
    is >> ch; // Read ','

    // Read the data of the tensor from the input stream
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

// Private methods

template <typename T>
int Tensor<T>::calculateIndex(const std::vector<int>& indices) const {
    int index = 0; // Initialize the index accumulator
    const int dimensions_size = dimensions.size(); // Get the number of dimensions
    for (int i = 0; i < dimensions_size; ++i) {
        index += indices[i] * strides[i]; // Calculate the index using strides
    }
    return index; // Return the computed flat index
}

template <typename T>
std::vector<int> Tensor<T>::calculateStrides() const {
    std::vector<int> localStrides(dimensions.size()); // Initialize the strides vector
    int stride = 1; // Start with a stride of 1 (last dimension)
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        localStrides[i] = stride; // Set the stride for the current dimension
        stride *= dimensions[i]; // Update the stride for the next dimension
    }
    return localStrides; // Return the computed strides
}

template <typename T>
int Tensor<T>::getTotalSize(const std::vector<int>& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()); // Multiply all dimensions to get total size
}

template <typename T>
template <typename D>
void Tensor<T>::flatten(const D& vec, std::vector<T>& result) {
    if constexpr (is_vector<D>::value) {
        for (const auto& elem : vec) {
            flatten(elem, result); // Recursively flatten nested vectors
        }
    } else {
        result.push_back(vec); // Add the scalar value to the result
    }
}

template <typename T>
template <typename D>
std::vector<int> Tensor<T>::compute_shape(const D& vec) {
    if constexpr (is_vector<D>::value) {
        if (vec.empty()) {
            return {0}; // Handle empty vector case
        }
        std::vector<int> shape;
        shape.push_back(static_cast<int>(vec.size())); // Record the size of the current dimension
        auto inner_shape = compute_shape(vec[0]); // Recursively compute the inner dimensions
        shape.insert(shape.end(), inner_shape.begin(), inner_shape.end()); // Combine with the inner dimensions' shape
        return shape; // Return the computed shape
    } else {
        return {}; // Base case: return an empty shape for non-vector type
    }
}

template <typename T>
std::vector<int> Tensor<T>::combineIndices(const std::vector<int>& this_indices,
    const std::vector<int>& other_indices, const int this_rank, const int other_rank) {
    std::vector<int> result_indices(this_rank + (other_rank - 1), 0); // Initialize the combined indices vector

    // Copy dimensions from this_indices
    for (int i = 0; i < this_rank - 1; ++i) {
        result_indices[i] = this_indices[i]; // Copy the corresponding index
    }

    // Insert dimensions from other_indices
    for (int i = 0; i < other_rank - 1; ++i) {
        result_indices[this_rank - 1 + i] = other_indices[i + 1]; // Adjust and insert the index
    }

    return result_indices; // Return the combined indices
}

template <typename T>
int Tensor<T>::toFlatIndex(const std::vector<int>& indices) const {
    size_t flatIndex = 0; // Initialize the flat index accumulator
    size_t product = 1; // Start with a product of 1
    for (size_t i = indices.size(); i > 0; --i) {
        const auto index = static_cast<size_t>(indices[i - 1]); // Get the current index
        flatIndex += index * product; // Accumulate the product to compute the flat index
        product *= dimensions[i - 1]; // Update the product for the next dimension
    }
    return static_cast<int>(flatIndex); // Return the computed flat index
}

#endif // TENSOR_TPP