#include <iostream>
#include <vector>
#include "../include/Tensor.h"

// Primary template for is_vector
template <typename D>
struct is_vector : std::false_type {};

// Specialization for std::vector
template <typename D, typename Allocator>
struct is_vector<std::vector<D, Allocator>> : std::true_type {};


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

int main() {
    // Define a 3D vector of floats
    std::vector<std::vector<std::vector<float>>> tripleVector = {
            {{1, 2}, {3, 4}},
            {{5, 6}, {7, 8}}
    };

    Tensor<float> tensor_data({2, 4}, tripleVector);
    tensor_data.print();

    std::vector<int> shape_data = tensor_data.shape();
    for (const auto& dim : shape_data) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    Tensor<float> reshaped = tensor_data.reshape({2, 2, 2, 1});
    reshaped.print();
    return 0;
}
