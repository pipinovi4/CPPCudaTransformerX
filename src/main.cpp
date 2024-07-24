#include <iostream>
#include <vector>
#include "../include/Tensor.h"

int main() {
    // Example 1: Create a Tensor with dimensions and data
    const std::vector<int> dims = {2, 3};  // Dimensions of the tensor
    const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // Data for the tensor

    try {
        const Tensor<float> tensor1(dims, data);
        tensor1.print();
        std::cout << "Tensor1 created with dimensions and data." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Example 2: Create a Tensor with only dimensions
    const std::vector<int> dims2 = {3, 4};  // Dimensions of the tensor

    try {
        Tensor<float> tensor2(dims2);
        std::cout << "Tensor2 created with dimensions only." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // Example 3: Create a Tensor with initializer list for dimensions
    Tensor<float> tensor3({4, 5});  // Using initializer list for dimensions

    std::cout << "Tensor3 created with initializer list for dimensions." << std::endl;

    // Example 4: Create a Tensor with only data
    const std::vector<float> data2 = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};  // Data for the tensor

    try {
        Tensor<float> tensor4(data2);
        std::cout << "Tensor4 created with data only." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
