#include <iostream>
#include <vector>
#include "../include/Tensor.h"

int main() {
    // Initialize tensor with dimensions and data
    Tensor<float> one_tensor({1, 2, 3}, std::vector<float>{1, 1, 1, 1, 1, 1});

    // Initialize tensor with dimensions only
    Tensor<int> two_tensor({1, 2});

    // Initialize tensor with data only
    Tensor<double> three_tensor(std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9});

    // Initialize tensor the tensor with dimensions from an initializer list
    Tensor<int> four_tensor = {1, 2, 3, 4};

    // Print the tensor
    one_tensor.print();
    two_tensor.print();
    three_tensor.print();
    four_tensor.print();

    return 0;
}