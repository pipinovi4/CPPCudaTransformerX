#include <iostream>
#include <vector>
#include "../include/Tensor.h"

int main() {
    // Matrices addition
    Tensor<float> one_tensor({2, 2, 2}, std::vector<float>{1, 1, 1, 1, 1, 1, 2, 2});

    Tensor<int> two_tensor({2, 2}, std::vector<int>({10, 10, 10, 10}));
    Tensor<int> three_tensor({2, 2}, std::vector<int>({25, 25, 25, 25}));
//
//    Tensor<int> fiveth_tensor = two_tensor + two_tensor;
//    fiveth_tensor.print();
//
//    // Matrices subtraction
//    Tensor<int> sixth_tensor = two_tensor - three_tensor;
//    sixth_tensor.print();
//
//    // Scalar operations with Tensor
//    Tensor<int> ninth_tensor = two_tensor + 5;
//    ninth_tensor.print();
//
//    Tensor<int> tenth_tensor = two_tensor - 5;
//    tenth_tensor.print();
//
//    Tensor<int> eleventh_tensor = two_tensor * 5;
//    eleventh_tensor.print();
//
//    Tensor<int> twelveth_tensor = two_tensor / 5;
//    twelveth_tensor.print();

    // Matrices multiplication
    Tensor<int> seventh_tensor = two_tensor * three_tensor;
    seventh_tensor.print();

//    // Matrices division
//    Tensor<int> eighth_tensor = three_tensor / two_tensor;
//    eighth_tensor.print();
    return 0;
}