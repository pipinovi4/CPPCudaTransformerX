#include <vector>
#include "../src/Tensor.cpp"

int main() {
    // Create tensors
    Tensor<float> one_tensor({2, 2, 2}, std::vector<float>{1, 1, 1, 1, 1, 1, 2, 2});
    Tensor<int> two_tensor({2, 2, 3}, std::vector<int>{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10});
    Tensor<int> three_tensor({3, 2, 2}, std::vector<int>{25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25});

     // Matrices multiplication (implement multiplication method in Tensor class)
//     Tensor<int> seventh_tensor = two_tensor * three_tensor;
//     seventh_tensor.print();

     // Matrices division (implement division method in Tensor class)
     // Tensor<int> eighth_tensor = three_tensor / two_tensor;
     // eighth_tensor.print();
    return 0;
}
