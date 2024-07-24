#include <gtest/gtest.h>
#include "../include/Tensor.h"  // Include your Tensor class header

// Helper function to create a Tensor for testing
template<typename T>
Tensor<T> createTestTensor() {
    std::vector<int> dims = {2, 2, 2};
    std::vector<T> data = {1, 2, 3, 4, 5, 6, 7, 8};
    return Tensor<T>(dims, data);
}

// Tests for constructors
TEST(Tensor, ConstructorWithDimsAndData) {
    const std::vector<int> dims = {2, 2};
    const std::vector<int> data = {1, 2, 3, 4};
    const Tensor<int> tensor(dims, data);
    EXPECT_EQ(tensor.shape(), dims);
    EXPECT_EQ(tensor.size(), 4);
}

TEST(Tensor, ConstructorWithData) {
    const std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
    const Tensor<int> tensor(data);
    EXPECT_EQ(tensor.size(), 4);
    EXPECT_EQ(tensor.shape(), std::vector<int>({2, 2}));
}

TEST(Tensor, ConstructorWithDimsOnly) {
    const std::vector<int> dims = {2, 2};
    const Tensor<int> tensor(dims);
    EXPECT_EQ(tensor.shape(), dims);
    EXPECT_EQ(tensor.size(), 4);
}

TEST(Tensor, ConstructorWithInitializerList) {
    const Tensor<int> tensor({2, 2});
    EXPECT_EQ(tensor.shape(), std::vector<int>({2, 2}));
}

// Tests for shape, size, and print
TEST(Tensor, Shape) {
    const Tensor<int> tensor({2, 3});
    EXPECT_EQ(tensor.shape(), std::vector<int>({2, 3}));
}

TEST(Tensor, Size) {
    const Tensor<int> tensor({2, 3});
    EXPECT_EQ(tensor.size(), 6);
}

// Tests for manipulation functions
TEST(Tensor, SetGet) {
    Tensor<int> tensor({2, 2});
    tensor.set({0, 0}, 5);
    EXPECT_EQ(tensor.get({0, 0}), 5);
}

TEST(Tensor, Fill) {
    Tensor<int> tensor({2, 2});
    tensor.fill(7);
    EXPECT_EQ(tensor.get({0, 0}), 7);
    EXPECT_EQ(tensor.get({1, 1}), 7);
}

TEST(Tensor, Slice) {
    const Tensor<int> tensor = Tensor<int>::ones({4, 4});
    const Tensor<int> sliced = tensor.slice(1, 1, 3);
    EXPECT_EQ(sliced.shape(), std::vector<int>({4, 2}));
    EXPECT_EQ(sliced.get({0, 0}), 1);
}

TEST(Tensor, Concatenate) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    const Tensor<int> concatenated = tensor1.concatenate(tensor2, 2);
    EXPECT_EQ(concatenated.shape(), std::vector<int>({2, 2, 4}));
}

TEST(Tensor, ExpandDims) {
    const Tensor<float> tensor({3, 4, 5});
    const Tensor<float> expanded = tensor.expandDims(1);
    EXPECT_EQ(expanded.shape(), std::vector<int>({3, 1, 4, 5}));
}

TEST(Tensor, Squeeze) {
    const Tensor<float> tensor({3, 1, 4, 1});
    const Tensor<float> squeezed = tensor.squeeze();
    EXPECT_EQ(squeezed.shape(), std::vector<int>({3, 4}));
}

TEST(Tensor, Reshape) {
    const Tensor<int> tensor({2, 2});
    const Tensor<int> reshaped = tensor.reshape({1, 4});
    EXPECT_EQ(reshaped.shape(), std::vector<int>({1, 4}));
}

TEST(Tensor, Transpose) {
    const Tensor<int> tensor({2, 3});
    const Tensor<int> transposed = tensor.transpose({1, 0});
    EXPECT_EQ(transposed.shape(), std::vector<int>({3, 2}));
}

TEST(Tensor, Zeros) {
    const Tensor<int> tensor = Tensor<int>::zeros({2, 2});
    EXPECT_EQ(tensor.get({0, 0}), 0);
}

TEST(Tensor, Ones) {
    const Tensor<int> tensor = Tensor<int>::ones({2, 2});
    EXPECT_EQ(tensor.get({0, 0}), 1);
}

// TEST(Tensor, Tril) {
//     const Tensor<int> tensor = Tensor<int>::ones({3, 3});
//     const Tensor<int> tril = tensor.tril();
//     EXPECT_EQ(tril.get({0, 2}), 0);
// }
//
// TEST(Tensor, Triu) {
//     const Tensor<int> tensor = Tensor<int>::ones({3, 3});
//     const Tensor<int> triu = tensor.triu();
//     EXPECT_EQ(triu.get({2, 0}), 0);
// }

// Tests for operators

TEST(Tensor, Addition) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    const Tensor<int> result = tensor1 + tensor2;
    EXPECT_EQ(result.get({0, 0, 0}), 2);
}

TEST(Tensor, Subtraction) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    const Tensor<int> result = tensor1 - tensor2;
    EXPECT_EQ(result.get({0, 0, 0}), 0);
}

// TEST(Tensor, Multiplication) {
//     const Tensor<int> tensor1 = createTestTensor<int>();
//     const Tensor<int> tensor2 = createTestTensor<int>();
//     const Tensor<int> result = tensor1 * tensor2;
//     EXPECT_EQ(result.get({0, 0, 0}), 1);
// }
//
// TEST(Tensor, Division) {
//     const Tensor<int> tensor1 = createTestTensor<int>();
//     const Tensor<int> tensor2 = createTestTensor<int>();
//     const Tensor<int> result = tensor1 / tensor2;
//     EXPECT_EQ(result.get({0, 0, 0}), 1);
// }

TEST(Tensor, ScalarAddition) {
    const Tensor<int> tensor = createTestTensor<int>();
    const Tensor<int> result = tensor + 10;
    EXPECT_EQ(result.get({0, 0, 0}), 11);
}

TEST(Tensor, ScalarSubtraction) {
    const Tensor<int> tensor = createTestTensor<int>();
    const Tensor<int> result = tensor - 1;
    EXPECT_EQ(result.get({0, 0, 0}), 0);
}

TEST(Tensor, ScalarMultiplication) {
    const Tensor<int> tensor = createTestTensor<int>();
    const Tensor<int> result = tensor * 2;
    EXPECT_EQ(result.get({0, 0, 0}), 2);
}

TEST(Tensor, ScalarDivision) {
    const Tensor<int> tensor = createTestTensor<int>();
    const Tensor<int> result = tensor / 2;
    EXPECT_EQ(result.get({0, 0, 0}), 0);
}

// TEST(Tensor, BracketOperator) {
//     Tensor<int> tensor = createTestTensor<int>();
//     EXPECT_EQ(tensor[{0, 0, 0}], 1);
// }
//
// TEST(Tensor, ParenthesisOperator) {
//     Tensor<int> tensor = createTestTensor<int>();
//     EXPECT_EQ(tensor({0, 0, 0}), 1);
// }

// TEST(Tensor, AssignmentOperator) {
//     Tensor<int> tensor = createTestTensor<int>();
//     tensor[{0, 0, 0}] = 10;
//     EXPECT_EQ(tensor[{0, 0, 0}], 10);
// }

// TEST(Tensor, ComparisonOperators) {
//     const Tensor<int> tensor1 = createTestTensor<int>();
//     const Tensor<int> tensor2 = createTestTensor<int>();
//     EXPECT_TRUE(tensor1 == tensor2);
//     EXPECT_FALSE(tensor1 != tensor2);
// }
