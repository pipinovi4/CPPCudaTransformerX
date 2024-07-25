#include <gtest/gtest.h>
#include "../include/Tensor.h"

// Helper function to create 2D Tensors for testing dot product
template <typename T>
Tensor<T> create2DTensor(const std::vector<int>& dims, const std::vector<T>& data) {
    return Tensor<T>(dims, data);
}

// Helper function to create a Tensor for testing
template<typename T>
Tensor<T> createTestTensor() {
    std::vector<int> dims = {2, 2, 2};
    std::vector<T> data = {1, 2, 3, 4, 5, 6, 7, 8};
    return Tensor<T>(dims, data);
}

// Tests for tensor addition
TEST(Tensor, Addition) {
    std::vector<int> dims = {2, 2};
    std::vector<int> dataA = {1, 2, 3, 4};  // A = [[1, 2], [3, 4]]
    std::vector<int> dataB = {5, 6, 7, 8};  // B = [[5, 6], [7, 8]]
    Tensor<int> tensorA = create2DTensor(dims, dataA);
    Tensor<int> tensorB = create2DTensor(dims, dataB);

    Tensor<int> result = tensorA + tensorB;

    std::vector<int> expectedData = {6, 8, 10, 12};  // Expected result A + B
    Tensor<int> expectedResult(dims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Tests for tensor subtraction
TEST(Tensor, Subtraction) {
    std::vector<int> dims = {2, 2};
    std::vector<int> dataA = {5, 6, 7, 8};  // A = [[5, 6], [7, 8]]
    std::vector<int> dataB = {1, 2, 3, 4};  // B = [[1, 2], [3, 4]]
    Tensor<int> tensorA = create2DTensor(dims, dataA);
    Tensor<int> tensorB = create2DTensor(dims, dataB);

    Tensor<int> result = tensorA - tensorB;

    std::vector<int> expectedData = {4, 4, 4, 4};  // Expected result A - B
    Tensor<int> expectedResult(dims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Tests for tensor multiplication (element-wise)
TEST(Tensor, ElementWiseMultiplication) {
    std::vector<int> dims = {2, 2};
    std::vector<int> dataA = {1, 2, 3, 4};  // A = [[1, 2], [3, 4]]
    std::vector<int> dataB = {5, 6, 7, 8};  // B = [[5, 6], [7, 8]]
    Tensor<int> tensorA = create2DTensor(dims, dataA);
    Tensor<int> tensorB = create2DTensor(dims, dataB);

    Tensor<int> result = tensorA * tensorB;

    std::vector<int> expectedData = {5, 12, 21, 32};  // Expected result A * B (element-wise)
    Tensor<int> expectedResult(dims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Tests for tensor transpose
TEST(Tensor, Transpose) {
    const std::vector<int> dims = {2, 3};
    const std::vector<int> data = {1, 2, 3, 4, 5, 6};  // A = [[1, 2, 3], [4, 5, 6]]
    const Tensor<int> tensor = create2DTensor(dims, data);

    const Tensor<int> result = tensor.transpose();

    const std::vector<int> expectedDims = {3, 2};
    const std::vector<int> expectedData = {1, 4, 2, 5, 3, 6};  // Expected result of transposing A
    const Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Tests for tensor reshaping
TEST(Tensor, Reshape) {
    const std::vector<int> dims = {2, 3};
    const std::vector<int> data = {1, 2, 3, 4, 5, 6};  // A = [[1, 2, 3], [4, 5, 6]]
    const Tensor<int> tensor = create2DTensor(dims, data);

    const Tensor<int> result = tensor.reshape({3, 2});

    const std::vector<int> expectedDims = {3, 2};
    const std::vector<int> expectedData = {1, 2, 3, 4, 5, 6};  // Expected reshaped tensor
    const Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
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

TEST(Tensor, Zeros) {
    const Tensor<int> tensor = Tensor<int>::zeros({2, 2});
    EXPECT_EQ(tensor.get({0, 0}), 0);
}

TEST(Tensor, Ones) {
    const Tensor<int> tensor = Tensor<int>::ones({2, 2});
    EXPECT_EQ(tensor.get({0, 0}), 1);
}

TEST(Tensor, Tril) {
    Tensor<int> tensor = Tensor<int>::ones({3, 3});
    const Tensor<int> tril = tensor.tril();
    EXPECT_EQ(tril.get({0, 2}), 0);
}

TEST(Tensor, Triu) {
    Tensor<int> tensor = Tensor<int>::ones({10, 10});
    const Tensor<int> triu = tensor.triu();
    EXPECT_EQ(triu.get({1, 0}), 0);
    EXPECT_EQ(triu.get({2, 0}), 0);
    EXPECT_EQ(triu.get({2, 1}), 0);

    const Tensor<int> triu_axis2 = tensor.triu(10);
}

TEST(Tensor, Multiplication) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    const Tensor<int> result = tensor1 * tensor2;
    EXPECT_EQ(result.get({0, 0, 0}), 1);
}

TEST(Tensor, Division) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    const Tensor<int> result = tensor1 / tensor2;
    EXPECT_EQ(result.get({0, 0, 0}), 1);
}

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

TEST(Tensor, ParenthesisOperator) {
    Tensor<int> tensor = createTestTensor<int>();
    const std::vector<int> resultData{1};
    // Define the expected result for the given indices
    const auto expectedResult = Tensor<int>({1}, resultData); // Value at index {0, 0, 0}

    // Fetch the result using the operator[] method
    const auto result = tensor[{0, 0, 0}];

    // Check if the result matches the expected result
    EXPECT_EQ(result, expectedResult);
}

TEST(Tensor, ComparisonOperators) {
    const Tensor<int> tensor1 = createTestTensor<int>();
    const Tensor<int> tensor2 = createTestTensor<int>();
    EXPECT_TRUE(tensor1 == tensor2);
    EXPECT_FALSE(tensor1 != tensor2);
}

// Test dot product with basic 2D tensors
TEST(Tensor, DotProductBasic) {
    // Test dot product for simple 2D tensors
    std::vector<int> dimsA = {2, 2};
    std::vector<int> dataA = {1, 2, 3, 4};  // A = [[1, 2], [3, 4]]
    Tensor<int> tensorA = create2DTensor(dimsA, dataA);

    std::vector<int> dimsB = {2, 2};
    std::vector<int> dataB = {5, 6, 7, 8};  // B = [[5, 6], [7, 8]]
    Tensor<int> tensorB = create2DTensor(dimsB, dataB);

    Tensor<int> result = tensorA.dot(tensorB);

    // Define the expected result
    std::vector<int> expectedDims = {2, 2};
    std::vector<int> expectedData = {19, 22, 43, 50};  // A dot B
    Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);

    std::cout << "Dot Product Basic Test Passed." << std::endl;
}

// Test dot product with a single-dimensional tensor and a 2D tensor
TEST(Tensor, DotProduct1DAnd2D) {
    std::vector<int> dimsA = {2};
    std::vector<int> dataA = {1, 2};  // A = [1, 2]
    Tensor<int> tensorA = create2DTensor(dimsA, dataA);

    std::vector<int> dimsB = {2, 2};
    std::vector<int> dataB = {1, 2, 3, 4};  // B = [[1, 2], [3, 4]]
    Tensor<int> tensorB = create2DTensor(dimsB, dataB);

    Tensor<int> result = tensorA.dot(tensorB);

    std::vector<int> expectedDims = {2};
    std::vector<int> expectedData = {7, 10};  // Result of A dot B
    Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Test dot product with tensors having mismatched dimensions
TEST(Tensor, DotProductDimensionMismatch) {
    std::vector<int> dimsA = {2, 3};
    std::vector<int> dataA = {1, 2, 3, 4, 5, 6};  // A = [[1, 2, 3], [4, 5, 6]]
    Tensor<int> tensorA = create2DTensor(dimsA, dataA);

    std::vector<int> dimsB = {3, 2};
    std::vector<int> dataB = {7, 8, 9, 10, 11, 12};  // B = [[7, 8], [9, 10], [11, 12]]
    Tensor<int> tensorB = create2DTensor(dimsB, dataB);

    Tensor<int> result = tensorA.dot(tensorB);

    std::vector<int> expectedDims = {2, 2};
    std::vector<int> expectedData = {58, 64, 139, 154};  // Result of A dot B
    Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Test dot product with a 3D tensor
TEST(Tensor, DotProduct3D) {
    std::vector<int> dimsA = {2, 2, 2};
    std::vector<int> dataA = {1, 2, 3, 4, 5, 6, 7, 8};  // A = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    Tensor<int> tensorA = create2DTensor(dimsA, dataA);

    std::vector<int> dimsB = {2, 2, 2};
    std::vector<int> dataB = {1, 0, 0, 1, 1, 0, 0, 1};  // B = [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]
    Tensor<int> tensorB = create2DTensor(dimsB, dataB);

    Tensor<int> result = tensorA.dot(tensorB);

    std::vector<int> expectedDims = {2, 2, 2};
    std::vector<int> expectedData = {1, 0, 0, 1, 5, 6, 7, 8};  // Result of A dot B
    Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}

// Test dot product with a zero-dimensional tensor (scalar)
TEST(Tensor, DotProductScalar) {
    std::vector<int> dimsA = {};  // Scalar
    std::vector<int> dataA = {1};
    Tensor<int> tensorA(dimsA, dataA);

    std::vector<int> dimsB = {2, 2};
    std::vector<int> dataB = {1, 2, 3, 4};  // B = [[1, 2], [3, 4]]
    Tensor<int> tensorB(dimsB, dataB);

    Tensor<int> result = tensorA.dot(tensorB);

    std::vector<int> expectedDims = {2, 2};
    std::vector<int> expectedData = {1, 2, 3, 4};  // Result of scalar times B
    Tensor<int> expectedResult(expectedDims, expectedData);

    EXPECT_EQ(result.shape(), expectedResult.shape());
    EXPECT_TRUE(result == expectedResult);
}