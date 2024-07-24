#include "gtest/gtest.h"
#include "../src/Tensor.cpp"

TEST(Tensor, Constructor) {
    const Tensor<int> tensor({2, 2, 2});
    EXPECT_EQ(tensor.size(), 8);
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);
    EXPECT_EQ(tensor.shape()[2], 2);
}

TEST(Tensor, ConstructorWithInitializerList) {
    const Tensor<int> tensor({2, 2, 2});
    EXPECT_EQ(tensor.size(), 8);
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);
    EXPECT_EQ(tensor.shape()[2], 2);
}

TEST(Tensor, ConstructorWithDimensions) {
    const Tensor<int> tensor({2, 2, 2});
    EXPECT_EQ(tensor.size(), 8);
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);
    EXPECT_EQ(tensor.shape()[2], 2);
}

TEST(Tensor, ConstructorWithData) {
    const Tensor<int> tensor;
    EXPECT_EQ(tensor.size(), 8);
    EXPECT_EQ(tensor.shape().size(), 3);
    EXPECT_EQ(tensor.shape()[0], 2);
    EXPECT_EQ(tensor.shape()[1], 2);
    EXPECT_EQ(tensor.shape()[2], 2);
}

TEST(Tensor, SetAndGet) {
    Tensor<int> tensor({2, 2, 2});
    tensor.set({0, 0, 0}, 1);
    tensor.set({0, 0, 1}, 2);
    tensor.set({0, 1, 0}, 3);
    tensor.set({0, 1, 1}, 4);
    tensor.set({1, 0, 0}, 5);
    tensor.set({1, 0, 1}, 6);
    tensor.set({1, 1, 0}, 7);
    tensor.set({1, 1, 1}, 8);
    EXPECT_EQ(tensor.get({0, 0, 0}), 1);
    EXPECT_EQ(tensor.get({0, 0, 1}), 2);
    EXPECT_EQ(tensor.get({0, 1, 0}), 3);
    EXPECT_EQ(tensor.get({0, 1, 1}), 4);
    EXPECT_EQ(tensor.get({1, 0, 0}), 5);
    EXPECT_EQ(tensor.get({1, 0, 1}), 6);
    EXPECT_EQ(tensor.get({1, 1, 0}), 7);
    EXPECT_EQ(tensor.get({1, 1, 1}), 8);
}

TEST(Tensor, Fill) {
    Tensor<int> tensor({2, 2, 2});
    tensor.fill(5);
    EXPECT_EQ(tensor.get({0, 0, 0}), 5);
    EXPECT_EQ(tensor.get({0, 0, 1}), 5);
    EXPECT_EQ(tensor.get({0, 1, 0}), 5);
    EXPECT_EQ(tensor.get({0, 1, 1}), 5);
    EXPECT_EQ(tensor.get({1, 0, 0}), 5);
    EXPECT_EQ(tensor.get({1, 0, 1}), 5);
    EXPECT_EQ(tensor.get({1, 1, 0}), 5);
    EXPECT_EQ(tensor.get({1, 1, 1}), 5);
}