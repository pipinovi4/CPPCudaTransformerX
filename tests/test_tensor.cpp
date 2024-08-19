#include <gtest/gtest.h>
#include "../include/Tensor.h"

std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i];
    if (i != vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

class TensorTest : public ::testing::Test {
protected:
  Tensor<float> input_tensor;
  Tensor<float> expected_tensor;
  Tensor<float> result_tensor;

  TensorTest() : input_tensor({2, 2}), expected_tensor({2, 2}), result_tensor({2, 2}) {}

  void SetUpData(const Tensor<float>& input, const Tensor<float>& expected) {
    input_tensor = input;
    expected_tensor = expected;
  }

  void ExpectTensorNear(const float abs_error = 1e-2) const {
    EXPECT_TRUE(result_tensor.shape() == expected_tensor.shape()) << "Expected shape: " << expected_tensor.shape() << " but got " << result_tensor.shape();
    EXPECT_TRUE(result_tensor.size() == expected_tensor.size()) << "Expected size: " << expected_tensor.size() << " but got " << result_tensor.size();
    input_tensor.print();
    expected_tensor.print();
    result_tensor.print();
    for (int i = 0; i < result_tensor.size(); i++) {
      EXPECT_NEAR(result_tensor.data[i], expected_tensor.data[i], abs_error) << "Expected value: " << expected_tensor.data[i] << " but got " << result_tensor.data[i] << " at index " << i;
    }
  }
};

class Fill : public TensorTest {
protected:
  Fill() {}

  void ProcessData(const float value) {
    result_tensor = input_tensor;
    result_tensor.fill(value);
  }
};

TEST_F(Fill, HandlesNormalCase) {
  const std::vector<int> dims {5};
  const std::vector<float> inputData = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  const std::vector<float> expectedData = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  Tensor<float> result(dims);
  SetUpData(input, expected);

  ProcessData(1.0f);

  ExpectTensorNear();
}

TEST_F(Fill, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {5};

  const std::vector<float> inputData = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  const std::vector<float> expectedData = {111111.1111111111f, 111111.1111111111f, 111111.1111111111f, 111111.1111111111f, 111111.1111111111f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(111111.1111111111f);

  ExpectTensorNear();
}

class Sqrt : public TensorTest {
protected:
  Sqrt() {}

  void ProcessData() {
    result_tensor = input_tensor.sqrt();
  }
};

TEST_F(Sqrt, HandlesNormalCase) {
  const std::vector<int> dims {5};
  const std::vector<float> inputData = {100.0f, 49.0f, 81.0f, 0.0001f, 0.5f};
  const std::vector<float> expectedData = {10.0f, 7.0f, 9.0f, 0.01f, 0.70710678f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData();

  ExpectTensorNear();
}

TEST_F(Sqrt, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {5};
  const std::vector<float> inputData = {1000000.0f, 4900.0f, 8100.0f, 0.00000001f, 0.5f};
  const std::vector<float> expectedData = {1000.0f, 70.0f, 90.0f, 0.0001f, 0.70710678f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData();

  ExpectTensorNear();
}

class Pow : public TensorTest {
protected:
  Pow() {}

  void ProcessData(const float& exponent) {
    result_tensor = input_tensor.pow(exponent);
  }
};

TEST_F(Pow, HandlesNormalCase) {
  const std::vector<int> dims {5};
  const std::vector<float> inputData = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> expectedData = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(2.0f);

  ExpectTensorNear();
}

class Sum : public TensorTest {
protected:
  Sum() {}

  void ProcessData(const int& axis) {
    result_tensor = input_tensor.sum(axis);
  }
};

TEST_F(Sum, HandlesNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {4.0f, 6.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2}, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

TEST_F(Sum, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {4000000.0f, 6000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2}, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

class Mean : public TensorTest {
protected:
  Mean() {}

  void ProcessData(const int axis) {
    result_tensor = input_tensor.mean(axis);
  }
};

TEST_F(Mean, HandlesNormalCase) {
  const std::vector<int> dims {10, 10};
  const std::vector<float> inputData = {
       0.27110479, 0.62862718, 0.49281226, 0.19090492, 0.36258228,
        0.82362251, 0.47257699, 0.90574549, 0.43314019, 0.88641438,
       0.30669065, 0.98994557, 0.66548694, 0.91916649, 0.84325348,
        0.86430052, 0.88607458, 0.49647809, 0.57642615, 0.44668161,
       0.11018024, 0.7086044 , 0.70025187, 0.18306647, 0.91899664,
        0.17522525, 0.19388817, 0.93600368, 0.82197419, 0.90138243,
       0.8061687 , 0.8532697 , 0.84162007, 0.99736561, 0.28665019,
        0.67569817, 0.28700122, 0.88498503, 0.08498305, 0.00216105,
       0.99514645, 0.66746488, 0.26415438, 0.15026899, 0.54645516,
        0.70478839, 0.93144364, 0.35711562, 0.80979372, 0.00188611,
       0.3218287 , 0.54064281, 0.09203637, 0.08812514, 0.05688664,
        0.66953922, 0.400226  , 0.59908694, 0.50799126, 0.15442722,
       0.12895682, 0.82466733, 0.73927711, 0.9219204 , 0.81801937,
        0.2375317 , 0.53187194, 0.21850902, 0.01650145, 0.93361651,
       0.08320101, 0.36992348, 0.02521315, 0.83040181, 0.33378798,
        0.38248155, 0.6323301 , 0.37128949, 0.6092686 , 0.10022388,
       0.86851145, 0.3406736 , 0.2955068 , 0.92227654, 0.55165393,
        0.0259786 , 0.56317192, 0.6103764 , 0.49638384, 0.24230716,
       0.7876868 , 0.04067058, 0.45939698, 0.26417751, 0.78770263,
        0.67777073, 0.63149595, 0.29830289, 0.12516273, 0.53884066
  };
  const std::vector<float> expectedData = {
    0.5467531, 0.69945041, 0.56495733, 0.57199028, 0.54285173,
    0.34307903, 0.53708717, 0.37381211, 0.49168402, 0.46112074
  };

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({dims[0]}, expectedData);
  SetUpData(input, expected);

  ProcessData(-1);

  ExpectTensorNear();
}

class Argmax : public TensorTest {
protected:
  Argmax() {}

  void ProcessData(const int axis) {
    result_tensor = input_tensor.argmax(axis);
  }
};

TEST_F(Argmax, HandleNormalCase) {
  const std::vector<int> dims {2, 3};
  const std::vector<float> inputData = {1.0f, 3.0f, 2.0f, 4.0f, 6.0f, 5.0f};
  const std::vector<float> expectedData = {1, 1};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2}, expectedData);
  SetUpData(input, expected);

  ProcessData(1);

  ExpectTensorNear();
}

TEST_F(Argmax, HandleEdgeCaseNegativeAxis) {
  const std::vector<int> dims {2, 3};
  const std::vector<float> inputData = {7.0f, 3.0f, 1.0f, 0.0f, 6.0f, 2.0f};
  const std::vector<float> expectedData = {0, 1};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2}, expectedData);
  SetUpData(input, expected);

  ProcessData(-1);

  ExpectTensorNear();
}

TEST_F(Argmax, HandleEdgeCaseSingleElement) {
  const std::vector<int> dims {1};
  const std::vector<float> inputData = {42.0f};
  const std::vector<float> expectedData = {0};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({1}, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}


class ExpandDimsAs : public TensorTest {
protected:
  ExpandDimsAs() {}

  void ProcessData(const std::vector<int>& other_dimensions) {
    result_tensor = input_tensor.expandDimsAs(other_dimensions);
  }
};

TEST_F(ExpandDimsAs, HandlesNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const std::vector<int> otherDimensions {2, 2, 2};
  const Tensor<float> expected(otherDimensions, expectedData);
  SetUpData(input, expected);

  ProcessData(otherDimensions);

  ExpectTensorNear();
}

class Slice : public TensorTest {
protected:
  Slice() {}

  void ProcessData(const int& axis, const int& start, const int& end, const int& step) {
    result_tensor = input_tensor.slice(axis, start, end, step);
  }
};

TEST_F(Slice, HandlesNormalCase) {
  const std::vector<int> dims {5};
  const std::vector<float> inputData = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f};
  const std::vector<float> expectedData = {1.0f, 0.5f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2}, expectedData);
  SetUpData(input, expected);

  ProcessData(0, 1, 4, 2);

  ExpectTensorNear();
}

TEST_F(Slice, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {10};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f, 5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f, 9000000.0f, 10000000.0f};
  const std::vector<float> expectedData = {3000000.0f, 4000000.0f, 5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f, 9000000.0f, 10000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({8}, expectedData);
  SetUpData(input, expected);

  ProcessData(0, 2, 10, 1);

  ExpectTensorNear();
}

class Concatenate : public TensorTest {
protected:
  Concatenate() {}

  void ProcessData(const Tensor<float>& other, const int& axis) {
    result_tensor = input_tensor.concatenate(other, axis);
  }
};

TEST_F(Concatenate, HandlesNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected({4, 2}, expectedData);
  SetUpData(input, expected);

  ProcessData(other, 0);

  ExpectTensorNear();
}

TEST_F(Concatenate, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> otherData = {5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f, 5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected({4, 2}, expectedData);
  SetUpData(input, expected);

  ProcessData(other, 0);

  ExpectTensorNear();
}

class ExpandDims : public TensorTest {
protected:
  ExpandDims() {}

  void ProcessData(const int& axis) {
    result_tensor = input_tensor.expandDims(axis);
  }
};

TEST_F(ExpandDims, HandlesNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 3.0f, 2.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2, 2, 1}, expectedData);
  SetUpData(input, expected);

  ProcessData(2);

  ExpectTensorNear();
}

TEST_F(ExpandDims, HandlesEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 3000000.0f, 2000000.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2, 2, 1}, expectedData);
  SetUpData(input, expected);

  ProcessData(2);

  ExpectTensorNear();
}

class Squeeze : public TensorTest {
protected:
  Squeeze() {}

  void ProcessData(const int& axis) {
    result_tensor = input_tensor.squeeze();
  }
};

TEST_F(Squeeze, HandleNormalCase) {
  const std::vector<int> dims {2, 1, 2, 1};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 2.0f, 3.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2, 2}, expectedData);
  SetUpData(input, expected);

  ProcessData(2);

  ExpectTensorNear();
}

TEST_F(Squeeze, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 1, 2, 1};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected({2, 2}, expectedData);
  SetUpData(input, expected);

  ProcessData(2);

  ExpectTensorNear();
}

class Reshape : public TensorTest {
protected:
  Reshape() {}

  void ProcessData(const std::vector<int>& newDimensions) {
    result_tensor = input_tensor.reshape(newDimensions);
  }
};

TEST_F(Reshape, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 2.0f, 3.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

TEST_F(Reshape, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

class Transpose : public TensorTest {
protected:
  Transpose() {}

  void ProcessData(const std::vector<int>& permutation) {
    result_tensor = input_tensor.transpose(permutation);
  }
};

TEST_F(Transpose, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 3.0f, 2.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const std::vector<int> permutation {1, 0};
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(permutation);

  ExpectTensorNear();
}

TEST_F(Transpose, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 3000000.0f, 2000000.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const std::vector<int> permutation {1, 0};
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(permutation);

  ExpectTensorNear();
}

class Zeros : public TensorTest {
protected:
  Zeros() {}

  void ProcessData(const std::vector<int>& dims) {
    result_tensor = Tensor<float>::zeros(dims);
  }
};

TEST_F(Zeros, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> expectedData = {0.0f, 0.0f, 0.0f, 0.0f};

  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input_tensor, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

TEST_F(Zeros, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> expectedData = {0.0f, 0.0f, 0.0f, 0.0f};

  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input_tensor, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

class Ones : public TensorTest {
protected:
  Ones() {}

  void ProcessData(const std::vector<int>& dims) {
    result_tensor = Tensor<float>::ones(dims);
  }
};

TEST_F(Ones, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> expectedData = {1.0f, 1.0f, 1.0f, 1.0f};

  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input_tensor, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

TEST_F(Ones, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> expectedData = {1.0f, 1.0f, 1.0f, 1.0f};

  const std::vector<int> newDimensions {4};
  const Tensor<float> expected(newDimensions, expectedData);
  SetUpData(input_tensor, expected);

  ProcessData(newDimensions);

  ExpectTensorNear();
}

class Tril : public TensorTest {
protected:
  Tril() {}

  void ProcessData(const int& axis) {
    result_tensor = input_tensor.tril(axis);
  }
};

TEST_F(Tril, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 0.0f, 3.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

TEST_F(Tril, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 0.0f, 3000000.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

class Triu : public TensorTest {
protected:
  Triu() {}

  void ProcessData(const int& axis) {
    result_tensor = input_tensor.triu(axis);
  }
};

TEST_F(Triu, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> expectedData = {1.0f, 2.0f, 0.0f, 4.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

TEST_F(Triu, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> expectedData = {1000000.0f, 2000000.0f, 0.0f, 4000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(0);

  ExpectTensorNear();
}

class Dot : public TensorTest {
protected:
  Dot() {}

  void ProcessData(const Tensor<float>& other) {
    result_tensor = input_tensor.dot(other);
  }
};

TEST_F(Dot, HandleNormalCase) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {17.0f, 20.0f, 28.0f, 33.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

TEST_F(Dot, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {2, 2};
  const std::vector<float> inputData = {1000.0f, 2000.0f, 3000.0f, 4000.0f};
  const std::vector<float> otherData = {5000.0f, 6000.0f, 7000.0f, 8000.0f};
  const std::vector<float> expectedData = {17000000.0f, 20000000.0f, 28000000.0f, 33000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

class Add : public TensorTest {
protected:
  Add() {}

  void ProcessData(const Tensor<float>& other) {
    result_tensor = input_tensor + other;
  }
};

TEST_F(Add, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {6.0f, 8.0f, 10.0f, 12.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}
TEST_F(Add, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> otherData = {5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};
  const std::vector<float> expectedData = {6000000.0f, 8000000.0f, 10000000.0f, 12000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

class Subtract : public TensorTest {
protected:
  Subtract() {}

  void ProcessData(const Tensor<float>& other) {
    result_tensor = input_tensor - other;
  }
};

TEST_F(Subtract, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {-4.0f, -4.0f, -4.0f, -4.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

TEST_F(Subtract, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> otherData = {5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};
  const std::vector<float> expectedData = {-4000000.0f, -4000000.0f, -4000000.0f, -4000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

class Multiply : public TensorTest {
protected:
  Multiply() {}

  void ProcessData(const Tensor<float>& other) {
    result_tensor = input_tensor * other;
  }
};

TEST_F(Multiply, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {5.0f, 12.0f, 21.0f, 32.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

TEST_F(Multiply, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> otherData = {5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};
  const std::vector<float> expectedData = {5000000000000.0f, 12000000000000.0f, 21000000000000.0f, 32000000000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

class Divide : public TensorTest{
protected:
  Divide() {}

  void ProcessData(const Tensor<float>& other) {
    result_tensor = input_tensor / other;
  }
};

TEST_F(Divide, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> otherData = {5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> expectedData = {0.2f, 0.33333333f, 0.42857143f, 0.5f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

TEST_F(Divide, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  const std::vector<float> otherData = {5000000.0f, 6000000.0f, 7000000.0f, 8000000.0f};
  const std::vector<float> expectedData = {0.2f, 0.33333333f, 0.42857143f, 0.5f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> other(dims, otherData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(other);

  ExpectTensorNear();
}

class AddScalar : public TensorTest {
protected:
  AddScalar() {}

  void ProcessData(const float scalar) {
    result_tensor = input_tensor + scalar;
  }
};

TEST_F(AddScalar, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  constexpr float scalar = 5.0f;
  const std::vector<float> expectedData = {6.0f, 7.0f, 8.0f, 9.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

TEST_F(AddScalar, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  constexpr float scalar = 5000000.0f;
  const std::vector<float> expectedData = {6000000.0f, 7000000.0f, 8000000.0f, 9000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

class SubtractScalar : public TensorTest {
protected:
  SubtractScalar() {}

  void ProcessData(const float scalar) {
    result_tensor = input_tensor - scalar;
  }
};

TEST_F(SubtractScalar, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  constexpr float scalar = 5.0f;
  const std::vector<float> expectedData = {-4.0f, -3.0f, -2.0f, -1.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

TEST_F(SubtractScalar, HandleEdgeCaseLargeValues)
{
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  constexpr float scalar = 5000000.0f;
  const std::vector<float> expectedData = {-4000000.0f, -3000000.0f, -2000000.0f, -1000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

class MultiplyScalar : public TensorTest {
protected:
  MultiplyScalar() {}

  void ProcessData(const float scalar) {
    result_tensor = input_tensor * scalar;
  }
};

TEST_F(MultiplyScalar, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  constexpr float scalar = 5.0f;
  const std::vector<float> expectedData = {5.0f, 10.0f, 15.0f, 20.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

TEST_F(MultiplyScalar, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  constexpr float scalar = 5000000.0f;
  const std::vector<float> expectedData = {5000000000000.0f, 10000000000000.0f, 15000000000000.0f, 20000000000000.0f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

class DivideScalar : public TensorTest {
protected:
  DivideScalar() {}

  void ProcessData(const float scalar) {
    result_tensor = input_tensor / scalar;
  }
};

TEST_F(DivideScalar, HandleNormalCase) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1.0f, 2.0f, 3.0f, 4.0f};
  constexpr float scalar = 5.0f;
  const std::vector<float> expectedData = {0.2f, 0.4f, 0.6f, 0.8f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}

TEST_F(DivideScalar, HandleEdgeCaseLargeValues) {
  const std::vector<int> dims {4};
  const std::vector<float> inputData = {1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f};
  constexpr float scalar = 5000000.0f;
  const std::vector<float> expectedData = {0.2f, 0.4f, 0.6f, 0.8f};

  const Tensor<float> input(dims, inputData);
  const Tensor<float> expected(dims, expectedData);
  SetUpData(input, expected);

  ProcessData(scalar);

  ExpectTensorNear();
}
