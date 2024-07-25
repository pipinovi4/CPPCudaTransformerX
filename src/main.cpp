#include <vector>
#include "../include/Tensor.h"

int main()
{
    // Create a tensor with dimensions
    // only
    Tensor<int> tensor({2, 3, 4});
    tensor.fill(10);
    const Tensor<int> result = tensor[{1}];
    result.print();
    return 0;
}
