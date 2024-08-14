#include "../include/Tensor.h"
#include <chrono>

// My custom float_16 type horrible and in 1.5 times slower than float

int main() {
    // Test the float_16 compuation speed with float
    Tensor<float_16> t1{100, 100, 100};
    Tensor<float_16> t2{100, 100, 100};

    t1.fill(2);
    t2.fill(3);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Perform 1000 multiplications
    for (int i = 0; i < 10; ++i) {
        t1 = t1 + t2;
    }

    // Stop timer
    auto stop = std::chrono::high_resolution_clock::now();

    // Print the time taken
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    Tensor<float> t3{100, 100, 100};
    Tensor<float> t4{100, 100, 100};

    // Fill the tensors with 2 and 3
    t3.fill(2);
    t4.fill(3);

    // Start timer
    start = std::chrono::high_resolution_clock::now();

    // Perform 1000 multiplications
    for (int i = 0; i < 10; ++i) {
        t3 = t3 + t4;
    }

    // Stop timer
    stop = std::chrono::high_resolution_clock::now();

    // Print the time taken
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
    return 0;
}
