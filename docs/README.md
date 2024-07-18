# C++CudaTransformerX

C++CudaTransformerX is a high-performance implementation of a Transformer model written from scratch in C++. This project leverages NVIDIA CUDA for GPU support and mixed precision to handle large-scale models with up to 1.5 billion parameters, similar to GPT-2.

## Features
- Written in C++ for performance.
- Written from scratch without external libraries.
- Written in a modular and extensible manner.
- GPU acceleration using NVIDIA CUDA.
- Mixed precision for efficient memory usage.
- Custom Tensor class with GPU support.

## Project Structure
- `include/`: Header files for tensors, transformers, and CUDA utilities.
- `src/`: Source files for the main application, tensor operations, transformer logic, and CUDA utilities.
- `tests/`: Unit tests for validating tensor operations and transformer functionality.
- `data/`: Directory for storing datasets.
- `docs/`: Documentation files.
- `CMakeLists.txt`: Build configuration.

## Getting Started
### Prerequisites
- C++17 or higher
- NVIDIA CUDA Toolkit
- CMake

### Build
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/C++CudaTransformerX.git
   cd C++CudaTransformerX
   gcc -o main main.cpp
    ```

