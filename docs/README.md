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
- `notebooks/`: Jupyter notebooks for experimenting with the transformer model.
- `utils/`: Utility scripts for data preprocessing and model evaluation.
- `CMakeLists.txt`: Build configuration.

## Getting Started
### Prerequisites
- C++17 or higher
- Python 3.6 or higher
- NVIDIA CUDA Toolkit
- CMake
- Conan (C++ package manager)
- Pybind11
- GoogleTest

### Build
1. Clone the repository:
   ```bash
   git clone https://github.com/pipinovi4/C++CudaTransformerX
   cd C++CudaTransformerX
   ```
2. Create a build directory:
   ```bash
   make build
    ```
3. Run tests:
   ```bash
   make test
   ```
4. Create a venv and install dependencies:
   ```bash
   make venv
   ```
   