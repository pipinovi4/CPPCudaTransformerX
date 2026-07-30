# BackpropLab

> Custom tensors, manual backpropagation, and neural-network internals in
> C++17.

BackpropLab is an experimental CPU-based neural-network project for exploring
how tensor operations, forward and backward passes, optimizers, attention, and
Transformer-style model components work below high-level ML frameworks.

The project implements a custom multidimensional `Tensor<T>`, explicit layer
gradients, Eigen-based matrix operations, and end-to-end C++ examples for
MNIST, embeddings, multi-head attention, training, and greedy text generation.

> [!IMPORTANT]
> BackpropLab is an educational and portfolio project, not a production ML
> framework. CUDA execution, Python bindings, general automatic
> differentiation, production inference, and large-scale training are not
> implemented.

## Highlights

- Modern C++17 and template-based APIs
- Custom contiguous multidimensional `Tensor<T>`
- Shape transformations, reductions, slicing, broadcasting, and dot products
- Eigen-backed matrix multiplication
- OpenMP and SIMD experimentation
- Manually implemented forward and backward passes
- Dense, embedding, normalization, residual, feed-forward, and attention
  components
- Binary cross-entropy, cross-entropy, MSE, and MAE losses
- SGD, Adam, and RMSprop optimizers
- Step and exponential learning-rate schedules
- Gradient clipping and weight decay
- MNIST classification example
- Tokenization, vocabulary handling, and BPE
- Experimental Transformer-style training and greedy generation
- GoogleTest unit-test source

## Current status

| Area | Status |
| --- | --- |
| Custom tensor operations | Implemented; additional edge-case validation planned |
| Dense layers and losses | Implemented with manual backward passes |
| SGD, Adam, and RMSprop | Implemented |
| Embedding | Experimental; gradient correctness work remains |
| Layer normalization and residual blocks | Experimental |
| Multi-head attention | Experimental; backward validation remains |
| MNIST model | End-to-end example |
| Transformer-style model | Experimental |
| Serialization | Basic and architecture-dependent |
| CUDA/GPU backend | Not implemented |
| Python bindings | Not implemented |
| General autograd | Not implemented |

## Architecture

```text
Dataset or token IDs
        |
        v
    Tensor<T>
        |
        v
 Layer forward passes
        |
        v
 Model predictions
        |
        v
   Loss function
        |
        v
 Manual backward passes
        |
        v
 Parameter and gradient tensors
        |
        v
 SGD / Adam / RMSprop
```

BackpropLab does not construct a dynamic computation graph. Each layer caches
the values required by its own `backward()` method, and model classes explicitly
propagate gradients through their components.

See [Architecture](docs/ARCHITECTURE.md) for the detailed component design and
known limitations.

## Tensor operations

`Tensor<T>` is backed by contiguous `std::vector<T>` storage and records its
dimensions and strides explicitly.

The API includes:

- creation from shapes, flat vectors, and nested vectors;
- multidimensional element access;
- element-wise tensor and scalar arithmetic;
- limited broadcasting;
- `sum`, `mean`, `argmax`, and `softmax`;
- slicing and concatenation;
- reshape, transpose, squeeze, and dimension expansion;
- zero, one, and uniform initialization;
- triangular operations;
- matrix and limited batched dot products;
- stream serialization and deserialization.

## Neural-network components

Implemented component types include:

- `DenseLayer<T>`
- `Embedding<T>`
- `MultiHeadAttention<T>`
- `LayerNorm<T>`
- `ResidualBlock<T, D>`
- `PositionalWiseDenseLayer<T>`

Activation functions:

- Linear
- Sigmoid
- Softmax
- ReLU
- Leaky ReLU
- ELU
- Tanh

Loss functions:

- Binary cross-entropy
- Cross-entropy
- Mean squared error
- Mean absolute error

## Repository structure

```text
.
├── include/       Public tensor and neural-network headers
├── src/           Template implementations and application entry points
├── models/        Model compositions
├── examples/      Standalone model examples
├── tests/         GoogleTest unit tests
├── utils/         C++ dataset and vocabulary loaders
├── utils_py/      Python dataset preparation scripts
├── notebooks/     Python baseline experiments
├── docs/          Architecture, roadmap, build guide, and license
├── CMakeLists.txt
├── CMakePresets.json
└── conanfile.txt
```

Most implementations use `.tpp` files because the public APIs are C++
templates.

## Requirements

- Git
- Python 3
- Conan 2
- CMake 3.15 or newer
- Ninja or another CMake-supported generator
- A C++17 compiler
- OpenMP-capable compiler recommended

The C++ dependencies declared through Conan are Eigen 3.4 and GoogleTest 1.15.
CUDA is not required.

## Quick start

### 1. Clone

```bash
git clone https://github.com/<your-github-username>/backprop-lab-cpp.git
cd backprop-lab-cpp
```

Replace `<your-github-username>` with the final repository owner.

### 2. Install dependencies

```bash
conan profile detect --force
conan install . \
  --output-folder=build \
  --build=missing \
  -s build_type=Release
```

### 3. Configure and build

```bash
cmake --preset conan-release
cmake --build --preset conan-release --parallel
```

### 4. Run tests

```bash
ctest --preset conan-release
```

Complete Linux, Windows, and macOS instructions are available in
[Building and Running with CMake](docs/BUILD_WITH_CMAKE.md).

## CMake targets

| Target | Purpose |
| --- | --- |
| `backprop_lab` | Header-only interface target |
| `backprop_lab_cli` | General demonstration executable |
| `global_tests` | GoogleTest suite |
| `digit_recognizer` | MNIST classifier example |
| `embedding_model` | Embedding experiment |
| `multi_head_attention_model` | Attention experiment |
| `train` | Experimental Transformer training |
| `generate` | Experimental greedy text generation |

## Preparing datasets

Create and activate a Python virtual environment, then install the data-tool
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

To run the optional Python baseline notebooks, install their additional direct
dependencies:

```bash
python -m pip install -r requirements-notebooks.txt
```

Prepare the datasets:

```bash
python utils_py/main.py
```

Generated data is stored under `data/`, which is excluded from version control.
The preparation scripts cover MNIST, AG News, WikiText, and vocabulary data.

## Running examples

The current example programs use dataset paths relative to the build directory.
After preparing the data:

```bash
cd build

./backprop_lab_cli
./global_tests
./digit_recognizer
./embedding_model
./multi_head_attention_model
./train
./generate
```

Visual Studio normally places Release executables under `build/Release/`.

### MNIST

`digit_recognizer` loads the prepared MNIST data, constructs a three-layer
fully connected classifier, executes explicit forward and backward passes, and
updates parameters with Adam.

No accuracy value is published here because a clean reproducible result has
not yet been recorded for the current revision.

### Embeddings and attention

`embedding_model` and `multi_head_attention_model` exercise their corresponding
components with synthetic data. Both should be treated as implementation
experiments rather than model-quality demonstrations.

### Transformer-style experiment

`train` and `generate` demonstrate how tokenization, embeddings, attention,
feed-forward layers, residual connections, normalization, losses, optimizers,
weight persistence, and greedy token selection can be composed.

This path is under active correctness work and is not a standard, validated
Transformer implementation.

## Python baseline notebooks

The notebooks under `notebooks/` are optional PyTorch baselines and exploratory
comparisons. They are not the Python API of BackpropLab and their outputs are
not evidence of C++ runtime performance or model quality.

Some notebooks contain saved historical outputs and environment-specific
kernel metadata. Re-run and clean a notebook before citing any result from it
in GitHub, Upwork, or CV material.

## C++ usage example

```cpp
#include "include/Tensor.h"

Tensor<float> a(
    {2, 3},
    std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }
);

Tensor<float> transposed = a.transpose({1, 0});
Tensor<float> product = a.dot(transposed);
Tensor<float> probabilities = product.softmax(-1);

probabilities.print();
```

Trainable components expose their state explicitly:

```cpp
auto parameters = model.parameters();
auto gradients = model.gradients();

optimizer.update(parameters, gradients, epoch);
```

## Serialization

The project currently contains:

- text-stream tensor serialization;
- tokenizer vocabulary save/load;
- raw Transformer parameter save/load;
- experimental optimizer-state persistence.

The model checkpoint format is not versioned and assumes an identical
architecture and parameter order. It should not be considered portable or
stable yet.

## Known limitations

- CPU execution only
- No CUDA kernels or device abstraction
- No pybind11 or importable Python API
- No dynamic computation graph or general autograd
- No custom DSL or interpreter
- Experimental attention backward implementation
- Experimental Transformer architecture and training loop
- No stable checkpoint format
- No published first-party benchmark results
- No verified large-scale training
- No stable ABI or API compatibility guarantee

## Roadmap

The immediate priorities are:

1. numerical correctness and bounds safety;
2. finite-difference gradient checks;
3. reproducible builds and CI;
4. attention and Transformer corrections;
5. versioned checkpoints;
6. first-party CPU benchmarks;
7. portfolio media based only on verified runs.

See the complete [Roadmap](docs/ROADMAP.md).

CUDA, Python bindings, and general automatic differentiation are optional
long-term directions, not promised current functionality.

## Portfolio relevance

BackpropLab demonstrates experience with:

- modern C++ and template-based APIs;
- multidimensional data structures;
- numerical and linear-algebra integration;
- neural-network internals;
- manual gradient implementation;
- optimizer algorithms;
- CPU parallelism concepts;
- serialization and data pipelines;
- unit testing of numerical code;
- CMake and Conan build configuration.

## Contributing

Focused bug reports and pull requests are welcome, particularly for numerical
correctness, memory safety, build portability, gradient verification,
documentation, and tests.

When reporting a problem, include:

- operating system;
- compiler and version;
- build type;
- tensor shapes;
- minimal reproduction;
- complete error output.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)
- [CMake build and run guide](docs/BUILD_WITH_CMAKE.md)
- [License](LICENSE)

## License

BackpropLab is available under the [MIT License](LICENSE).

## Author

**Mykita Bozhenko (Pipin)**

Backend and machine-learning engineer focused on C++, numerical systems, ML
infrastructure, and low-level implementations of deep-learning components.
