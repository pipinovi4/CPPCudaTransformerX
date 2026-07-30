# Architecture

## Overview

This repository is an experimental CPU-based neural-network framework written
in C++17. Its main purpose is to explore how multidimensional tensors,
neural-network layers, manually implemented backward passes, optimizers, and
Transformer-style components can be built below high-level ML frameworks.

The implementation is organized around a custom `Tensor<T>` type. Layers
consume and return tensors, cache the values needed by their backward passes,
and expose trainable parameters and their corresponding gradients to an
optimizer.

The current project is:

- a C++ tensor and neural-network experimentation codebase;
- a manually differentiated training pipeline;
- a collection of model and dataset examples;
- a foundation for studying numerical and ML systems engineering.

The current project is not:

- a CUDA or GPU runtime;
- a Python extension module;
- a general automatic-differentiation engine;
- a production inference service;
- a validated large-scale Transformer implementation.

## High-level data flow

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
  Loss gradient
        |
        v
 Manual layer backward passes
        |
        v
 Parameter and gradient tensors
        |
        v
 SGD / Adam / RMSprop
```

There is no runtime-generated computation graph. Model classes explicitly call
their layers in forward order and call `backward()` in reverse order.

## Repository layout

```text
.
├── include/       Public headers for tensors and neural-network components
├── src/           Template implementations and executable entry points
├── models/        Model compositions
├── examples/      Standalone training and integration examples
├── tests/         GoogleTest unit tests
├── utils/         C++ dataset and vocabulary loaders
├── utils_py/      Python dataset preparation scripts
├── notebooks/     Python baseline experiments
├── docs/          Architecture, roadmap, build guide, and license
├── CMakeLists.txt
├── CMakePresets.json
└── conanfile.txt
```

The core APIs are C++ templates. Their implementations therefore live mostly
in `.tpp` files included by the corresponding public headers.

## Core tensor subsystem

### Storage model

`Tensor<T>` is defined in `include/Tensor.h` and implemented in
`src/Tensor.tpp`.

Each tensor owns:

- a flat `std::vector<T>` containing its elements;
- a vector of integer dimensions;
- calculated strides used for multidimensional indexing.

Data is stored contiguously. Operations calculate offsets into the flat vector
rather than building a hierarchy of nested containers.

The data and dimension members are currently public. This simplifies
experimentation but allows callers to violate tensor invariants. Encapsulation
and stronger invariant validation are future work.

### Construction and indexing

The tensor can be created from:

- a shape initialized with zeros;
- a shape and flat data;
- nested `std::vector` values;
- initializer-list dimensions;
- a reserve-oriented constructor used by some internal components.

The API supports element access through index vectors, `get`, `set`, function
call operators, and slice-style operators.

### Tensor operations

Implemented operations include:

- element-wise tensor and scalar arithmetic;
- limited broadcasting;
- `sum`, `mean`, `argmax`, and `softmax`;
- slicing and concatenation;
- `reshape`, `transpose`, `squeeze`, and dimension expansion;
- zero, one, and uniform initialization;
- lower and upper triangular operations;
- matrix and limited batched dot products;
- text-stream serialization and deserialization.

### Matrix operations

`Tensor<T>::dot()` maps contiguous tensor memory into Eigen matrices. The
operation validates the inner matrix dimensions, calculates the result shape,
and executes matrix multiplication through `Eigen::Map`.

Selected tensor and layer loops contain OpenMP and SIMD pragmas. These are CPU
parallelism experiments rather than verified performance guarantees. OpenMP is
enabled by CMake only when a compatible implementation is found.

### Shape semantics

Shape behavior is implemented independently by each tensor operation. The
project does not yet provide a centralized shape-inference subsystem.

Consequences of the current design:

- higher-rank broadcasting behavior requires additional validation;
- not every operation supports the same set of batch dimensions;
- shape errors are generally reported with `std::invalid_argument`;
- bounds and overflow validation are incomplete.

## Neural-network component model

### Layer interface

`include/Layer.h` defines the common layer abstraction:

```cpp
forward(input)
backward(gradient)
parameters()
gradients()
```

Trainable model components expose `Tensor<T>` references through
`parameters()` and matching gradient references through `gradients()`.
Optimizers receive these collections and update the parameter data directly.

The interface and some implementations currently contain type and ownership
inconsistencies. For example, parts of the abstraction use `Tensor<float>`
despite being templated, and several activation objects are passed through raw
pointers. These are known design issues rather than intentional stable API
contracts.

### Dense layer

`DenseLayer<T>` implements:

1. Xavier/Glorot weight initialization;
2. an affine transformation;
3. activation application;
4. cached input storage;
5. manually calculated weight, bias, and input gradients.

Selected inner loops use OpenMP and SIMD pragmas.

### Embedding

`Embedding<T>` maps token IDs to rows of a trainable embedding matrix. Its
backward pass accumulates incoming gradients into the corresponding embedding
rows.

This implementation is experimental. Token-index validation, gradient reset
behavior, generic type consistency, and the current accumulation loop require
additional correctness work.

### Activation functions

The activation subsystem implements:

- Linear;
- Sigmoid;
- Softmax;
- ReLU;
- Leaky ReLU;
- ELU;
- Tanh.

Each activation provides an explicit forward transformation and backward
gradient transformation. Activation implementations do not participate in a
general computation graph.

### Loss functions

The loss subsystem implements:

- binary cross-entropy;
- cross-entropy;
- mean squared error;
- mean absolute error.

Each loss supplies a scalar forward result and a manually calculated gradient
with respect to its predictions.

### Layer normalization

`LayerNorm<T>` contains gamma and beta tensors and implements forward and
backward calculations.

Its current `parameters()` and `gradients()` interface returns empty
collections, so normalization parameters are not included in generic optimizer
or model-checkpoint traversal. This must be corrected before the full model
training path can be considered complete.

### Residual blocks

`ResidualBlock<T, D>` composes a processing layer with a residual connection
and layer normalization. It is used to assemble attention and feed-forward
components in the experimental Transformer.

Residual forward and backward behavior requires stronger integration and
numerical-gradient tests. The existing residual-block tests are currently
disabled in the source.

### Position-wise feed-forward layer

`PositionalWiseDenseLayer<T>` implements a two-stage feed-forward transformation
with an activation between its projections. It provides parameter and gradient
collections for optimizer integration.

## Multi-head attention

`MultiHeadAttention<T>` implements an experimental self-attention pipeline:

```text
Input
 ├── Wq + bq ──> Queries ─┐
 ├── Wk + bk ──> Keys    ├── Split heads
 └── Wv + bv ──> Values  ┘
                              |
                              v
                 scaled Q × transpose(K)
                              |
                        optional mask
                              |
                           softmax
                              |
                              v
                     attention × Values
                              |
                        concatenate heads
                              |
                          Wo + bo
                              |
                            Output
```

The class caches input, projected heads, and attention outputs for its manual
backward pass. It exposes query, key, value, output projection parameters, and
their biases.

The current implementation should be treated as experimental because:

- the backward equations need numerical-gradient verification;
- a parallel `push_back` into the attention-head collection is unsafe;
- activation state is shared between parallel head calculations;
- mask dimensions and semantics are not fully validated;
- standard encoder-decoder cross-attention is not implemented.

No attention performance or model-quality claims should be made until these
issues are addressed and verified.

## Optimizer subsystem

The optimizer abstraction accepts parameter and gradient tensor references.
The implemented algorithms are:

- SGD;
- Adam;
- RMSprop.

Additional behavior includes:

- step-decay learning-rate scheduling;
- exponential-decay learning-rate scheduling;
- weight decay;
- value-based gradient clipping;
- experimental optimizer-state save/load methods.

Optimizer state persistence is not yet covered by round-trip tests and is not
part of a versioned checkpoint format.

## Model compositions

### DigitRecognizer

`models/DigitRecognizer.*` defines a three-layer fully connected classifier:

```text
784 inputs
    |
Dense + ReLU
    |
Dense + ReLU
    |
Dense + Softmax
    |
10 outputs
```

Its training loop:

1. slices MNIST samples from an input tensor;
2. creates one-hot targets;
3. executes the model forward pass;
4. calculates loss and accuracy for the current batch;
5. runs manual backward passes;
6. accumulates and averages gradients;
7. invokes an optimizer.

This is the clearest end-to-end training example in the repository. Published
accuracy values should only be added after a clean, reproducible run.

### EmbeddingModel

`EmbeddingModel<T>` wraps the embedding layer and a sigmoid activation. It is
used by a synthetic-data example.

The current backward ordering applies embedding backward before the sigmoid
backward transformation. It requires correction and validation before being
presented as a verified training example.

### MultiHeadAttentionModel

`MultiHeadAttentionModel<T>` combines multi-head attention with dense layers
and is exercised with synthetic tensors.

Its parameter and gradient traversal currently duplicates attention parameters
and omits part of the dense-layer composition. It is an integration experiment,
not a validated model.

### Transformer

`models/Transformer.*` assembles:

- a token embedding;
- tokenizer/vocabulary handling;
- encoder-style residual attention and feed-forward blocks;
- decoder-style residual blocks;
- an output dense layer with softmax;
- explicit training and greedy generation loops;
- raw model-weight loading and saving.

The high-level forward flow is:

```text
Source IDs ──> Shared embedding ──> Encoder blocks ─────────────┐
                                                               |
Target IDs ──> Shared embedding ──> Output-attention block      |
                                    ──> Decoder blocks          |
                                    ──> Vocabulary projection ──┘
```

The class demonstrates model composition, but it is not yet a standard or
validated Transformer implementation:

- there is no separate positional-encoding implementation;
- the object named `positional_encoder_` is a tokenizer;
- standard encoder-decoder cross-attention is absent;
- source embeddings are currently passed through an attention mask interface;
- the number of layer blocks is not configurable as a stack;
- embedding and LayerNorm parameters are omitted from generic parameter
  traversal;
- training and generation contain boundary and shape risks;
- dropout is stored as configuration but is not applied;
- no full Transformer integration or convergence tests exist.

The training and generation executables should therefore be presented as
experimental code paths.

## Training flow

The general model-training pattern is:

```cpp
Tensor<float> predictions = model.forward(input);
float loss = loss_function.forward(predictions, targets);
Tensor<float> gradient = loss_function.backward(predictions, targets);

model.backward(gradient);
optimizer.update(model.parameters(), model.gradients(), epoch);
```

Unlike an autograd framework, this flow depends on every model and layer
explicitly propagating the incoming gradient in the correct order.

## Inference and generation

Inference is represented by direct model `forward()` calls.

The experimental Transformer generation path:

1. converts tokens into tensor IDs;
2. inserts a limited number of context tokens;
3. executes the Transformer forward pass;
4. selects the next token using `argmax`;
5. appends the prediction;
6. stops at the end-of-sequence token or maximum length.

It is greedy generation only. Sampling, beam search, KV caching, batched
serving, model export, and production inference APIs are not implemented.

## Tokenization and BPE

`Tokenizer<T>` provides:

- lowercase text normalization;
- whitespace-oriented tokenization;
- vocabulary construction;
- text-to-ID and ID-to-text conversion;
- padding and truncation;
- vocabulary save/load.

`BPE` provides subword merge calculation and application. It is separate from
the Python SentencePiece-based vocabulary preparation scripts.

The project does not define a custom programming language, parser, DSL, or
interpreter.

## Data subsystem

### C++ loaders

Utilities under `utils/` load:

- MNIST image and label data;
- AG News text;
- WikiText splits;
- vocabulary files.

The loaders currently rely on relative paths selected by the executable entry
points. A future configuration layer should make dataset and checkpoint paths
explicit command-line arguments.

### Python preparation scripts

Scripts under `utils_py/` download, extract, normalize, and rewrite datasets.
Python is used as an offline data-preparation tool.

There is no importable Python package and no pybind11 binding to the C++
runtime.

### Notebooks

The notebooks are baseline and exploratory Python experiments. They are not
the public API of the C++ framework and should be documented separately from
the native runtime.

## Serialization

There are several independent persistence mechanisms:

### Tensor serialization

`Tensor<T>::serialize()` and `deserialize()` use a custom text representation
containing dimensions and values.

### Transformer weights

The Transformer reads and writes parameter arrays as raw binary data in the
order returned by `parameters()`.

The format currently has no:

- file signature;
- format version;
- tensor names;
- stored shapes;
- stored dtype;
- model hyperparameters;
- vocabulary identity;
- integrity checksum.

Loading therefore requires an identical model configuration and parameter
ordering.

### Optimizer state

Optimizer classes include experimental state-persistence methods. Their format
is not integrated with Transformer model weights and has no verified
round-trip tests.

### Vocabulary

Tokenizer vocabulary persistence is separate from model and optimizer state.

A unified, versioned checkpoint design is a high-priority roadmap item.

## Build architecture

The CMake project defines:

- `backprop_lab`, an interface target containing include paths and
  dependency requirements;
- application targets for the main demonstration, training, and generation;
- optional example targets;
- an optional GoogleTest target registered with CTest.

Dependencies:

- Eigen 3.4 for matrix operations, supplied by Conan or the system;
- GoogleTest for C++ tests;
- OpenMP when available;
- Conan as the recommended dependency provider.

The build contains no CUDA language, CUDA target, `.cu` source, or GPU
dependency.

See `docs/BUILD_WITH_CMAKE.md` for setup and execution instructions.

## Test architecture

The repository contains GoogleTest suites for:

- tensor operations;
- activations;
- loss functions;
- dense layers;
- embeddings;
- tokenization and BPE;
- multi-head attention;
- layer normalization;
- position-wise feed-forward layers;
- optimizers and learning-rate schedules.

Current test limitations:

- no CI result is currently documented;
- no numerical finite-difference gradient checks;
- no Transformer integration tests;
- no checkpoint round-trip tests;
- residual-block tests are disabled;
- no sanitizer or coverage pipeline is documented;
- some tests verify execution or shape more strongly than numerical
  correctness.

The existence of test sources should not be presented as a successful current
build until the suite has been run in a clean environment.

## Numerical and runtime limitations

The current architecture has several cross-cutting limitations:

- tensor dimensions use `int`, limiting safe representation of very large
  shapes;
- multiplication of dimensions is not comprehensively overflow-checked;
- tensor invariants can be bypassed through public fields;
- raw pointers make ownership and copying rules unclear in some layers;
- random initialization is not uniformly seed-controlled;
- forward/backward caches assume a simple sequential execution model;
- thread safety is not guaranteed;
- error messages and shape contracts are inconsistent;
- there is no device abstraction;
- there is no stable ABI or public API compatibility policy.

These limitations are acceptable for an experimental learning project but must
be addressed before describing the code as a reusable production library.

## Intended positioning

The strongest accurate description is:

> An experimental C++17 tensor and neural-network internals project featuring
> a custom multidimensional `Tensor<T>`, manually implemented forward and
> backward passes, Eigen-based matrix operations, CPU parallelism experiments,
> optimizers, MNIST classification, attention, and Transformer-style model
> composition.

This positioning highlights the project’s systems and numerical-engineering
value without claiming unimplemented GPU, Python, autograd, or production
capabilities.
