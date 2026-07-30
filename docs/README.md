# BackpropLab Documentation

BackpropLab is an experimental CPU-based C++17 project for studying custom
tensors, neural-network internals, manual backpropagation, optimizers,
attention, and Transformer-style model composition.

Start with the root [README](../README.md), then use:

- [Architecture](ARCHITECTURE.md) for the tensor, layer, optimizer, model,
  serialization, and build design;
- [Roadmap](ROADMAP.md) for prioritized correctness, testing, documentation,
  and optional future work;
- [Building and Running with CMake](BUILD_WITH_CMAKE.md) for Linux, Windows,
  and macOS setup instructions;
- [MIT License](../LICENSE) for licensing terms.

The current implementation is CPU-based. CUDA execution, Python bindings,
general automatic differentiation, production inference, and large-scale
training are not implemented.
