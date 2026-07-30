# Roadmap

## Purpose

This roadmap describes the work required to turn the current experimental
C++17 tensor and neural-network codebase into a reliable, professionally
presented open-source portfolio project.

It separates:

- features that already exist in source;
- features that require correction or verification;
- optional future directions.

Items are prioritized by engineering risk and portfolio value rather than by
estimated completion time.

## Current baseline

The repository currently contains:

- a custom multidimensional `Tensor<T>`;
- element-wise operations, reductions, shape transformations, and dot products;
- Eigen-backed matrix multiplication;
- OpenMP and SIMD experimentation;
- dense, embedding, normalization, residual, feed-forward, and attention
  components;
- manually implemented layer and loss backward passes;
- binary cross-entropy, cross-entropy, MSE, and MAE losses;
- SGD, Adam, and RMSprop;
- learning-rate schedules, weight decay, and gradient clipping;
- tokenization, vocabulary handling, and BPE;
- MNIST, embedding, and attention examples;
- experimental Transformer training and greedy generation;
- tensor, vocabulary, model-weight, and optimizer-state persistence code;
- GoogleTest source files;
- Conan and CMake build configuration;
- Python scripts for offline dataset preparation.

The following are not current features:

- CUDA or GPU execution;
- pybind11 or another Python binding layer;
- general automatic differentiation;
- production inference or model serving;
- demonstrated large-scale training;
- a custom DSL or interpreter;
- published performance benchmarks.

## Priority definitions

| Priority | Meaning |
| --- | --- |
| P0 | Correctness, memory safety, or reproducibility blocker |
| P1 | Required for a credible public portfolio release |
| P2 | Important engineering improvement after the foundation is stable |
| P3 | Optional long-term direction |

## P0 — Correctness and memory safety

### Tensor invariants

- [ ] Validate that all dimensions are non-negative.
- [ ] Detect overflow when calculating total tensor size and byte counts.
- [ ] Define consistent behavior for empty and scalar tensors.
- [ ] Add bounds validation to every public indexing path.
- [ ] Prevent callers from making `data`, `dimensions`, and strides
      inconsistent.
- [ ] Replace the reserve-oriented tensor constructor with explicit,
      invariant-preserving APIs.
- [ ] Document supported broadcasting rules.
- [ ] Add tests for invalid shapes, incompatible broadcasting, empty tensors,
      and out-of-range indices.

### Dense layers and ownership

- [ ] Replace ambiguous raw activation pointers with explicit ownership or
      non-owning references.
- [ ] Define copy and move behavior for layers containing activation objects.
- [ ] Make `Layer<T>::forward()` consistently return `Tensor<T>`.
- [ ] Validate the complete input shape rather than relying only on flat data.
- [ ] Add deterministic numerical-gradient tests for weights, biases, and input
      gradients.

### Embedding

- [ ] Remove the redundant vocabulary-wide loop in the embedding backward pass.
- [ ] Define whether gradients accumulate or reset on every backward call.
- [ ] Validate every token ID before accessing the embedding table.
- [ ] Return `Tensor<T>` parameter and gradient references rather than
      hard-coded `Tensor<float>` references.
- [ ] Correct and test activation/backward ordering in `EmbeddingModel`.
- [ ] Add repeated-token and invalid-token gradient tests.

### Multi-head attention

- [ ] Remove the parallel `push_back` data race.
- [ ] Preserve deterministic head ordering under parallel execution.
- [ ] Avoid sharing mutable activation cache state between parallel heads.
- [ ] Re-derive and correct gradients for queries, keys, values, and attention
      probabilities.
- [ ] Add finite-difference checks for every projection matrix and bias.
- [ ] Validate input, head, mask, and output dimensions.
- [ ] Define causal-mask semantics independently of source representations.
- [ ] Add batch-dimension tests.
- [ ] Run attention tests under ThreadSanitizer where supported.

### Normalization and residual connections

- [ ] Include LayerNorm gamma and beta in `parameters()`.
- [ ] Include gamma and beta gradients in `gradients()`.
- [ ] Remove nondeterministic gradient noise from the default backward path.
- [ ] Verify the LayerNorm backward equation with finite differences.
- [ ] Re-enable and repair residual-block tests.
- [ ] Verify residual gradient addition and backward ordering.

### Model parameter traversal

- [ ] Include embedding parameters and gradients in the Transformer.
- [ ] Include LayerNorm parameters and gradients in every model.
- [ ] Remove duplicated attention parameters from
      `MultiHeadAttentionModel::parameters()`.
- [ ] Add missing dense-layer parameters to `MultiHeadAttentionModel`.
- [ ] Assert that parameter and gradient lists have matching lengths and shapes.
- [ ] Add a named-parameter representation to make traversal auditable.

### Transformer safety

- [ ] Fix `j + 1` out-of-bounds writes in training and generation loops.
- [ ] Validate `context_tokens_size`, `max_len`, vocabulary size, and special
      token availability.
- [ ] Separate source tokens, target tokens, attention masks, and encoded
      representations.
- [ ] Replace the current source-as-mask behavior.
- [ ] Define output tensor shape and token-axis semantics consistently.
- [ ] Add a tiny deterministic forward-pass test.
- [ ] Add a tiny dataset overfitting test before presenting training as
      functional.

## P0 — Reproducible build and test baseline

- [x] Remove the developer-specific absolute path from `CMakePresets.json`.
- [x] Replace duplicated CMake source entries with reusable targets.
- [x] Add CTest and GoogleTest discovery.
- [x] Make OpenMP optional and dependency-driven.
- [x] Add a portable CMake build guide.
- [ ] Verify a fresh Conan/CMake build on Linux.
- [ ] Verify a fresh Conan/CMake build on Windows.
- [ ] Verify a fresh Conan/CMake build on macOS or document it as unsupported.
- [x] Use Eigen as a Conan/system dependency instead of vendoring its source.
- [ ] Remove the tracked extensionless compiled `main` binary.
- [ ] Update or remove the legacy Makefile so it cannot invoke stale target
      names.
- [ ] Add Debug, Release, and sanitizer presets.
- [ ] Record the first verified build commands and compiler versions.

## P1 — Automated verification

### Continuous integration

- [ ] Add a GitHub Actions workflow for Linux Debug and Release builds.
- [ ] Add a Windows build job.
- [ ] Run CTest with failure output enabled.
- [ ] Add AddressSanitizer and UndefinedBehaviorSanitizer jobs.
- [ ] Add ThreadSanitizer for supported Linux configurations.
- [ ] Cache Conan dependencies without caching generated project outputs.
- [ ] Add a visible build-status badge only after CI passes.

### Test quality

- [ ] Inventory active, disabled, and assertion-free tests.
- [ ] Replace execution-only tests with precise expected-value checks.
- [ ] Use fixed random seeds in numerical tests.
- [ ] Add finite-difference gradient utilities shared across layer tests.
- [ ] Add tensor serialization round-trip tests.
- [ ] Add optimizer-state round-trip tests.
- [ ] Add malformed and truncated checkpoint tests.
- [ ] Add integration tests for MNIST-shaped batches without downloading MNIST.
- [ ] Add Transformer forward and mask tests.
- [ ] Add regression tests for every P0 defect.

### Static analysis and formatting

- [ ] Add `.clang-format`.
- [ ] Add `.clang-tidy`.
- [ ] Enable useful compiler warnings for project targets.
- [ ] Treat warnings as errors in at least one CI job.
- [ ] Add include-order and unused-code cleanup.
- [ ] Document the supported compiler/version matrix.

## P1 — Professional open-source presentation

### Naming

- [ ] Select the final project name.
- [ ] Rename the GitHub repository.
- [ ] Update README, CMake project, targets, documentation, and source comments
      together.
- [ ] Update local and published clone URLs.
- [ ] Avoid CUDA, GPU, autograd, Python-binding, production, and scale claims
      unless corresponding implementations are added and verified.

### Repository hygiene

- [x] Move the MIT license to a root-level `LICENSE`.
- [ ] Add `CONTRIBUTING.md`.
- [ ] Add `CODE_OF_CONDUCT.md`.
- [ ] Add `SECURITY.md`.
- [ ] Add issue and pull-request templates.
- [ ] Remove committed IDE metadata and notebook checkpoints.
- [ ] Reduce the Python requirements file to direct project dependencies.
- [ ] Document dataset sources and licenses.
- [ ] Add a release/versioning policy.

### Documentation

- [x] Add an accurate root README.
- [x] Add portable CMake instructions.
- [x] Replace aspirational architecture documentation with current
      implementation details.
- [x] Replace timeline estimates with a risk-prioritized roadmap.
- [ ] Add a minimal guaranteed-working C++ example.
- [ ] Add an API reference for `Tensor<T>` and layer interfaces.
- [ ] Add shape examples for every tensor operation.
- [ ] Document model input/output shapes.
- [ ] Document error handling and unsupported operations.
- [ ] Add verified terminal output only after clean execution.
- [ ] Add an architecture image suitable for GitHub and Upwork.

## P1 — Checkpoint and serialization redesign

- [ ] Define one versioned checkpoint container.
- [ ] Add a magic header and format version.
- [ ] Store model type and hyperparameters.
- [ ] Store named tensor entries.
- [ ] Store shape and dtype per tensor.
- [ ] Store vocabulary identity or content.
- [ ] Store optimizer type, step, schedules, and moment tensors.
- [ ] Detect architecture and dtype mismatches before reading tensor bytes.
- [ ] Validate complete reads and reject trailing or truncated data.
- [ ] Write checkpoints atomically through a temporary file and rename.
- [ ] Document backward-compatibility expectations.
- [ ] Add complete model/optimizer/vocabulary round-trip tests.

## P2 — Tensor API and performance engineering

### API design

- [ ] Introduce a `Shape` or dimension type using `std::size_t` where
      appropriate.
- [ ] Centralize stride and shape inference.
- [ ] Separate tensor views from owning tensors.
- [ ] Add const-correct data access.
- [ ] Define row-major storage explicitly.
- [ ] Replace repeated allocations with reusable buffers where safe.
- [ ] Reduce unnecessary tensor copies in forward and backward passes.
- [ ] Document aliasing guarantees.

### Numerical behavior

- [ ] Use stable formulations for softmax and cross-entropy consistently.
- [ ] Define epsilon behavior for divisions and logarithms.
- [ ] Add NaN and infinity handling tests.
- [ ] Add numerical comparisons with documented tolerances.
- [ ] Test both `float` and `double` template instantiations.
- [ ] Decide whether the custom `float_16` type should be completed or removed
      from the public surface.

### First-party benchmarks

- [ ] Add a dedicated benchmark target.
- [ ] Benchmark element-wise operations.
- [ ] Benchmark reshape/transpose and identify copy costs.
- [ ] Benchmark matrix multiplication across representative shapes.
- [ ] Benchmark attention forward and backward passes.
- [ ] Measure single-threaded and OpenMP configurations separately.
- [ ] Record CPU, compiler, flags, thread count, warmup, repetitions, and
      variance.
- [ ] Compare only equivalent operations and shapes.
- [ ] Publish results only when they are reproducible.

Dependency benchmarks are not first-party BackpropLab benchmarks and must not
be presented as such.

## P2 — Model architecture improvements

### MNIST

- [ ] Add command-line configuration for epochs, batch size, and data paths.
- [ ] Add deterministic shuffling.
- [ ] Separate training and evaluation modes.
- [ ] Add checkpoint save/load for `DigitRecognizer`.
- [ ] Run and document a reproducible baseline.
- [ ] Publish accuracy only with dataset split, seed, configuration, and commit
      hash.

### Attention model

- [ ] Correct parameter traversal.
- [ ] Define a meaningful supervised objective.
- [ ] Replace oversized synthetic allocations with configurable fixtures.
- [ ] Add a deterministic smoke-training test.

### Transformer-style model

- [ ] Implement sinusoidal or learned positional encoding.
- [ ] Implement a correct causal self-attention mask.
- [ ] Implement separate encoder-decoder cross-attention if the architecture
      remains encoder-decoder based.
- [ ] Make encoder and decoder depth configurable.
- [ ] Apply and test dropout, or remove the unused configuration.
- [ ] Define training targets and axis semantics clearly.
- [ ] Normalize gradients by effective token or batch count.
- [ ] Add train/evaluation modes.
- [ ] Add a tiny reproducible language-model experiment.
- [ ] Add generation controls only after the basic path is correct.

### Generation

- [ ] Fix boundary handling.
- [ ] Separate prompt encoding from teacher-forcing conversion.
- [ ] Return generated tokens without padding artifacts.
- [ ] Add temperature and top-k sampling as optional strategies.
- [ ] Add deterministic greedy-generation tests.
- [ ] Consider KV caching only after attention semantics are correct.

## P2 — Configuration and usability

- [ ] Replace hard-coded dataset and checkpoint paths with command-line
      arguments.
- [ ] Add `--help` output for every executable.
- [ ] Add a shared configuration structure.
- [ ] Validate files before model initialization.
- [ ] Create output directories safely when needed.
- [ ] Add structured but lightweight logging.
- [ ] Return meaningful process exit codes.
- [ ] Keep example defaults small enough for portfolio demonstrations.

## P3 — Optional Python bindings

Python bindings are a possible future direction, not a current project feature.

If selected:

- [ ] Define a small stable C++ API before binding it.
- [ ] Add pybind11 as an optional dependency.
- [ ] Bind `Tensor<float>` construction and shape inspection.
- [ ] Add safe NumPy copy or view interoperability.
- [ ] Bind a minimal inference or experimentation surface.
- [ ] Create `pyproject.toml` packaging.
- [ ] Add Python unit tests.
- [ ] Build wheels only for platforms verified in CI.

Python dataset scripts should remain separate from the native binding package.

## P3 — Optional automatic differentiation

General autograd would be a major architectural expansion and should not be
inferred from the current manual backward methods.

If selected:

- [ ] Design operation nodes and tensor ownership rules.
- [ ] Track parents and backward functions.
- [ ] Implement topological backward traversal.
- [ ] Implement broadcasting-aware gradient reduction.
- [ ] Define leaf tensors, retained gradients, and graph lifetime.
- [ ] Add gradient checks for every differentiable tensor operation.
- [ ] Decide how manual layer backward code interoperates with the graph.

This work should be developed as a distinct milestone rather than added
incrementally without an architecture design.

## P3 — Optional GPU backend

GPU support is not required to make the current CPU project portfolio-ready.

If GPU work is selected later:

- [ ] Design a device and allocator abstraction first.
- [ ] Separate host and device storage.
- [ ] Add explicit data-transfer semantics.
- [ ] Implement a small set of CUDA kernels behind the same tested tensor API.
- [ ] Add CPU/GPU numerical parity tests.
- [ ] Add CUDA error handling and synchronization rules.
- [ ] Add CUDA-capable CI or a documented external verification environment.
- [ ] Benchmark only after correctness parity is established.

The repository should not use CUDA in its name until meaningful GPU execution
is implemented and continuously verified.

## P3 — Optional production inference work

Production inference is outside the current scope. If it becomes a goal:

- [ ] freeze a stable model API;
- [ ] implement model evaluation mode;
- [ ] design validated, versioned model loading;
- [ ] add batching and memory limits;
- [ ] add latency and throughput benchmarks;
- [ ] add observability and structured errors;
- [ ] define supported models and platforms;
- [ ] complete security review of all untrusted file parsing.

## Suggested release sequence

### Milestone 1 — Correctness baseline

Completion criteria:

- all P0 memory-safety defects are fixed;
- attention and layer gradients pass finite-difference checks;
- model parameter traversal is complete;
- Transformer boundary writes are fixed;
- the full test suite passes in a clean environment.

### Milestone 2 — Portfolio-ready CPU project

Completion criteria:

- final project rename is complete;
- Linux and Windows CI pass;
- repository hygiene files are present;
- one MNIST workflow is reproducible;
- architecture and build documentation match the code;
- screenshots contain only verified output.

### Milestone 3 — Reliable experimental framework

Completion criteria:

- tensor invariants and shape contracts are documented and enforced;
- checkpoint format is versioned and tested;
- model examples expose command-line configuration;
- first-party benchmarks are reproducible;
- a small Transformer-style experiment passes deterministic integration tests.

### Milestone 4 — Optional specialization

Select one direction rather than advertising all of them simultaneously:

- Python-accessible C++ tensor backend;
- general automatic differentiation;
- CUDA tensor backend;
- inference-focused runtime;
- deeper CPU performance engineering.

## Definition of “portfolio ready”

The repository is ready for GitHub, Upwork, and a technical CV when:

- [ ] the public name matches the CPU implementation;
- [ ] a fresh clone can be built from documented commands;
- [ ] CI shows passing tests;
- [ ] no committed binary or private IDE state remains;
- [ ] examples use configurable paths;
- [ ] at least one end-to-end workflow has verified output;
- [ ] numerical claims include reproducible evidence;
- [ ] experimental components are clearly labeled;
- [ ] README, architecture, roadmap, and code agree on the feature set;
- [ ] no CUDA, Python-binding, autograd, production, or scale capability is
      implied unless implemented and tested.
