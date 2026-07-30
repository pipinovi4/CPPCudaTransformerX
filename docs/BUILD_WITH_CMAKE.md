# Building and Running with CMake

This guide explains how to download, configure, build, test, and run
BackpropLab using Conan and CMake.

The current implementation is CPU-based. An NVIDIA GPU and the CUDA Toolkit are not required.

## 1. Install the required tools

You need:

- Git
- Python 3
- Conan 2
- CMake 3.15 or newer
- Ninja or another CMake-supported build system
- a C++17 compiler

Supported compiler examples:

- GCC or Clang on Linux;
- Apple Clang on macOS;
- Visual Studio 2022 on Windows.

### Ubuntu or Debian

```bash
sudo apt update
sudo apt install git python3 python3-pip cmake ninja-build build-essential
python3 -m pip install --user "conan>=2,<3"
```

If `conan` is not found after installation, add the Python user binary directory to `PATH` or use a Python virtual environment.

### Windows

Install:

1. Git for Windows;
2. Python 3;
3. CMake;
4. Visual Studio 2022 with the **Desktop development with C++** workload;
5. Ninja, unless it is already supplied by Visual Studio or CMake.

Then open PowerShell:

```powershell
python -m pip install "conan>=2,<3"
```

Check the tools:

```powershell
git --version
python --version
conan --version
cmake --version
ninja --version
```

### macOS

Using Homebrew:

```bash
brew install git python cmake ninja conan
```

## 2. Download the repository

```bash
git clone https://github.com/<your-github-username>/backprop-lab-cpp.git
cd backprop-lab-cpp
```

Replace `<your-github-username>` with the repository owner's GitHub username.

## 3. Create a Conan compiler profile

Run this once on a development machine:

```bash
conan profile detect --force
```

Conan records the detected compiler, architecture, and standard-library settings in the user's Conan configuration directory. No absolute machine-specific path is stored in this repository.

## 4. Install C++ dependencies

The repository declares Eigen and GoogleTest in `conanfile.txt`.

### Linux and macOS

```bash
conan install . \
  --output-folder=build \
  --build=missing \
  -s build_type=Release
```

### Windows PowerShell

```powershell
conan install . `
  --output-folder=build `
  --build=missing `
  -s build_type=Release
```

This creates Conan-generated CMake files under `build/`. The directory is ignored by Git.

## 5. Configure the project

The repository includes a portable `conan-release` CMake preset:

```bash
cmake --preset conan-release
```

The preset resolves its paths from `${sourceDir}`. It contains no user-specific or operating-system-specific absolute directories.

If Ninja is not available, configure without the preset and choose a generator installed on the machine.

### Linux or macOS fallback

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release
```

### Windows Visual Studio fallback

```powershell
cmake -S . -B build `
  -G "Visual Studio 17 2022" `
  -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
```

For a multi-configuration Visual Studio build, add `--config Release` to build and test commands.

## 6. Build

Using the preset:

```bash
cmake --build --preset conan-release --parallel
```

Without the preset:

```bash
cmake --build build --parallel
```

Windows Visual Studio:

```powershell
cmake --build build --config Release --parallel
```

## 7. Run the tests

CTest knows about the GoogleTest test cases registered by CMake.

Using the preset:

```bash
ctest --preset conan-release
```

Using a build directory directly:

```bash
ctest --test-dir build --output-on-failure
```

Windows Visual Studio:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

The complete test executable can also be run directly:

```bash
./build/global_tests
```

On Windows with the Visual Studio generator:

```powershell
.\build\Release\global_tests.exe
```

## 8. Build options

CMake exposes the following options:

| Option | Default | Purpose |
| --- | --- | --- |
| `BACKPROP_LAB_BUILD_TESTS` | `ON` | Build the GoogleTest suite |
| `BACKPROP_LAB_BUILD_EXAMPLES` | `ON` | Build the standalone model examples |
| `BACKPROP_LAB_ENABLE_OPENMP` | `ON` | Use OpenMP when the compiler provides it |
| `BUILD_TESTING` | `ON` | Enable CTest integration |

For example, to build only the main applications:

```bash
cmake -S . -B build/minimal \
  -DBACKPROP_LAB_BUILD_TESTS=OFF \
  -DBACKPROP_LAB_BUILD_EXAMPLES=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build/minimal --parallel
```

## 9. Prepare the datasets

The model examples expect generated files under `data/`.

Create a virtual environment:

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download and preprocess the datasets:

```bash
python utils_py/main.py
```

The `data/` directory is ignored by Git because datasets and generated model weights should not be committed.

## 10. Run the executables

Single-configuration generators such as Ninja place executables directly in the build directory:

```bash
./build/backprop_lab_cli
./build/digit_recognizer
./build/embedding_model
./build/multi_head_attention_model
./build/train
./build/generate
```

Visual Studio normally places Release executables in:

```text
build/Release/
```

PowerShell examples:

```powershell
.\build\Release\backprop_lab_cli.exe
.\build\Release\digit_recognizer.exe
.\build\Release\train.exe
.\build\Release\generate.exe
```

The dataset-dependent executables currently use paths relative to their working directory. If an executable cannot locate `../data/...`, run it from the build directory:

```bash
cd build
./digit_recognizer
```

## 11. Available CMake targets

| Target | Description |
| --- | --- |
| `backprop_lab` | Header-only interface target for the framework |
| `backprop_lab_cli` | General demonstration executable |
| `global_tests` | GoogleTest suite |
| `digit_recognizer` | MNIST classifier example |
| `embedding_model` | Embedding experiment |
| `multi_head_attention_model` | Attention experiment |
| `train` | Experimental Transformer training |
| `generate` | Experimental greedy text generation |

Build only one target:

```bash
cmake --build build --target global_tests --parallel
```

## 12. Troubleshooting

### CMake cannot find the Conan toolchain

Run `conan install` before `cmake --preset conan-release`. Confirm that this file exists:

```text
build/conan_toolchain.cmake
```

### CMake cannot find Eigen or GoogleTest

Delete only the generated build directory, rerun Conan, and configure again:

```bash
cmake -E remove_directory build
conan install . --output-folder=build --build=missing -s build_type=Release
cmake --preset conan-release
```

Do not delete source directories such as `include/`, `src/`, or `models/`.

### Ninja is not installed

Install Ninja or use a different generator with a manual `cmake -S . -B ...` command.

### OpenMP is unavailable

OpenMP is optional. CMake continues without it. It can also be disabled explicitly:

```bash
cmake -S . -B build/no-openmp \
  -DBACKPROP_LAB_ENABLE_OPENMP=OFF
```

### Dataset files cannot be opened

Run:

```bash
python utils_py/main.py
```

Then execute the program from the expected build directory. The dataset paths in the current examples are relative.

### Model weights cannot be loaded

Run `train` before `generate`, and ensure that the `data/weights/` directory exists. Checkpoints are tied to the exact model configuration that created them.

## 13. Clean the generated build

CMake provides a portable way to remove a generated build directory:

```bash
cmake -E remove_directory build
```

This removes build artifacts only. It does not remove datasets, source files, or the Python virtual environment.
