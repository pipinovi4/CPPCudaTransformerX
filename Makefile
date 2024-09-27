# Makefile for C++CudaTransformerX

# Define build directory and executables
BUILD_DIR = build
EXECUTABLES = C++CudaTransformerX global_tests digit_recognizer embedding_model multi_head_attention_model train generate

# Default target: create virtual environment, build, test, run digit recognizer, then clean
all: venv build test digit_recognizer clean

# Create a Python virtual environment and install dependencies
venv:
	@echo "Installing Python3 virtual environment..."
	sudo apt install python3-venv
	@echo "Creating and activating the virtual environment..."
	python3 -m venv .venv
	@echo "Installing Python dependencies..."
	. .venv/bin/activate && pip install -r requirements.txt

# Build the project using Conan and CMake
build:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Detecting Conan profile..."
	conan profile detect --force
	@echo "Installing Conan dependencies..."
	conan install . --output-folder=build --build=missing
	@echo "Running CMake build..."
	cd $(BUILD_DIR) && cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release && cmake --build .

# Clean build directory and virtual environment
clean:
	@echo "Cleaning build directory and virtual environment..."
	rm -rf $(BUILD_DIR) .venv

# Targets to run tests, models, and executables
test digit_recognizer embedding_model multi_head_attention_model train generate:
	@echo "Running $(subst _, ,$@)..."
	cd $(BUILD_DIR) && cmake --build . && ./$(subst _,,$@)

# Run the main executable
main:
	@echo "Running main executable..."
	cd $(BUILD_DIR) && cmake --build . && ./C++CudaTransformerX

# Profile the main executable using Valgrind
profile_main:
	@echo "Profiling main executable with Valgrind..."
	make build && cd $(BUILD_DIR) && ulimit -s 16384 && valgrind --tool=callgrind ./C++CudaTransformerX

# Download necessary datasets and vocabulary
download_data:
	@echo "Downloading data..."
	. .venv/bin/activate && python utils_py/main.py

# Display help message for available targets
help:
	@echo "Available targets:"
	@echo "  all - Runs 'venv', 'build', 'test', 'digit_recognizer', and 'clean'"
	@echo "  venv - Set up Python virtual environment and install dependencies"
	@echo "  build - Build project using CMake and Conan"
	@echo "  clean - Clean build artifacts and virtual environment"
	@echo "  test, digit_recognizer, embedding_model, multi_head_attention_model, main, train, generate - Run respective executables"
	@echo "  profile_main - Profile the main executable using Valgrind"
	@echo "  download_data - Download datasets and vocab"
	@echo "  help - Display this help message"

# Declare phony targets to prevent conflicts with files
.PHONY: all venv build clean test digit_recognizer embedding_model multi_head_attention_model profile_main download_data main train generate help
