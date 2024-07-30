# Specify the build directory
BUILD_DIR = build

# Specify the name of the executable
TEST_EXECUTABLE = global_tests

# The 'all' target will run 'venv', 'build', 'test', and 'clean' targets
all: venv build test clean

# The 'venv' target to create and activate the Python virtual environment
venv:
	@echo "Creating Python virtual environment..."
	python3 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	. .venv/bin/activate && pip install -r requirements.txt

# The 'build' target
build:
	rm -rf $(BUILD_DIR)
	conan profile detect --force
	conan profile path default
	conan install . --output-folder=build --build=missing
	cd ${BUILD_DIR} && cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release && cmake --build .

# The 'test' target
test:
	cd ${BUILD_DIR} && cmake --build . && ./$(TEST_EXECUTABLE)

# The 'clean' target to remove build artifacts
clean:
	rm -rf $(BUILD_DIR)
