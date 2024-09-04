# Specify the build directory
BUILD_DIR = build

# Specify the name of the executable
MAIN_EXECUTABLE = C++CudaTransformerX
TEST_EXECUTABLE = global_tests
DIGIT_RECOGNIZER_EXECUTABLE = digit_recognizer
EMBEDDING_MODEL_EXECUTABLE = embedding_model
MULTI_HEAD_ATTENTION_MODEL_EXECUTABLE = multi_head_attention_model

# The 'all' target will run 'venv', 'build', 'test', 'digit_recognizer', and 'clean' targets
all: venv build test digit_recognizer clean

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

# The 'clean' target to remove build artifacts
clean:
	rm -rf $(BUILD_DIR)

# The 'clean_venv' target to remove the Python virtual environment
clean_venv:
	rm -rf .venv

# The 'test' target
test:
	cd ${BUILD_DIR} && cmake --build . && ./$(TEST_EXECUTABLE)

# The 'digit_recognizer' target to run the DigitRecognizer executable
digit_recognizer:
	cd ${BUILD_DIR} && cmake --build . && ./$(DIGIT_RECOGNIZER_EXECUTABLE)

# The 'embedding_model' target to run the EmbeddingModel executable
embedding_model:
	cd ${BUILD_DIR} && cmake --build . && ./$(EMBEDDING_MODEL_EXECUTABLE)

# The 'multi_head_attention_model' target to run the MultiHeadAttentionModel executable
multi_head_attention_model:
	cd ${BUILD_DIR} && cmake --build . && ./$(MULTI_HEAD_ATTENTION_MODEL_EXECUTABLE)

# Profile targets
profile_main:
	make build && cd ${BUILD_DIR} && ulimit -s 16384 && valgrind --tool=callgrind ./$(MAIN_EXECUTABLE)

# Download datasets and vocab
download_data:
	@mkdir -p data
	. .venv/bin/activate && python utils_py/main.py


# Phony targets
.PHONY: all venv build test digit_recognizer clean profile_main clean_venv embedding_model multi_head_attention_model download_data

# End of Makefile
