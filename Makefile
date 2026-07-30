# Optional convenience wrapper around the documented Conan/CMake workflow.
# CMake remains the authoritative build interface.

BUILD_DIR := build
PRESET := conan-release
PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: configure build test all main digit_recognizer embedding_model
.PHONY: multi_head_attention_model train generate venv download_data clean help

configure:
	conan profile detect --force
	conan install . --output-folder=$(BUILD_DIR) --build=missing -s build_type=Release
	cmake --preset $(PRESET)

build:
	cmake --build --preset $(PRESET) --parallel

test:
	ctest --preset $(PRESET)

all: configure build test

main:
	cmake --build $(BUILD_DIR) --target backprop_lab_cli --parallel
	cd $(BUILD_DIR) && ./backprop_lab_cli

digit_recognizer embedding_model multi_head_attention_model train generate:
	cmake --build $(BUILD_DIR) --target $@ --parallel
	cd $(BUILD_DIR) && ./$@

venv:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -r requirements.txt

download_data:
	. .venv/bin/activate && python utils_py/main.py

clean:
	cmake -E remove_directory $(BUILD_DIR)

help:
	@echo "BackpropLab convenience targets:"
	@echo "  make configure        Install Conan dependencies and configure CMake"
	@echo "  make build            Build all configured targets"
	@echo "  make test             Run the CTest suite"
	@echo "  make all              Configure, build, and test"
	@echo "  make main             Build and run backprop_lab_cli"
	@echo "  make <example>        Build and run a named example"
	@echo "  make venv             Create a Python venv and install data dependencies"
	@echo "  make download_data    Download and preprocess datasets"
	@echo "  make clean            Remove generated CMake build artifacts"
