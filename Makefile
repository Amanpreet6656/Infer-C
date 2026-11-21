# Compiler settings - standard C++17
CXX ?= g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic -Wno-unused-parameter
PYTHON ?= python3

# Hardware optimization flags (optional but recommended for Edge/Inference)
OPTFLAGS = -march=native

# Directories
INCLUDES = -I./include
SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin

# Target executable and model weights
TARGET = $(BIN_DIR)/infer_engine
WEIGHTS = $(BIN_DIR)/model_weights.bin
TEST_TARGET = $(BIN_DIR)/test_tolerance
BENCH_TARGET = $(BIN_DIR)/benchmark

# Sources and Objects
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# Default target
all: directories $(TARGET) $(WEIGHTS)

# Linking
$(TARGET): $(OBJECTS) | directories
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $^

# Compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | directories
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) -c $< -o $@

# Testing
$(TEST_TARGET): tests/test_tolerance.cpp | directories
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $< -o $@

test: $(TEST_TARGET)
	@./$(TEST_TARGET)

# Benchmarking
$(BENCH_TARGET): benchmarks/benchmark.cpp | directories
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) $< -o $@

benchmark: $(BENCH_TARGET)
	@./$(BENCH_TARGET)

# Utility targets
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

$(WEIGHTS): scripts/export_weights.py | directories
	@echo "Generating dummy weights using Python..."
	$(PYTHON) scripts/export_weights.py $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -f model_weights.bin *.exe

# Run the inference engine
run: all
	@cd $(BIN_DIR) && ./infer_engine

.PHONY: all clean directories run test benchmark
