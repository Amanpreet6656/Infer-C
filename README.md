# InferC

[![CI](https://github.com/Amanpreet6656/Infer-C/actions/workflows/ci.yml/badge.svg)](https://github.com/Amanpreet6656/Infer-C/actions/workflows/ci.yml)

InferC is a lightweight, header-only C++17 feed-forward neural network inference library designed for CPU inference without external dependencies (no BLAS, Eigen, or runtime ML frameworks). Core tensor operations, cache-conscious dense matrix multiplication, and floating-point activations are built solely on the C++17 Standard Library, prioritizing predictable buffer reuse, numerical stability, and verified memory safety.

## Scope & Design Boundaries

- **Implemented Primitives:** Contiguous row-major 2D `Tensor`, `DenseLayer` (`gemm_bias` forward pass), branchless `relu`, and overflow-guarded `softmax`.
- **Execution Model:** Supports a destination-passing API (`layer.forward(input, output)`) to avoid repeated heap allocations on the inference hot path once destination tensors are sized. Note that `Tensor` stores weights in `std::vector<float>` and reports invalid formats via C++ exceptions, targeting hosted edge CPUs rather than exception-free bare-metal microcontrollers.
- **Sample Exporter (`scripts/export_weights.py`):** Generates deterministic synthetic weights (`random.seed(42)`) for a 2-layer `4 -> 16 -> 3` MLP to validate binary serialization, numerical tolerances, and smoke tests.

## Mathematical Architecture

The library models multilayer perceptrons through sequential affine projections and non-linear activations.

### Fully Connected Layers

For an input activation matrix $X \in \mathbb{R}^{M \times K}$, weight matrix $W \in \mathbb{R}^{K \times N}$, and row bias vector $B \in \mathbb{R}^{1 \times N}$, the layer computes:

$$Y = X \cdot W + B$$

where $M$ denotes batch size, $K$ represents input features, and $N$ represents output features. Rather than running matrix multiplication followed by a separate pass over $Y$ to add $B$, `gemm_bias` initializes each output row directly from the bias vector $B$ and accumulates the matrix product into the output buffer:

$$Y_{i, j} = B_j + \sum_{k=0}^{K-1} X_{i, k} \cdot W_{k, j}$$

### Non-Linear Activations

Rectified Linear Unit (ReLU) computes $f(x) = \max(0, x)$ using a branchless ternary selection that preserves IEEE 754 quiet NaN values.

Softmax normalization converts unconstrained real logits $\mathbf{z} \in \mathbb{R}^{N}$ into normalized probabilities:

$$P_j = \frac{e^{z_j - \max(\mathbf{z})}}{\sum_{k=0}^{N-1} e^{z_k - \max(\mathbf{z})}}$$

Subtracting $\max(\mathbf{z})$ bounds exponents to $(-\infty, 0]$ to prevent IEEE 754 single-precision overflow. Saturated $+\infty$ rows distribute probability mass uniformly across maximal infinite indices; all-$-\infty$ rows fall back to a uniform distribution $1/N$; `NaN` entries propagate explicitly.

## Cache Topology and Memory Safety

- **Loop Ordering (`i -> k -> j`):** `inferc::Tensor` stores elements in contiguous row-major order. Organizing dense multiplication as `i -> k -> j` streams both weight rows $W_{k, :}$ and output rows $Y_{i, :}$ sequentially with stride-1 locality.
- **Unrolling & Aliasing Guards:** The inner reduction loop is unrolled 4-way along $K$ with `INFERC_RESTRICT` pointer qualifiers and `INFERC_IVDEP` loop hints (guarded for GCC). Before binding restrict pointers, `matmul`, `gemm_bias`, and `add_bias` check pointer identity at runtime (`&out == this`, etc.) and stage through a temporary tensor if aliasing occurs.
- **Exception Safety & Overflow Checks:** `Tensor::validate_dimensions` guards against `size_t` multiplication overflow prior to allocation. `Tensor::resize` commits dimensions only after `std::vector::assign` succeeds. `DenseLayer::load_weights` verifies little-endian host ordering, magic/version/endianness headers, dimensions, and `std::isfinite` across all weights and biases in staged buffers before moving them into the layer.

## Binary Serialization Protocol

```
Offset  Type      Field       Description
0x00    uint32_t  magic       Format signature (0x43464E49, ASCII "INFC")
0x04    uint16_t  version     Protocol version (currently 1)
0x06    uint16_t  endianness  Endian marker (0x0102 for Little-Endian)
0x08    uint32_t  in_dim      Input feature dimension
0x0C    uint32_t  out_dim     Output feature dimension
0x10    float[]   weights     Contiguous IEEE-754 float32 weights (in_dim * out_dim)
var     float[]   biases      Contiguous IEEE-754 float32 biases (out_dim)
```

## Verification and Benchmarking

All verification checks and benchmarks are executed automatically in GitHub Actions across both `g++` and `clang++` (including AddressSanitizer and UndefinedBehaviorSanitizer runs).

- **`tests/test_tolerance.cpp`:** Verifies fused `gemm_bias` and `matmul` against scalar reference math (including non-multiple-of-4 reduction dimensions), `add_bias`, ReLU quiet-NaN preservation, Softmax stability/saturation edge cases, dimension mismatch exceptions, in-place self-aliasing, and binary deserialization error rejection.
- **`benchmarks/benchmark.cpp`:** Compares the fused `i-k-j` kernel against a baseline `i-j-k` GEMM + separate bias pass across `16x64x32`, `64x128x128`, `128x256x256`, and `256x256x256` matrices with numerical equivalence assertions (`max_diff < 1e-4`).

## Building and Running

Requires a C++17 compiler (`g++` 9.0+ or `clang++` 10.0+), GNU Make, and Python 3.8+.

```bash
git clone https://github.com/Amanpreet6656/Infer-C.git
cd Infer-C

# Build and run sample inference
make
make run

# Run numerical tolerance and invariant tests
make test

# Run local GEMM benchmark
make benchmark
```

To compile manually without Make:

```bash
python3 scripts/export_weights.py bin/model_weights.bin
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native -I./include src/main.cpp -o bin/infer_engine
./bin/infer_engine
```
