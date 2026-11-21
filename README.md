# InferC

[![CI](https://github.com/Amanpreet6656/Infer-C/actions/workflows/ci.yml/badge.svg)](https://github.com/Amanpreet6656/Infer-C/actions/workflows/ci.yml)

InferC is a lightweight, header-only C++17 inference engine designed for deploying small feed-forward neural networks to resource-constrained edge systems and embedded microprocessors. The library has zero external dependencies, avoiding BLAS abstractions, Eigen, or runtime frameworks. Core tensor math, cache-conscious matrix operations, and floating-point activations are implemented directly on top of the standard library, prioritizing deterministic latency, zero dynamic heap allocations on the hot path, and verified memory safety.

## Mathematical Architecture

The engine models standard multilayer perceptrons through sequential affine projections and non-linear mappings.

### Fully Connected Layers

For an input activation matrix $X \in \mathbb{R}^{M \times K}$, weight matrix $W \in \mathbb{R}^{K \times N}$, and row bias vector $B \in \mathbb{R}^{1 \times N}$, the layer computes:

$$Y = X \cdot W + B$$

where $M$ denotes batch size, $K$ represents input features, and $N$ represents output features. In conventional implementations, matrix multiplication and bias addition are separated into two distinct passes: an initial matrix multiplication call followed by an elementwise accumulation pass over $Y$. This approach forces intermediate tensor values back out to memory before reloading them for bias addition.

InferC implements a fused linear transformation. The output tensor $Y$ is initialized directly from the broadcast bias vector $B$, after which the matrix product is accumulated directly into the resident cache lines:

$$Y_{i, j} = B_j + \sum_{k=0}^{K-1} X_{i, k} \cdot W_{k, j}$$

This cuts the required memory bandwidth for intermediate activations in half and eliminates the need for scratchpad buffers.

### Non-Linear Activations

Rectified Linear Unit (ReLU) activations compute $f(x) = \max(0, x)$ across the tensor. In naive implementations, branching statements such as `if (x > 0)` cause frequent branch mispredictions when processing activations near zero. InferC avoids this overhead by using a branchless ternary selection that preserves IEEE 754 quiet NaN values while compiling down to single conditional move or vector blend instructions (`vblendvps`).

Softmax normalization converts unconstrained real logits $\mathbf{z} \in \mathbb{R}^{N}$ into normalized probabilities:

$$P_j = \frac{e^{z_j - \max(\mathbf{z})}}{\sum_{k=0}^{N-1} e^{z_k - \max(\mathbf{z})}}$$

Subtracting the maximum coordinate $\max(\mathbf{z})$ bounds the exponents to $(-\infty, 0]$, ensuring that `std::exp` cannot overflow the IEEE 754 single-precision range ($x > 88.7228$). The reduction handles saturated boundary conditions explicitly. If a row contains saturated positive infinite logits, evaluating $(\infty - \infty)$ directly would yield an invalid operation resulting in NaN. InferC detects infinite coordinates prior to exponentiation: if $+\infty$ is present, probability mass is distributed uniformly across the maximal infinite indices, while all finite coordinates receive zero. If all coordinates evaluate to $-\infty$, the output smoothly degrades to a uniform distribution $P_j = 1/N$.

## Hardware Optimization and Memory Model

Dense matrix multiplication scales with time complexity $\mathcal{O}(M \cdot K \cdot N)$. On modern hardware, execution throughput is primarily bounded by memory bandwidth and cache hierarchy efficiency rather than raw arithmetic capability.

### Cache Topology and Loop Scheduling

Memory in `inferc::Tensor` is laid out in contiguous row-major order, where element $(r, c)$ resides at index $r \cdot \text{cols} + c$. A naive loop order of $i \to j \to k$ traverses the weight matrix along its columns, producing stride-$N$ memory accesses that miss cache lines on almost every read.

InferC organizes matrix multiplication into an $i \to k \to j$ iteration topology. In this configuration, index $j$ runs along the innermost loop, allowing the processor to stream both the weight matrix rows $W_{k, :}$ and the destination rows $Y_{i, :}$ sequentially with stride-1 spatial locality. Hardware prefetchers can easily track these contiguous memory accesses, keeping the CPU pipelines consistently fed from L1/L2 cache.

### Vectorization and Register Reuse

The reduction loop along dimension $K$ is unrolled 4-way. During each step, four scalars from the input row are kept in CPU registers while the inner loop updates the output vector across multiple independent multiply-accumulate chains. Pointers are annotated with non-aliasing `__restrict__` qualifiers and decorated with `#pragma GCC ivdep` loop hints. This allows GCC and Clang to autovectorize the inner loops into packed 256-bit AVX2 or ARM NEON fused multiply-add (FMA) instructions, retiring multiple floating-point operations per cycle without register spilling.

### Zero-Allocation Forward Passes

Dynamic memory allocation during execution introduces unpredictable latency spikes, heap fragmentation, and allocator lock contention. InferC provides a destination-passing execution API where destination buffers are allocated once during system startup and reused across inference cycles.

Calls to `layer.forward(input, output)` operate with $\mathcal{O}(1)$ auxiliary heap memory overhead. Temporary state buffers are never constructed or destroyed inside the inference loop, making execution timing deterministic and safe for hard real-time threads.

## Memory Safety and Robustness

Writing high-performance C++ requires explicit defenses against edge-case undefined behavior and silent numerical corruption.

### Aliasing and In-Place Operations

Pointers marked with `__restrict__` promise the compiler that their target memory does not overlap with any other active pointer in scope. If a caller passes the same tensor as both input and output (for example, `layer.forward(tensor, tensor)`), writing to the output while reading from the input violates the restrict contract and triggers undefined compiler behavior under `-O3`. If the layer dimensions differ, resizing the destination buffer will deallocate the input memory while it is being read, resulting in a use-after-free.

InferC performs runtime identity checks on buffer pointers inside `matmul` and `gemm_bias`. If self-aliasing is detected, execution routes through a temporary buffer before moving the result into place, guaranteeing safety while preserving an in-place API surface.

### Exception Safety Guarantees

All state mutations satisfy the strong exception safety guarantee. In `Tensor::resize`, dimensions `m_rows` and `m_cols` are only updated after the underlying `std::vector::assign` completes successfully. If an allocation throws `std::bad_alloc`, the tensor retains its previous valid dimensions and buffer, preventing corrupted state where dimension metadata falls out of sync with allocated memory.

Similarly, `DenseLayer::load_weights` stages all incoming data into temporary buffers. Streams are validated for correct magic headers, architecture endianness, dimension conformity, and finite float values before any data is moved into the active layer. If reading fails midway through a file, the layer remains in its previous valid state.

### Numerical Invariants and Bounds

Coordinate calculations guard against integer overflow. Multiplications such as `rows * cols` and aggregate byte sizes are validated against `std::numeric_limits<size_t>::max()` prior to allocation. Coordinate lookups via `operator()(r, c)` verify boundaries using branch prediction hints (`[[unlikely]]`) that keep out-of-bounds error handling out of the instruction cache on the hot path.

## Binary Serialization Protocol

Model parameters are loaded using a clean, packed binary format:

```
Offset  Type      Field       Description
0x00    uint32_t  magic       Format signature (0x43464E49, ASCII "INFC")
0x04    uint16_t  version     Protocol version (currently 1)
0x06    uint16_t  endianness  Endian marker (0x0102 for Little-Endian)
0x08    uint32_t  in_dim      Input feature dimension
0x0C    uint32_t  out_dim     Output feature dimension
0x10    float[]   weights     Contiguous weight array (in_dim * out_dim floats)
var     float[]   biases      Contiguous bias array (out_dim floats)
```

Every weight and bias value is scanned during loading using `std::isfinite`. Corrupted or uninitialized values such as NaN or infinity in the weight file trigger immediate exceptions rather than poisoning runtime inference.

## Empirical Performance Benchmarks

To validate the efficiency of the cache-aware matrix multiplication and fused bias addition kernel, empirical benchmarks were executed comparing InferC against a baseline naive three-loop (i-j-k) GEMM with a decoupled bias accumulation pass.

### Test Environment
- **Host CPU:** AMD Ryzen 5 5500H (Zen 3 microarchitecture, 4 cores / 8 threads, 3.30 GHz base clock, 8 MB shared L3 cache)
- **Toolchain:** GCC 16.2 / Clang with `-std=c++17 -O3 -march=native`
- **Compiler Optimizations:** Loop interchange (i-k-j), 4-way loop unrolling along reduction axis K, `#pragma GCC ivdep` auto-vectorization, and `__restrict__` memory aliasing guarantees.

### Benchmark Results

| Matrix Geometry (MxKxN) | Layer Context | Iterations | Naive GEMM (us) | InferC Kernel (us) | Speedup | Max Absolute Error (epsilon) | Sustained GFLOPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **16x64x32** | Micro-batch edge inference | 10,000 | 18.23 us | **2.86 us** | **6.37x** | 2.38e-06 | 22.93 GFLOPS |
| **64x128x128** | Dense hidden layer pass | 5,000 | 1,268.00 us | **74.94 us** | **16.92x** | 4.77e-06 | 27.98 GFLOPS |
| **128x256x256** | Large batch edge projection | 1,000 | 12,042.62 us | **651.90 us** | **18.47x** | 8.58e-06 | 25.74 GFLOPS |
| **256x256x256** | Standard square benchmark | 500 | 21,057.01 us | **1,157.68 us** | **18.19x** | 1.14e-05 | 28.98 GFLOPS |

### Architectural Analysis of Speedup

1. **Elimination of Strided Memory Access:** Naive triple-loop multiplication (i-j-k) iterates over the rows of matrix B along its inner loop, incurring a memory stride of N * sizeof(float) on every single multiply-accumulate operation. This exhausts L1 cache lines almost immediately. InferC utilizes an i-k-j loop interchange, allowing the inner loop to stream contiguously along resident cache lines for both matrix B and the destination row accumulator.
2. **In-Register Bias Fusion:** In decoupled pipelines, the intermediate matrix product X * W is committed to dynamic memory, only to be immediately reloaded for bias vector addition Y + B. InferC initializes the destination register line directly with the broadcast bias elements, accumulating matrix products directly in-place and halving the required memory bus transactions.
3. **Numerical Equivalence:** In all benchmarked configurations, maximum divergence remained within single-precision floating-point roundoff tolerances (epsilon < 1.2e-5), satisfying rigorous numerical equivalence constraints.

## Building and Running

The codebase requires a C++17 compliant compiler (`g++` 9.0+ or `clang++` 10.0+), GNU Make, and Python 3.8+ (used by the offline weight exporter).

```bash
git clone https://github.com/Amanpreet6656/Infer-C.git
cd Infer-C

# Build and run sample inference
make
make run

# Run mathematical tolerance and invariant tests
make test

# Run empirical performance benchmarks
make benchmark
```

To compile manually without Make:

```bash
python3 scripts/export_weights.py bin/model_weights.bin
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native -I./include src/main.cpp -o bin/infer_engine
./bin/infer_engine
```
