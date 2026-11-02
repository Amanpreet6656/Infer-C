# InferC

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

## Building and Running

The codebase requires a C++17 compliant compiler (`g++` 9.0+ or `clang++` 10.0+), GNU Make, and Python 3.8+ (used by the offline weight exporter).

```bash
git clone https://github.com/your-username/InferC.git
cd InferC

make
make run
```

To compile manually without Make:

```bash
python3 scripts/export_weights.py
g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -march=native -I./include src/main.cpp -o bin/infer_engine
./bin/infer_engine
```
