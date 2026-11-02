#ifndef INFERC_HPP
#define INFERC_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <istream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <memory>
#include <limits>
#include <cassert>
#include <string>
#include <cstring>
#include <initializer_list>
#include <utility>

namespace inferc {

#if defined(__GNUC__) || defined(__clang__)
    #define INFERC_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define INFERC_UNLIKELY(x) __builtin_expect(!!(x), 0)
    #define INFERC_RESTRICT    __restrict__
#else
    #define INFERC_LIKELY(x)   (x)
    #define INFERC_UNLIKELY(x) (x)
    #define INFERC_RESTRICT
#endif

/**
 * @file InferC.hpp
 * @brief High-performance header-only neural network inference library for bare-metal and edge platforms.
 * 
 * Provides memory-safe continuous tensor representations, cache-blocked matrix operations,
 * fused feed-forward transformations, and numerically stabilized non-linear activations without
 * dynamic memory allocations on hot execution paths.
 */

/**
 * @struct ModelFileHeader
 * @brief Binary serialization header for validated cross-platform model ingestion.
 * 
 * Encapsulates format metadata required to establish architecture compatibility, endianness
 * negotiation, and payload alignment prior to reading binary parameters into memory.
 */
#pragma pack(push, 1)
struct ModelFileHeader {
    uint32_t magic;      ///< Format signature token (ASCII representation: "INFC")
    uint16_t version;    ///< Binary protocol schema version
    uint16_t endianness; ///< Endianness validation flag (0x0102 denotes Little-Endian)
};
#pragma pack(pop)

static_assert(sizeof(ModelFileHeader) == 8, "ModelFileHeader binary layout must evaluate to exactly 8 octets.");

constexpr uint32_t EXPECTED_MAGIC      = 0x43464E49; ///< Constant token for format validation ("INFC")
constexpr uint16_t EXPECTED_VERSION    = 1;          ///< Supported binary specification revision
constexpr uint16_t EXPECTED_ENDIANNESS = 0x0102;     ///< Target Little-Endian wire encoding

namespace detail {
    /**
     * @brief Cold out-of-line exception dispatchers.
     * 
     * Separating exception construction from hot execution branches eliminates stack-frame
     * setup and register spilling in latency-critical inner loops.
     */
    [[noreturn]] inline void throw_out_of_range(size_t r, size_t c, size_t rows, size_t cols) {
        throw std::out_of_range("Tensor index (" + std::to_string(r) + ", " + 
                               std::to_string(c) + ") out of bounds for shape (" + 
                               std::to_string(rows) + ", " + std::to_string(cols) + ").");
    }

    [[noreturn]] inline void throw_dim_mismatch(size_t r1, size_t c1, size_t r2, size_t c2) {
        throw std::invalid_argument("Matrix dimension mismatch for multiplication: (" +
                                   std::to_string(r1) + "x" + std::to_string(c1) +
                                   ") cannot multiply (" +
                                   std::to_string(r2) + "x" + std::to_string(c2) + ").");
    }

    [[noreturn]] inline void throw_overflow(const char* msg) {
        throw std::overflow_error(msg);
    }
} // namespace detail

/**
 * @class Tensor
 * @brief Dense two-dimensional tensor backed by a contiguous row-major memory allocation.
 * 
 * Manages memory via RAII semantics using a standard contiguous buffer. Computations are structured
 * to leverage spatial locality and SIMD pipeline saturation, providing both checked reference
 * accessors and zero-allocation destination-passing mathematical routines.
 */
class Tensor {
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<float> m_data;

    /**
     * @brief Asserts mathematical dimension bounds against system pointer capacity.
     * 
     * Verifies that neither the coordinate product nor the aggregate byte demand exceeds
     * std::numeric_limits<size_t>::max(), guarding against integer wrap-around vulnerabilities.
     */
    static void validate_dimensions(size_t rows, size_t cols) {
        if (rows != 0) {
            if (cols > std::numeric_limits<size_t>::max() / rows) {
                detail::throw_overflow("Tensor coordinate dimensions overflow addressable index capacity.");
            }
            if ((rows * cols) > std::numeric_limits<size_t>::max() / sizeof(float)) {
                detail::throw_overflow("Aggregate tensor allocation exceeds addressable byte limits.");
            }
        }
    }

public:
    /**
     * @brief Constructs an empty tensor possessing zero-dimensional bounds.
     */
    Tensor() noexcept : m_rows(0), m_cols(0) {}
    
    /**
     * @brief Allocates an initialized contiguous tensor of dimension rows x cols.
     * @param rows Dimension of primary axis (row cardinality).
     * @param cols Dimension of secondary axis (column cardinality).
     * @param init_val Uniform scalar applied across all allocated coordinates.
     * @complexity O(rows * cols) time, O(rows * cols) space.
     */
    Tensor(size_t rows, size_t cols, float init_val = 0.0f) 
        : m_rows(rows), m_cols(cols) {
        validate_dimensions(rows, cols);
        const size_t total_elements = rows * cols;
        if (total_elements > m_data.max_size()) {
            throw std::length_error("Specified coordinate capacity exceeds container allocation bounds.");
        }
        m_data.resize(total_elements, init_val);
    }

    /**
     * @brief Constructs a tensor from a bracketed initializer list in row-major layout.
     * @param rows Row cardinality.
     * @param cols Column cardinality.
     * @param list Initializer sequence containing exactly rows * cols scalar values.
     */
    Tensor(size_t rows, size_t cols, std::initializer_list<float> list)
        : m_rows(rows), m_cols(cols) {
        validate_dimensions(rows, cols);
        const size_t total_elements = rows * cols;
        if (list.size() != total_elements) {
            throw std::invalid_argument("Initializer list cardinality (" + std::to_string(list.size()) +
                                       ") mismatches target tensor geometry (" + std::to_string(total_elements) + ").");
        }
        m_data.assign(list.begin(), list.end());
    }

    /**
     * @brief Constructs a tensor by copying an unmanaged flat float buffer.
     * @param rows Row cardinality.
     * @param cols Column cardinality.
     * @param src Pointer to contiguous source buffer containing at least rows * cols values.
     */
    Tensor(size_t rows, size_t cols, const float* src)
        : m_rows(rows), m_cols(cols) {
        validate_dimensions(rows, cols);
        const size_t total_elements = rows * cols;
        m_data.resize(total_elements);
        if (total_elements > 0 && src != nullptr) {
            std::memcpy(m_data.data(), src, total_elements * sizeof(float));
        }
    }

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    ~Tensor() = default;

    [[nodiscard]] inline size_t rows() const noexcept { return m_rows; }
    [[nodiscard]] inline size_t cols() const noexcept { return m_cols; }
    [[nodiscard]] inline size_t size() const noexcept { return m_data.size(); }
    [[nodiscard]] inline bool empty() const noexcept { return m_data.empty(); }
    
    [[nodiscard]] inline float* data() noexcept { return m_data.data(); }
    [[nodiscard]] inline const float* data() const noexcept { return m_data.data(); }

    /**
     * @brief Obtains a direct pointer to the origin of the requested row.
     * @param r Index along primary dimension.
     * @return Raw pointer to contiguous sequence of secondary dimension scalars.
     */
    [[nodiscard]] inline float* row(size_t r) noexcept { 
        return m_data.data() + r * m_cols; 
    }
    [[nodiscard]] inline const float* row(size_t r) const noexcept { 
        return m_data.data() + r * m_cols; 
    }

    /**
     * @brief Reconfigures memory capacity to accommodate the designated shape.
     */
    void resize(size_t rows, size_t cols, float init_val = 0.0f) {
        validate_dimensions(rows, cols);
        m_rows = rows;
        m_cols = cols;
        m_data.assign(rows * cols, init_val);
    }

    /**
     * @brief Subscript accessor providing boundary verification.
     * 
     * Uses compiler branch prediction primitives to bias code generation toward the valid
     * coordinate pathway, routing violations to a cold out-of-line throw handler.
     */
    inline float& operator()(size_t r, size_t c) {
        if (INFERC_UNLIKELY(r >= m_rows || c >= m_cols)) {
            detail::throw_out_of_range(r, c, m_rows, m_cols);
        }
        return m_data[r * m_cols + c];
    }
    
    inline const float& operator()(size_t r, size_t c) const {
        if (INFERC_UNLIKELY(r >= m_rows || c >= m_cols)) {
            detail::throw_out_of_range(r, c, m_rows, m_cols);
        }
        return m_data[r * m_cols + c];
    }

    inline float& at(size_t r, size_t c) {
        return (*this)(r, c);
    }

    inline const float& at(size_t r, size_t c) const {
        return (*this)(r, c);
    }

    /**
     * @brief Zero-allocation matrix multiplication: out = this * other.
     * 
     * Evaluates general matrix product using an i-k-j loop topology with 4-way unrolling along
     * the reduction axis. Non-aliasing pointers and vectorization pragmas permit auto-vectorization
     * into fused multiply-add (FMA) instructions with optimal L1/L2 cache prefetching.
     * 
     * @param other Multiplicand tensor of geometry (K x N).
     * @param out Pre-allocated destination tensor of geometry (M x N).
     * @complexity O(M * K * N) time, O(1) auxiliary heap space.
     */
    void matmul(const Tensor& other, Tensor& out) const {
        if (INFERC_UNLIKELY(this->m_cols != other.m_rows)) {
            throw std::invalid_argument("Matrix dimension mismatch for multiplication: (" +
                                       std::to_string(this->m_rows) + "x" + std::to_string(this->m_cols) +
                                       ") cannot multiply (" +
                                       std::to_string(other.m_rows) + "x" + std::to_string(other.m_cols) + ").");
        }

        if (this->m_rows != 0 && other.m_cols > std::numeric_limits<size_t>::max() / this->m_rows) {
            throw std::overflow_error("Result matrix dimensions overflow addressable memory.");
        }

        const size_t M = this->m_rows;
        const size_t K = this->m_cols;
        const size_t N = other.m_cols;

        if (INFERC_UNLIKELY(out.m_rows != M || out.m_cols != N)) {
            out.resize(M, N, 0.0f);
        } else {
            std::memset(out.data(), 0, M * N * sizeof(float));
        }

        if (M == 0 || K == 0 || N == 0) {
            return;
        }

        float* const INFERC_RESTRICT res_ptr = out.data();
        const float* const INFERC_RESTRICT this_ptr = this->data();
        const float* const INFERC_RESTRICT other_ptr = other.data();

        for (size_t i = 0; i < M; ++i) {
            float* const INFERC_RESTRICT res_row = res_ptr + i * N;
            const float* const INFERC_RESTRICT a_row = this_ptr + i * K;

            size_t k = 0;
            // 4-way loop unrolling across the reduction axis to maximize register reuse
            for (; k + 4 <= K; k += 4) {
                const float a0 = a_row[k];
                const float a1 = a_row[k + 1];
                const float a2 = a_row[k + 2];
                const float a3 = a_row[k + 3];
                const float* const INFERC_RESTRICT b0 = other_ptr + k * N;
                const float* const INFERC_RESTRICT b1 = other_ptr + (k + 1) * N;
                const float* const INFERC_RESTRICT b2 = other_ptr + (k + 2) * N;
                const float* const INFERC_RESTRICT b3 = other_ptr + (k + 3) * N;

                #pragma GCC ivdep
                for (size_t j = 0; j < N; ++j) {
                    res_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                }
            }
            // Scalar epilogue handling trailing reduction elements
            for (; k < K; ++k) {
                const float a_ik = a_row[k];
                const float* const INFERC_RESTRICT other_row = other_ptr + k * N;

                #pragma GCC ivdep
                for (size_t j = 0; j < N; ++j) {
                    res_row[j] += a_ik * other_row[j];
                }
            }
        }
    }

    /**
     * @brief Allocating matrix multiplication variant returning result by value.
     */
    [[nodiscard]] Tensor matmul(const Tensor& other) const {
        if (this->m_cols != other.m_rows) {
            throw std::invalid_argument("Matrix dimension mismatch for multiplication: (" +
                                       std::to_string(this->m_rows) + "x" + std::to_string(this->m_cols) +
                                       ") cannot multiply (" +
                                       std::to_string(other.m_rows) + "x" + std::to_string(other.m_cols) + ").");
        }

        if (this->m_rows != 0 && other.m_cols > std::numeric_limits<size_t>::max() / this->m_rows) {
            throw std::overflow_error("Result matrix dimensions overflow addressable memory.");
        }

        Tensor result(this->m_rows, other.m_cols, 0.0f);
        matmul(other, result);
        return result;
    }

    /**
     * @brief Fused GEMM + Bias transformation: out = this * weights + bias.
     * 
     * Bypasses intermediate zero-clearing passes by initializing accumulator registers directly
     * from the broadcasted bias vector. Eliminates an entire DRAM read/write roundtrip.
     * 
     * @param weights Kernel parameter matrix (K x N).
     * @param bias Row broadcast vector (1 x N).
     * @param out Destination matrix (M x N).
     * @complexity O(M * K * N) time, O(1) auxiliary heap space.
     */
    void gemm_bias(const Tensor& weights, const Tensor& bias, Tensor& out) const {
        if (INFERC_UNLIKELY(this->m_cols != weights.m_rows)) {
            throw std::invalid_argument("Matrix dimension mismatch for multiplication: (" +
                                       std::to_string(this->m_rows) + "x" + std::to_string(this->m_cols) +
                                       ") cannot multiply (" +
                                       std::to_string(weights.m_rows) + "x" + std::to_string(weights.m_cols) + ").");
        }
        if (INFERC_UNLIKELY(bias.rows() != 1 || bias.cols() != weights.m_cols)) {
            throw std::invalid_argument("Bias dimension mismatch: expected (1x" +
                                       std::to_string(weights.m_cols) + "), got (" +
                                       std::to_string(bias.rows()) + "x" + std::to_string(bias.cols()) + ").");
        }

        const size_t M = this->m_rows;
        const size_t K = this->m_cols;
        const size_t N = weights.m_cols;

        if (INFERC_UNLIKELY(out.m_rows != M || out.m_cols != N)) {
            out.resize(M, N);
        }

        if (M == 0 || N == 0) {
            return;
        }

        const float* const INFERC_RESTRICT a_ptr = this->data();
        const float* const INFERC_RESTRICT w_ptr = weights.data();
        const float* const INFERC_RESTRICT b_ptr = bias.data();
        float* const INFERC_RESTRICT c_ptr = out.data();

        for (size_t i = 0; i < M; ++i) {
            float* const INFERC_RESTRICT c_row = c_ptr + i * N;
            const float* const INFERC_RESTRICT a_row = a_ptr + i * K;

            // Direct vector assignment from bias eliminates dedicated clearing pass
            std::memcpy(c_row, b_ptr, N * sizeof(float));

            size_t k = 0;
            // 4-way unrolling along K reduction dimension
            for (; k + 4 <= K; k += 4) {
                const float a0 = a_row[k];
                const float a1 = a_row[k + 1];
                const float a2 = a_row[k + 2];
                const float a3 = a_row[k + 3];
                const float* const INFERC_RESTRICT w0 = w_ptr + k * N;
                const float* const INFERC_RESTRICT w1 = w_ptr + (k + 1) * N;
                const float* const INFERC_RESTRICT w2 = w_ptr + (k + 2) * N;
                const float* const INFERC_RESTRICT w3 = w_ptr + (k + 3) * N;

                #pragma GCC ivdep
                for (size_t j = 0; j < N; ++j) {
                    c_row[j] += a0 * w0[j] + a1 * w1[j] + a2 * w2[j] + a3 * w3[j];
                }
            }
            // Reduction epilogue
            for (; k < K; ++k) {
                const float a_ik = a_row[k];
                const float* const INFERC_RESTRICT w_row = w_ptr + k * N;

                #pragma GCC ivdep
                for (size_t j = 0; j < N; ++j) {
                    c_row[j] += a_ik * w_row[j];
                }
            }
        }
    }

    /**
     * @brief In-place broadcasting bias accumulation across batch coordinates.
     * @param bias Row vector of dimensions (1 x cols).
     * @complexity O(rows * cols) time, O(1) auxiliary space.
     */
    void add_bias(const Tensor& bias) {
        if (bias.rows() != 1 || bias.cols() != this->m_cols) {
            throw std::invalid_argument("Bias dimension mismatch: expected (1x" +
                                       std::to_string(this->m_cols) + "), got (" +
                                       std::to_string(bias.rows()) + "x" + std::to_string(bias.cols()) + ").");
        }
        if (this->m_rows == 0 || this->m_cols == 0) {
            return;
        }

        float* const INFERC_RESTRICT this_ptr = this->data();
        const float* const INFERC_RESTRICT bias_ptr = bias.data();
        const size_t rows_count = this->m_rows;
        const size_t cols_count = this->m_cols;

        // Pointer equality assertion prevents undefined behavior from restricted aliasing
        if (INFERC_UNLIKELY(this_ptr == bias_ptr)) {
            for (size_t j = 0; j < cols_count; ++j) {
                this_ptr[j] += this_ptr[j];
            }
            return;
        }

        for (size_t i = 0; i < rows_count; ++i) {
            float* const INFERC_RESTRICT row_ptr = this_ptr + i * cols_count;
            #pragma GCC ivdep
            for (size_t j = 0; j < cols_count; ++j) {
                row_ptr[j] += bias_ptr[j];
            }
        }
    }
};

/**
 * @class DenseLayer
 * @brief Fully connected neural network transformation stage computing Y = XW + B.
 * 
 * Encapsulates learnable weight and bias tensors. Provides memory-safe binary deserialization
 * with transactional commit semantics, alongside zero-allocation feed-forward execution.
 */
class DenseLayer {
private:
    Tensor weights;
    Tensor biases;

public:
    /**
     * @brief Constructs an uninitialized dense layer with specified boundary geometry.
     * @param input_dim Cardinality of input feature space.
     * @param output_dim Cardinality of projected output space.
     */
    DenseLayer(size_t input_dim, size_t output_dim) 
        : weights(input_dim, output_dim), biases(1, output_dim) {
        if (input_dim == 0 || output_dim == 0) {
            throw std::invalid_argument("DenseLayer dimensions must be positive non-zero integers.");
        }
    }

    /**
     * @brief Deserializes parameters from a binary stream with transactional exception safety.
     * 
     * Ingests header tokens, validates architecture dimensions, allocates isolated staging
     * buffers, and verifies IEEE 754 numerical finiteness before moving buffers into active state.
     * 
     * @param is Readable binary input stream.
     * @throws std::runtime_error On truncation, signature mismatch, or non-finite float detection.
     */
    void load_weights(std::istream& is) {
        if (!is.good()) {
            throw std::runtime_error("Weight input stream is not open or in an invalid state.");
        }

        // Header ingestion and byte-order integrity verification
        ModelFileHeader header{};
        is.read(reinterpret_cast<char*>(&header), sizeof(ModelFileHeader));
        if (is.gcount() != sizeof(ModelFileHeader) || !is) {
            throw std::runtime_error("Failed to read ModelFileHeader: stream truncated or unreadable.");
        }

        if (header.magic != EXPECTED_MAGIC) {
            throw std::runtime_error("Invalid model file magic identifier (corrupted format or unknown file type).");
        }
        if (header.version != EXPECTED_VERSION) {
            throw std::runtime_error("Unsupported model version: expected " + std::to_string(EXPECTED_VERSION) +
                                     ", got " + std::to_string(header.version));
        }
        if (header.endianness != EXPECTED_ENDIANNESS) {
            throw std::runtime_error("Unsupported model endianness encoding.");
        }

        // Geometry specification verification
        uint32_t in_dim = 0;
        uint32_t out_dim = 0;
        is.read(reinterpret_cast<char*>(&in_dim), sizeof(uint32_t));
        if (is.gcount() != sizeof(uint32_t) || !is) {
            throw std::runtime_error("Failed to read input dimension from stream.");
        }

        is.read(reinterpret_cast<char*>(&out_dim), sizeof(uint32_t));
        if (is.gcount() != sizeof(uint32_t) || !is) {
            throw std::runtime_error("Failed to read output dimension from stream.");
        }

        if (in_dim == 0 || out_dim == 0) {
            throw std::runtime_error("Invalid zero dimension loaded from weight header.");
        }
        if (static_cast<size_t>(in_dim) != weights.rows() || static_cast<size_t>(out_dim) != weights.cols()) {
            throw std::runtime_error("Weight dimension mismatch during loading: expected (" +
                                     std::to_string(weights.rows()) + "x" + std::to_string(weights.cols()) +
                                     "), got (" + std::to_string(in_dim) + "x" + std::to_string(out_dim) + ").");
        }

        // Byte count boundary validation preventing stream buffer overflow
        if (weights.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Weights byte count exceeds stream reading capacity.");
        }
        if (biases.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Biases byte count exceeds stream reading capacity.");
        }

        // Transactional staging buffers to guarantee strong exception safety
        Tensor staged_weights(weights.rows(), weights.cols());
        Tensor staged_biases(biases.rows(), biases.cols());

        const std::streamsize weight_bytes = static_cast<std::streamsize>(staged_weights.size() * sizeof(float));
        if (weight_bytes > 0) {
            if (staged_weights.data() == nullptr) {
                throw std::runtime_error("Null buffer pointer encountered for weights.");
            }
            is.read(reinterpret_cast<char*>(staged_weights.data()), weight_bytes);
            if (is.gcount() != weight_bytes || !is) {
                throw std::runtime_error("Stream truncated while reading layer weights (expected " +
                                         std::to_string(weight_bytes) + " bytes, read " +
                                         std::to_string(is.gcount()) + " bytes).");
            }
        }

        const std::streamsize bias_bytes = static_cast<std::streamsize>(staged_biases.size() * sizeof(float));
        if (bias_bytes > 0) {
            if (staged_biases.data() == nullptr) {
                throw std::runtime_error("Null buffer pointer encountered for biases.");
            }
            is.read(reinterpret_cast<char*>(staged_biases.data()), bias_bytes);
            if (is.gcount() != bias_bytes || !is) {
                throw std::runtime_error("Stream truncated while reading layer biases (expected " +
                                         std::to_string(bias_bytes) + " bytes, read " +
                                         std::to_string(is.gcount()) + " bytes).");
            }
        }

        // Sanitize incoming floats to guard against poison NaN/Inf values
        for (size_t i = 0; i < staged_weights.size(); ++i) {
            if (!std::isfinite(staged_weights.data()[i])) {
                throw std::runtime_error("Corrupted weight detected: non-finite float at index " + std::to_string(i));
            }
        }
        for (size_t i = 0; i < staged_biases.size(); ++i) {
            if (!std::isfinite(staged_biases.data()[i])) {
                throw std::runtime_error("Corrupted bias detected: non-finite float at index " + std::to_string(i));
            }
        }

        // Atomic commit via non-throwing move semantics
        weights = std::move(staged_weights);
        biases  = std::move(staged_biases);
    }

    /**
     * @brief Convenience wrapper loading model parameters from an explicit file path.
     */
    void load_weights(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open weight file: " + filepath);
        }
        file.exceptions(std::ifstream::badbit);
        load_weights(file);
    }

    /**
     * @brief Zero-allocation forward evaluation: output = input * W + B.
     * @param input Ingestion tensor of shape (Batch x InputDim).
     * @param output Destination buffer of shape (Batch x OutputDim).
     */
    void forward(const Tensor& input, Tensor& output) const {
        input.gemm_bias(weights, biases, output);
    }

    /**
     * @brief Allocating forward evaluation returning output tensor by value.
     */
    [[nodiscard]] Tensor forward(const Tensor& input) const {
        Tensor output(input.rows(), weights.cols());
        forward(input, output);
        return output;
    }

    [[nodiscard]] inline const Tensor& get_weights() const noexcept { return weights; }
    [[nodiscard]] inline const Tensor& get_biases() const noexcept { return biases; }
};

namespace activation {

    /**
     * @brief In-place Rectified Linear Unit activation with branchless NaN preservation.
     * 
     * Computes f(x) = max(0, x) over contiguous memory. Replaces standard conditional jumps with
     * branchless select operations, avoiding branch prediction penalties and enabling SIMD pipelining.
     * 
     * @param t Target tensor mutated in place.
     * @complexity O(N) time, O(1) space.
     */
    inline void relu(Tensor& t) {
        float* const INFERC_RESTRICT ptr = t.data();
        const size_t total_size = t.size();

        #pragma GCC ivdep
        for (size_t i = 0; i < total_size; ++i) {
            const float val = ptr[i];
            ptr[i] = (val > 0.0f) ? val : (std::isnan(val) ? val : 0.0f);
        }
    }

    /**
     * @brief Out-of-place variant of ReLU transformation.
     */
    [[nodiscard]] inline Tensor relu_copy(const Tensor& t) {
        Tensor result = t;
        relu(result);
        return result;
    }

    /**
     * @brief In-place numerically stabilized Softmax activation.
     * 
     * Maps unconstrained logits to valid probability distributions over each row vector:
     * \f[
     *   P_j = \frac{e^{z_j - \max(\mathbf{z})}}{\sum_k e^{z_k - \max(\mathbf{z})}}
     * \f]
     * Executes in two memory passes utilizing raw row pointers, eliminating indexing checks while
     * strictly guaranteeing protection against floating-point underflow, overflow, and division by zero.
     * 
     * @param t Logit tensor mutated into row-wise categorical probabilities.
     * @complexity O(rows * cols) time, O(1) auxiliary space.
     */
    inline void softmax(Tensor& t) {
        if (t.rows() == 0 || t.cols() == 0) {
            return;
        }

        const size_t rows_count = t.rows();
        const size_t cols_count = t.cols();
        float* const INFERC_RESTRICT data_ptr = t.data();

        for (size_t i = 0; i < rows_count; ++i) {
            float* const INFERC_RESTRICT row = data_ptr + i * cols_count;

            // Pass 1: Reduction to identify row maximum for exponent subtraction
            float max_val = -std::numeric_limits<float>::infinity();
            bool has_nan = false;

            for (size_t j = 0; j < cols_count; ++j) {
                const float val = row[j];
                if (INFERC_UNLIKELY(std::isnan(val))) {
                    has_nan = true;
                    break;
                }
                if (val > max_val) {
                    max_val = val;
                }
            }

            // Propagate quiet NaN across the coordinate axis if encountered
            if (has_nan) {
                const float nan_val = std::numeric_limits<float>::quiet_NaN();
                for (size_t j = 0; j < cols_count; ++j) {
                    row[j] = nan_val;
                }
                continue;
            }

            // Fallback to uniform distribution if coordinates contain exclusively negative infinity
            if (!std::isfinite(max_val)) {
                const float uniform = 1.0f / static_cast<float>(cols_count);
                for (size_t j = 0; j < cols_count; ++j) {
                    row[j] = uniform;
                }
                continue;
            }

            // Pass 2: Exponentiation and scalar sum accumulation
            float sum_exp = 0.0f;
            #pragma GCC ivdep
            for (size_t j = 0; j < cols_count; ++j) {
                const float val = row[j];
                if (val == -std::numeric_limits<float>::infinity()) {
                    row[j] = 0.0f;
                } else {
                    const float e = std::exp(val - max_val);
                    row[j] = e;
                    sum_exp += e;
                }
            }

            // Normalization phase guarding against zero-division singularities
            if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) {
                const float uniform = 1.0f / static_cast<float>(cols_count);
                for (size_t j = 0; j < cols_count; ++j) {
                    row[j] = uniform;
                }
            } else {
                const float inv_sum = 1.0f / sum_exp;
                #pragma GCC ivdep
                for (size_t j = 0; j < cols_count; ++j) {
                    row[j] *= inv_sum;
                }
            }
        }
    }

    /**
     * @brief Out-of-place variant of Softmax transformation.
     */
    [[nodiscard]] inline Tensor softmax_copy(const Tensor& t) {
        Tensor result = t;
        softmax(result);
        return result;
    }

} // namespace activation

} // namespace inferc

#endif // INFERC_HPP
