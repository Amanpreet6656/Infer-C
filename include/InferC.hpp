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
 * Binary serialization header for validated cross-platform model ingestion.
 */
#pragma pack(push, 1)
struct ModelFileHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t endianness;
};
#pragma pack(pop)

static_assert(sizeof(ModelFileHeader) == 8, "ModelFileHeader must be exactly 8 bytes.");

constexpr uint32_t EXPECTED_MAGIC      = 0x43464E49; // "INFC"
constexpr uint16_t EXPECTED_VERSION    = 1;
constexpr uint16_t EXPECTED_ENDIANNESS = 0x0102;     // Little-Endian

namespace detail {
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
}

/**
 * Continuous 2D dense tensor stored in row-major order.
 */
class Tensor {
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<float> m_data;

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
    Tensor() noexcept : m_rows(0), m_cols(0) {}
    
    Tensor(size_t rows, size_t cols, float init_val = 0.0f) 
        : m_rows(rows), m_cols(cols) {
        validate_dimensions(rows, cols);
        const size_t total_elements = rows * cols;
        if (total_elements > m_data.max_size()) {
            throw std::length_error("Specified coordinate capacity exceeds container allocation bounds.");
        }
        m_data.resize(total_elements, init_val);
    }

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

    [[nodiscard]] inline float* row(size_t r) noexcept { 
        return m_data.data() + r * m_cols; 
    }
    [[nodiscard]] inline const float* row(size_t r) const noexcept { 
        return m_data.data() + r * m_cols; 
    }

    /**
     * Resizes tensor capacity. Coordinates are committed after allocation succeeds.
     */
    void resize(size_t rows, size_t cols, float init_val = 0.0f) {
        validate_dimensions(rows, cols);
        m_data.assign(rows * cols, init_val);
        m_rows = rows;
        m_cols = cols;
    }

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
     * Zero-allocation matrix multiplication: out = this * other.
     * Evaluates in i-k-j loop topology with 4-way unrolling along reduction axis K.
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

        // Disallow in-place aliasing that violates __restrict__ rules
        if (INFERC_UNLIKELY(&out == this || &out == &other)) {
            Tensor temp;
            matmul(other, temp);
            out = std::move(temp);
            return;
        }

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
                    res_row[j] += a0 * b0[j];
                    res_row[j] += a1 * b1[j];
                    res_row[j] += a2 * b2[j];
                    res_row[j] += a3 * b3[j];
                }
            }
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
     * Fused GEMM + Bias kernel: out = this * weights + bias.
     * Accumulates matrix multiplication directly on top of broadcast bias vector.
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

        // Disallow in-place aliasing that violates __restrict__ rules
        if (INFERC_UNLIKELY(&out == this || &out == &weights || &out == &bias)) {
            Tensor temp;
            gemm_bias(weights, bias, temp);
            out = std::move(temp);
            return;
        }

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

            std::memcpy(c_row, b_ptr, N * sizeof(float));

            size_t k = 0;
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
                    c_row[j] += a0 * w0[j];
                    c_row[j] += a1 * w1[j];
                    c_row[j] += a2 * w2[j];
                    c_row[j] += a3 * w3[j];
                }
            }
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
     * Broadcasts bias vector across batch coordinates.
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
 * Fully connected linear transformation layer (Y = XW + B).
 */
class DenseLayer {
private:
    Tensor weights;
    Tensor biases;

public:
    DenseLayer(size_t input_dim, size_t output_dim) 
        : weights(input_dim, output_dim), biases(1, output_dim) {
        if (input_dim == 0 || output_dim == 0) {
            throw std::invalid_argument("DenseLayer dimensions must be positive non-zero integers.");
        }
    }

    /**
     * Deserializes layer parameters from stream with transactional commit semantics.
     */
    void load_weights(std::istream& is) {
        if (!is.good()) {
            throw std::runtime_error("Weight input stream is not open or in an invalid state.");
        }

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

        if (weights.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Weights byte count exceeds stream reading capacity.");
        }
        if (biases.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Biases byte count exceeds stream reading capacity.");
        }

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

        weights = std::move(staged_weights);
        biases  = std::move(staged_biases);
    }

    void load_weights(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open weight file: " + filepath);
        }
        file.exceptions(std::ifstream::badbit);
        load_weights(file);
    }

    /**
     * Pre-allocated forward evaluation: output = input * W + B.
     */
    void forward(const Tensor& input, Tensor& output) const {
        input.gemm_bias(weights, biases, output);
    }

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
     * Branchless Rectified Linear Unit with IEEE 754 quiet NaN preservation.
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

    [[nodiscard]] inline Tensor relu_copy(const Tensor& t) {
        Tensor result = t;
        relu(result);
        return result;
    }

    /**
     * Numerically stabilized Softmax with saturation boundary handling.
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

            if (has_nan) {
                const float nan_val = std::numeric_limits<float>::quiet_NaN();
                for (size_t j = 0; j < cols_count; ++j) {
                    row[j] = nan_val;
                }
                continue;
            }

            // Guard against saturated infinities to prevent (Inf - Inf) NaN generation
            if (INFERC_UNLIKELY(std::isinf(max_val))) {
                if (max_val < 0.0f) {
                    const float uniform = 1.0f / static_cast<float>(cols_count);
                    for (size_t j = 0; j < cols_count; ++j) {
                        row[j] = uniform;
                    }
                } else {
                    size_t inf_count = 0;
                    for (size_t j = 0; j < cols_count; ++j) {
                        if (row[j] == max_val) {
                            ++inf_count;
                        }
                    }
                    const float inf_prob = 1.0f / static_cast<float>(inf_count);
                    for (size_t j = 0; j < cols_count; ++j) {
                        row[j] = (row[j] == max_val) ? inf_prob : 0.0f;
                    }
                }
                continue;
            }

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

    [[nodiscard]] inline Tensor softmax_copy(const Tensor& t) {
        Tensor result = t;
        softmax(result);
        return result;
    }

}

}

#endif
