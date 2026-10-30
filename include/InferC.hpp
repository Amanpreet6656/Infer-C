#ifndef INFERC_HPP
#define INFERC_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <memory>
#include <limits>
#include <cassert>
#include <string>

namespace inferc {

/**
 * @struct ModelFileHeader
 * @brief Formal binary header specification for InferC weight serialization.
 * 
 * Guarantees cross-architecture verification of magic signature, version,
 * and byte-order endianness before attempting memory ingestion.
 */
#pragma pack(push, 1)
struct ModelFileHeader {
    uint32_t magic;      // Magic identifier: 0x43464E49 ("INFC")
    uint16_t version;    // Protocol version: 1
    uint16_t endianness; // Endian marker: 0x0102 (Little-Endian)
};
#pragma pack(pop)

constexpr uint32_t EXPECTED_MAGIC = 0x43464E49;      // "INFC"
constexpr uint16_t EXPECTED_VERSION = 1;
constexpr uint16_t EXPECTED_ENDIANNESS = 0x0102;     // Little-Endian

/**
 * @class Tensor
 * @brief Continuous 2D memory block for matrix operations with bounds safety.
 * 
 * Uses std::vector for RAII-compliant memory management.
 * Elements are stored in contiguous row-major order.
 */
class Tensor {
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<float> m_data;

public:
    Tensor() noexcept : m_rows(0), m_cols(0) {}
    
    Tensor(size_t rows, size_t cols, float init_val = 0.0f) 
        : m_rows(rows), m_cols(cols) {
        // Guard against integer overflow in allocation size computation
        if (rows != 0 && cols > std::numeric_limits<size_t>::max() / rows) {
            throw std::overflow_error("Tensor dimensions overflow addressable memory size.");
        }
        const size_t total_elements = rows * cols;
        if (total_elements > m_data.max_size()) {
            throw std::length_error("Tensor size exceeds maximum vector capacity.");
        }
        m_data.resize(total_elements, init_val);
    }

    // Expose dimensions
    inline size_t rows() const noexcept { return m_rows; }
    inline size_t cols() const noexcept { return m_cols; }
    inline size_t size() const noexcept { return m_data.size(); }
    inline bool empty() const noexcept { return m_data.empty(); }
    
    // Raw contiguous data access
    inline float* data() noexcept { return m_data.data(); }
    inline const float* data() const noexcept { return m_data.data(); }

    /**
     * @brief 2D Element Access with strict boundary checking.
     * Prevents out-of-bounds heap corruption and column aliasing.
     */
    inline float& operator()(size_t r, size_t c) {
        if (r >= m_rows || c >= m_cols) {
            throw std::out_of_range("Tensor index (" + std::to_string(r) + ", " + 
                                   std::to_string(c) + ") out of bounds for shape (" + 
                                   std::to_string(m_rows) + ", " + std::to_string(m_cols) + ").");
        }
        return m_data[r * m_cols + c];
    }
    
    inline const float& operator()(size_t r, size_t c) const {
        if (r >= m_rows || c >= m_cols) {
            throw std::out_of_range("Tensor index (" + std::to_string(r) + ", " + 
                                   std::to_string(c) + ") out of bounds for shape (" + 
                                   std::to_string(m_rows) + ", " + std::to_string(m_cols) + ").");
        }
        return m_data[r * m_cols + c];
    }

    /**
     * @brief Matrix Multiplication (C = A * B)
     * Validates dimensions and guards against overflow.
     * Traverses in i-k-j loop order with row pointers for cache optimization and SIMD vectorization.
     */
    Tensor matmul(const Tensor& other) const {
        if (this->m_cols != other.m_rows) {
            throw std::invalid_argument("Matrix dimension mismatch for multiplication: (" +
                                       std::to_string(this->m_rows) + "x" + std::to_string(this->m_cols) +
                                       ") cannot multiply (" +
                                       std::to_string(other.m_rows) + "x" + std::to_string(other.m_cols) + ").");
        }

        // Verify result tensor dimensions do not overflow size_t
        if (this->m_rows != 0 && other.m_cols > std::numeric_limits<size_t>::max() / this->m_rows) {
            throw std::overflow_error("Result matrix dimensions overflow addressable memory.");
        }

        Tensor result(this->m_rows, other.m_cols, 0.0f);
        if (this->m_rows == 0 || other.m_cols == 0 || this->m_cols == 0) {
            return result;
        }

        float* res_ptr = result.data();
        const float* this_ptr = this->data();
        const float* other_ptr = other.data();

        // Optimized i-k-j loop order
        for (size_t i = 0; i < this->m_rows; ++i) {
            float* res_row = res_ptr + i * other.m_cols;
            for (size_t k = 0; k < this->m_cols; ++k) {
                const float a_ik = this_ptr[i * this->m_cols + k];
                const float* other_row = other_ptr + k * other.m_cols;
                for (size_t j = 0; j < other.m_cols; ++j) {
                    res_row[j] += a_ik * other_row[j];
                }
            }
        }
        return result;
    }

    /**
     * @brief Row-wise addition (broadcasting bias vector across batch dimension).
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

        float* this_ptr = this->data();
        const float* bias_ptr = bias.data();
        for (size_t i = 0; i < this->m_rows; ++i) {
            float* row = this_ptr + i * this->m_cols;
            for (size_t j = 0; j < this->m_cols; ++j) {
                row[j] += bias_ptr[j];
            }
        }
    }
};

/**
 * @class DenseLayer
 * @brief Fully Connected Layer computing Y = XW + B
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
     * @brief Deserializes weights and biases from a binary file stream with strict security checks.
     * Validates magic bytes, protocol version, endianness, dimensions, byte counts, and float integrity.
     */
    void load_weights(std::ifstream& is) {
        if (!is.is_open() || !is.good()) {
            throw std::runtime_error("Weight input stream is not open or in an invalid state.");
        }

        // 1. Read and validate ModelFileHeader
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

        // 2. Read and validate Layer Dimensions (unsigned 32-bit integers)
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

        // 3. Overflow validation for stream byte counts
        if (weights.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Weights byte count exceeds stream reading capacity.");
        }
        if (biases.size() > (static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(float))) {
            throw std::overflow_error("Biases byte count exceeds stream reading capacity.");
        }

        // 4. Read Weights
        const std::streamsize weight_bytes = static_cast<std::streamsize>(weights.size() * sizeof(float));
        if (weight_bytes > 0) {
            if (weights.data() == nullptr) {
                throw std::runtime_error("Null buffer pointer encountered for weights.");
            }
            is.read(reinterpret_cast<char*>(weights.data()), weight_bytes);
            if (is.gcount() != weight_bytes || !is) {
                throw std::runtime_error("Stream truncated while reading layer weights (expected " +
                                         std::to_string(weight_bytes) + " bytes, read " +
                                         std::to_string(is.gcount()) + " bytes).");
            }
        }

        // 5. Read Biases
        const std::streamsize bias_bytes = static_cast<std::streamsize>(biases.size() * sizeof(float));
        if (bias_bytes > 0) {
            if (biases.data() == nullptr) {
                throw std::runtime_error("Null buffer pointer encountered for biases.");
            }
            is.read(reinterpret_cast<char*>(biases.data()), bias_bytes);
            if (is.gcount() != bias_bytes || !is) {
                throw std::runtime_error("Stream truncated while reading layer biases (expected " +
                                         std::to_string(bias_bytes) + " bytes, read " +
                                         std::to_string(is.gcount()) + " bytes).");
            }
        }

        // 6. Verify Numerical Integrity (sanitize against NaNs and Infs)
        for (size_t i = 0; i < weights.size(); ++i) {
            if (!std::isfinite(weights.data()[i])) {
                throw std::runtime_error("Corrupted weight detected: non-finite float at index " + std::to_string(i));
            }
        }
        for (size_t i = 0; i < biases.size(); ++i) {
            if (!std::isfinite(biases.data()[i])) {
                throw std::runtime_error("Corrupted bias detected: non-finite float at index " + std::to_string(i));
            }
        }
    }

    /**
     * @brief Forward pass: computes X * W + B
     */
    Tensor forward(const Tensor& input) const {
        Tensor output = input.matmul(weights);
        output.add_bias(biases);
        return output;
    }

    inline const Tensor& get_weights() const noexcept { return weights; }
    inline const Tensor& get_biases() const noexcept { return biases; }
};

namespace activation {

    /**
     * @brief Rectified Linear Unit (ReLU)
     * Mathematical definition: f(x) = max(0, x)
     * Strictly preserves NaN propagation to prevent masking numerical bugs.
     * Operates in-place on the input tensor.
     */
    inline void relu(Tensor& t) {
        float* ptr = t.data();
        const size_t total_size = t.size();
        for (size_t i = 0; i < total_size; ++i) {
            const float val = ptr[i];
            if (std::isnan(val)) {
                continue; // Strictly propagate NaN
            }
            ptr[i] = (val > 0.0f) ? val : 0.0f;
        }
    }

    /**
     * @brief Thread-safe, non-destructive ReLU variant.
     */
    [[nodiscard]] inline Tensor relu_copy(const Tensor& t) {
        Tensor result = t;
        relu(result);
        return result;
    }

    /**
     * @brief Softmax Activation with full IEEE 754 floating-point safety.
     * Guaranteed safe from division-by-zero, negative infinity cancelation, and NaNs.
     * Operates in-place on the input tensor.
     */
    inline void softmax(Tensor& t) {
        if (t.rows() == 0 || t.cols() == 0) {
            return;
        }

        for (size_t i = 0; i < t.rows(); ++i) {
            // Find max finite value for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            bool has_nan = false;
            bool has_finite = false;

            for (size_t j = 0; j < t.cols(); ++j) {
                const float val = t(i, j);
                if (std::isnan(val)) {
                    has_nan = true;
                } else if (std::isfinite(val)) {
                    has_finite = true;
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }

            // Propagate NaN across the row if present
            if (has_nan) {
                const float nan_val = std::numeric_limits<float>::quiet_NaN();
                for (size_t j = 0; j < t.cols(); ++j) {
                    t(i, j) = nan_val;
                }
                continue;
            }

            // If no finite values exist (e.g. all -Inf or +Inf), distribute uniformly
            if (!has_finite) {
                const float uniform = 1.0f / static_cast<float>(t.cols());
                for (size_t j = 0; j < t.cols(); ++j) {
                    t(i, j) = uniform;
                }
                continue;
            }

            // Exponentiate and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < t.cols(); ++j) {
                const float val = t(i, j);
                if (val == -std::numeric_limits<float>::infinity()) {
                    t(i, j) = 0.0f;
                } else {
                    const float e = std::exp(val - max_val);
                    t(i, j) = e;
                    sum_exp += e;
                }
            }

            // Normalize safely with division-by-zero protection
            if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) {
                const float uniform = 1.0f / static_cast<float>(t.cols());
                for (size_t j = 0; j < t.cols(); ++j) {
                    t(i, j) = uniform;
                }
            } else {
                const float inv_sum = 1.0f / sum_exp;
                for (size_t j = 0; j < t.cols(); ++j) {
                    t(i, j) *= inv_sum;
                }
            }
        }
    }

    /**
     * @brief Thread-safe, non-destructive Softmax variant.
     */
    [[nodiscard]] inline Tensor softmax_copy(const Tensor& t) {
        Tensor result = t;
        softmax(result);
        return result;
    }

} // namespace activation

} // namespace inferc

#endif // INFERC_HPP
