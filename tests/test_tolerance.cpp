#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include <limits>
#include "InferC.hpp"

using namespace inferc;

namespace {

std::string build_binary_payload(uint32_t magic, uint16_t version, uint16_t endianness,
                                 uint32_t in_dim, uint32_t out_dim,
                                 const std::vector<float>& weights,
                                 const std::vector<float>& biases) {
    std::string buf;
    auto append = [&](const void* ptr, size_t bytes) {
        buf.append(reinterpret_cast<const char*>(ptr), bytes);
    };
    ModelFileHeader hdr{magic, version, endianness};
    append(&hdr, sizeof(hdr));
    append(&in_dim, sizeof(in_dim));
    append(&out_dim, sizeof(out_dim));
    if (!weights.empty()) {
        append(weights.data(), weights.size() * sizeof(float));
    }
    if (!biases.empty()) {
        append(biases.data(), biases.size() * sizeof(float));
    }
    return buf;
}

} // namespace

// Test 1: Numerical tolerance on fused GEMM vs manual mathematical accumulation
void test_gemm_tolerance() {
    std::cout << "[RUN] Testing Fused GEMM numerical tolerance (unrolled + tail K)... ";
    constexpr float EPSILON = 1e-5f;

    // 4x4 * 4x3 (exact 4-way unrolled path)
    Tensor A(4, 4, {
        1.5f, -0.5f,  2.0f,  0.0f,
        0.0f,  1.0f, -1.0f,  3.0f,
       -2.0f,  0.5f,  0.0f,  1.0f,
        1.0f,  1.0f,  1.0f,  1.0f
    });
    Tensor W(4, 3, {
        0.5f,  1.0f, -0.5f,
       -1.0f,  0.0f,  2.0f,
        2.0f, -1.5f,  0.5f,
        0.0f,  0.5f, -1.0f
    });
    Tensor B(1, 3, {0.1f, -0.2f, 0.3f});
    Tensor C(4, 3);
    A.gemm_bias(W, B, C);

    assert(std::fabs(C(0, 0) - 5.35f) < EPSILON);
    assert(std::fabs(C(0, 1) - (-1.70f)) < EPSILON);
    assert(std::fabs(C(0, 2) - (-0.45f)) < EPSILON);

    // Non-multiple-of-4 reduction dimension (K=5) exercises both unrolled and remainder loops
    Tensor A5(2, 5, {
        1.0f, 2.0f, -1.0f, 0.5f, 3.0f,
       -2.0f, 0.0f,  4.0f, 1.0f, -0.5f
    });
    Tensor W5(5, 3, {
        1.0f,  0.0f, -1.0f,
        0.5f,  1.0f,  0.5f,
       -1.0f,  2.0f,  0.0f,
        2.0f, -1.0f,  1.0f,
        0.25f, 0.5f, -0.5f
    });
    Tensor B5(1, 3, {0.5f, -0.5f, 1.0f});
    Tensor C5(2, 3);
    A5.gemm_bias(W5, B5, C5);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float expected = B5(0, j);
            for (size_t k = 0; k < 5; ++k) {
                expected += A5(i, k) * W5(k, j);
            }
            assert(std::fabs(C5(i, j) - expected) < EPSILON);
        }
    }

    // Verify matmul on K=5 as well (unrolled + tail)
    Tensor M5 = A5.matmul(W5);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(std::fabs(M5(i, j) - (C5(i, j) - B5(0, j))) < EPSILON);
        }
    }

    std::cout << "PASS (epsilon < 1e-5)\n";
}

// Test 2: Softmax activation numerical stability, saturation, and NaN propagation
void test_softmax_tolerance() {
    std::cout << "[RUN] Testing Softmax stability, Inf saturation, and NaN handling... ";
    constexpr size_t BATCH = 3;
    constexpr size_t CLASSES = 5;

    Tensor logits(BATCH, CLASSES, {
        1000.0f, 1001.0f, 1002.0f,  999.0f, 1000.0f,
         -50.0f,  -49.0f,  -48.0f,  -52.0f,  -51.0f,
           0.0f,    0.0f,    0.0f,    0.0f,    0.0f
    });
    activation::softmax(logits);

    constexpr float EPSILON = 1e-5f;
    for (size_t b = 0; b < BATCH; ++b) {
        float row_sum = 0.0f;
        for (size_t c = 0; c < CLASSES; ++c) {
            float p = logits(b, c);
            assert(!std::isnan(p) && !std::isinf(p) && "Softmax produced NaN or Inf");
            assert(p >= 0.0f && p <= 1.0f && "Probability outside valid [0, 1] range");
            row_sum += p;
        }
        assert(std::fabs(row_sum - 1.0f) < EPSILON && "Probabilities do not sum to 1.0");
    }

    const float inf = std::numeric_limits<float>::infinity();
    const float qnan = std::numeric_limits<float>::quiet_NaN();

    Tensor sat_pos(1, 3, {inf, 1.0f, inf});
    activation::softmax(sat_pos);
    assert(std::fabs(sat_pos(0, 0) - 0.5f) < EPSILON);
    assert(sat_pos(0, 1) == 0.0f);
    assert(std::fabs(sat_pos(0, 2) - 0.5f) < EPSILON);

    Tensor sat_neg(1, 3, {-inf, -inf, -inf});
    activation::softmax(sat_neg);
    for (size_t j = 0; j < 3; ++j) {
        assert(std::fabs(sat_neg(0, j) - (1.0f / 3.0f)) < EPSILON);
    }

    Tensor nan_row(1, 3, {1.0f, qnan, 2.0f});
    activation::softmax(nan_row);
    for (size_t j = 0; j < 3; ++j) {
        assert(std::isnan(nan_row(0, j)));
    }

    Tensor empty_sm(0, 0);
    activation::softmax(empty_sm);
    assert(empty_sm.empty());

    std::cout << "PASS\n";
}

// Test 3: ReLU non-linear activation bounds & quiet NaN preservation
void test_relu_tolerance() {
    std::cout << "[RUN] Testing ReLU thresholding and quiet NaN preservation... ";
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    Tensor t(1, 5, {-10.5f, 0.0f, 0.0001f, 15.2f, qnan});
    activation::relu(t);

    assert(t(0, 0) == 0.0f);
    assert(t(0, 1) == 0.0f);
    assert(t(0, 2) == 0.0001f);
    assert(t(0, 3) == 15.2f);
    assert(std::isnan(t(0, 4)));

    std::cout << "PASS\n";
}

// Test 4: Boundary, aliasing, add_bias, and binary serialization invariants
void test_edge_cases() {
    std::cout << "[RUN] Testing edge cases, add_bias, aliasing, and binary serialization... ";
    constexpr float EPSILON = 1e-5f;

    Tensor empty_t(0, 5);
    assert(empty_t.empty());
    assert(empty_t.size() == 0);

    bool caught_mismatch = false;
    try {
        Tensor A(2, 3);
        Tensor B(4, 5);
        Tensor C;
        A.matmul(B, C);
    } catch (const std::invalid_argument&) {
        caught_mismatch = true;
    }
    assert(caught_mismatch && "Failed to catch matrix dimension mismatch exception");

    // In-place aliasing for matmul and gemm_bias
    Tensor A(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor I(2, 2, {1.0f, 0.0f, 0.0f, 1.0f});
    Tensor B0(1, 2, {0.5f, -0.5f});
    A.matmul(I, A);
    assert(std::fabs(A(0, 0) - 1.0f) < EPSILON);
    assert(std::fabs(A(1, 1) - 4.0f) < EPSILON);

    A.gemm_bias(I, B0, A);
    assert(std::fabs(A(0, 0) - 1.5f) < EPSILON);
    assert(std::fabs(A(0, 1) - 1.5f) < EPSILON);
    assert(std::fabs(A(1, 0) - 3.5f) < EPSILON);
    assert(std::fabs(A(1, 1) - 3.5f) < EPSILON);

    // add_bias normal & self-aliasing
    Tensor M(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor bias(1, 2, {10.0f, -2.0f});
    M.add_bias(bias);
    assert(std::fabs(M(0, 0) - 11.0f) < EPSILON);
    assert(std::fabs(M(1, 1) - 2.0f) < EPSILON);
    bias.add_bias(bias);
    assert(std::fabs(bias(0, 0) - 20.0f) < EPSILON);
    assert(std::fabs(bias(0, 1) - (-4.0f)) < EPSILON);

    // Valid binary deserialization + strong exception guarantee on failure
    DenseLayer layer(2, 2);
    const std::string valid_blob = build_binary_payload(
        EXPECTED_MAGIC, EXPECTED_VERSION, EXPECTED_ENDIANNESS,
        2, 2, {1.0f, 2.0f, 3.0f, 4.0f}, {0.5f, -0.5f}
    );
    std::istringstream good_stream(valid_blob, std::ios::binary);
    layer.load_weights(good_stream);
    assert(std::fabs(layer.get_weights()(1, 1) - 4.0f) < EPSILON);
    assert(std::fabs(layer.get_biases()(0, 1) - (-0.5f)) < EPSILON);

    auto expect_load_error = [&](const std::string& payload) {
        std::istringstream bad_stream(payload, std::ios::binary);
        bool threw = false;
        try {
            layer.load_weights(bad_stream);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
        // Existing weights must remain untouched after failed load
        assert(std::fabs(layer.get_weights()(1, 1) - 4.0f) < EPSILON);
    };

    expect_load_error(build_binary_payload(0xDEADBEEF, EXPECTED_VERSION, EXPECTED_ENDIANNESS, 2, 2, {0, 0, 0, 0}, {0, 0}));
    expect_load_error(build_binary_payload(EXPECTED_MAGIC, 99, EXPECTED_ENDIANNESS, 2, 2, {0, 0, 0, 0}, {0, 0}));
    expect_load_error(build_binary_payload(EXPECTED_MAGIC, EXPECTED_VERSION, 0x0201, 2, 2, {0, 0, 0, 0}, {0, 0}));
    expect_load_error(build_binary_payload(EXPECTED_MAGIC, EXPECTED_VERSION, EXPECTED_ENDIANNESS, 2, 2, {0, 0, 0}, {}));
    expect_load_error(build_binary_payload(
        EXPECTED_MAGIC, EXPECTED_VERSION, EXPECTED_ENDIANNESS,
        2, 2, {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f}, {0.5f, -0.5f}
    ));

    std::cout << "PASS\n";
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << " InferC Test Suite: Numerical Tolerance & Invariants\n";
    std::cout << "=====================================================\n";

    try {
        test_gemm_tolerance();
        test_softmax_tolerance();
        test_relu_tolerance();
        test_edge_cases();
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED]: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n[ALL TESTS PASSED]: Core mathematical invariants verified.\n";
    return 0;
}
