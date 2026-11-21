#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include "InferC.hpp"

using namespace inferc;

// Test 1: Numerical tolerance on fused GEMM vs manual mathematical accumulation
void test_gemm_tolerance() {
    std::cout << "[RUN] Testing Fused GEMM numerical tolerance... ";
    constexpr size_t M = 4;
    constexpr size_t K = 4;
    constexpr size_t N = 3;

    Tensor A(M, K, {
        1.5f, -0.5f,  2.0f,  0.0f,
        0.0f,  1.0f, -1.0f,  3.0f,
       -2.0f,  0.5f,  0.0f,  1.0f,
        1.0f,  1.0f,  1.0f,  1.0f
    });

    Tensor W(K, N, {
        0.5f,  1.0f, -0.5f,
       -1.0f,  0.0f,  2.0f,
        2.0f, -1.5f,  0.5f,
        0.0f,  0.5f, -1.0f
    });

    Tensor B(1, N, {0.1f, -0.2f, 0.3f});

    Tensor C(M, N);
    A.gemm_bias(W, B, C);

    // Analytical expected values for sample 0:
    // C[0,0] = (1.5*0.5 + -0.5*-1.0 + 2.0*2.0 + 0.0*0.0) + 0.1 = 0.75 + 0.5 + 4.0 + 0 + 0.1 = 5.35
    // C[0,1] = (1.5*1.0 + -0.5*0.0 + 2.0*-1.5 + 0.0*0.5) - 0.2 = 1.5 - 3.0 - 0.2 = -1.70
    // C[0,2] = (1.5*-0.5 + -0.5*2.0 + 2.0*0.5 + 0.0*-1.0) + 0.3 = -0.75 - 1.0 + 1.0 + 0.3 = -0.45
    constexpr float EPSILON = 1e-5f;
    assert(std::fabs(C(0, 0) - 5.35f) < EPSILON);
    assert(std::fabs(C(0, 1) - (-1.70f)) < EPSILON);
    assert(std::fabs(C(0, 2) - (-0.45f)) < EPSILON);

    std::cout << "PASS (epsilon < 1e-5)\n";
}

// Test 2: Softmax activation numerical stability & unit probability sum
void test_softmax_tolerance() {
    std::cout << "[RUN] Testing Softmax numerical stability and unit sum... ";
    constexpr size_t BATCH = 3;
    constexpr size_t CLASSES = 5;

    // Test with large dynamic range to verify max-subtraction numerical stability
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

    std::cout << "PASS (no NaN, sum = 1.000000)\n";
}

// Test 3: ReLU non-linear activation bounds
void test_relu_tolerance() {
    std::cout << "[RUN] Testing ReLU non-linear thresholding... ";
    Tensor t(1, 4, {-10.5f, 0.0f, 0.0001f, 15.2f});
    activation::relu(t);

    assert(t(0, 0) == 0.0f);
    assert(t(0, 1) == 0.0f);
    assert(t(0, 2) == 0.0001f);
    assert(t(0, 3) == 15.2f);

    std::cout << "PASS\n";
}

// Test 4: Boundary & Edge Case Protections
void test_edge_cases() {
    std::cout << "[RUN] Testing Tensor edge cases and exception handling... ";

    // Zero-dimension tensor should not crash
    Tensor empty_t(0, 5);
    assert(empty_t.empty());
    assert(empty_t.size() == 0);

    // Dimension mismatch must throw std::invalid_argument
    bool caught_mismatch = false;
    try {
        Tensor A(2, 3);
        Tensor B(4, 5); // Incompatible inner dimension (3 != 4)
        Tensor C;
        A.matmul(B, C);
    } catch (const std::invalid_argument&) {
        caught_mismatch = true;
    }
    assert(caught_mismatch && "Failed to catch matrix dimension mismatch exception");

    // In-place aliasing protection (A.matmul(B, A))
    Tensor A(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor I(2, 2, {1.0f, 0.0f, 0.0f, 1.0f}); // Identity matrix
    A.matmul(I, A); // in-place multiplication
    constexpr float EPSILON = 1e-5f;
    assert(std::fabs(A(0, 0) - 1.0f) < EPSILON);
    assert(std::fabs(A(0, 1) - 2.0f) < EPSILON);
    assert(std::fabs(A(1, 0) - 3.0f) < EPSILON);
    assert(std::fabs(A(1, 1) - 4.0f) < EPSILON);

    std::cout << "PASS (exceptions and aliasing verified)\n";
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
