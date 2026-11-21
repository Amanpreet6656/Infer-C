#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>
#include <cassert>
#include "InferC.hpp"

using namespace inferc;

// Naive triple-loop (i-j-k) matrix multiplication with separate bias addition pass
void naive_gemm_bias(const float* A, const float* B, const float* bias, float* C,
                     size_t M, size_t K, size_t N) {
    // 1. Matmul: C = A * B using naive i-j-k loop ordering (strided access on B)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    // 2. Separate bias addition pass (forces reload from memory)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            C[i * N + j] += bias[j];
        }
    }
}

struct BenchResult {
    size_t M, K, N;
    size_t iterations;
    double naive_us;
    double inferc_us;
    double speedup;
    float max_diff;
    double gflops;
};

BenchResult run_benchmark(size_t M, size_t K, size_t N, size_t iterations) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Tensor input(M, K);
    Tensor weights(K, N);
    Tensor bias(1, N);

    for (size_t i = 0; i < input.size(); ++i) input.data()[i] = dist(rng);
    for (size_t i = 0; i < weights.size(); ++i) weights.data()[i] = dist(rng);
    for (size_t i = 0; i < bias.size(); ++i) bias.data()[i] = dist(rng);

    Tensor out_inferc(M, N);
    std::vector<float> out_naive(M * N, 0.0f);

    // Warm-up passes
    for (int w = 0; w < 10; ++w) {
        input.gemm_bias(weights, bias, out_inferc);
        naive_gemm_bias(input.data(), weights.data(), bias.data(), out_naive.data(), M, K, N);
    }

    // Verify numerical tolerance: max absolute difference < 1e-4
    float max_diff = 0.0f;
    for (size_t i = 0; i < M * N; ++i) {
        float diff = std::fabs(out_inferc.data()[i] - out_naive[i]);
        if (diff > max_diff) max_diff = diff;
    }
    assert(max_diff < 1e-4f && "Numerical equivalence check failed");

    // Benchmark Infer-C Fused GEMM
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        input.gemm_bias(weights, bias, out_inferc);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double inferc_total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double inferc_us = inferc_total_us / iterations;

    // Benchmark Naive GEMM
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < iterations; ++it) {
        naive_gemm_bias(input.data(), weights.data(), bias.data(), out_naive.data(), M, K, N);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double naive_total_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    double naive_us = naive_total_us / iterations;

    double speedup = naive_us / inferc_us;
    // FLOPS for GEMM + Bias = 2 * M * K * N + M * N
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N);
    double gflops = (flops / (inferc_us * 1e-6)) / 1e9;

    return {M, K, N, iterations, naive_us, inferc_us, speedup, max_diff, gflops};
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << " InferC Performance Benchmark: Fused i-k-j Kernel vs Naive i-j-k GEMM\n";
    std::cout << "========================================================================\n\n";

    std::vector<std::tuple<size_t, size_t, size_t, size_t>> test_cases = {
        {16, 64, 32, 10000},    // Micro-batch edge layer
        {64, 128, 128, 5000},   // Dense hidden layer
        {128, 256, 256, 1000},  // Mid-sized inference pass
        {256, 256, 256, 500}    // Standard square benchmark
    };

    std::cout << std::left
              << std::setw(18) << "Dimensions (MxKxN)"
              << std::setw(12) << "Iterations"
              << std::setw(14) << "Naive (us)"
              << std::setw(14) << "InferC (us)"
              << std::setw(12) << "Speedup"
              << std::setw(14) << "Max Diff"
              << std::setw(10) << "GFLOPS" << "\n";
    std::cout << std::string(94, '-') << "\n";

    for (const auto& [M, K, N, iters] : test_cases) {
        BenchResult res = run_benchmark(M, K, N, iters);
        std::string dims = std::to_string(res.M) + "x" + std::to_string(res.K) + "x" + std::to_string(res.N);

        std::cout << std::left
                  << std::setw(18) << dims
                  << std::setw(12) << res.iterations
                  << std::setw(14) << std::fixed << std::setprecision(2) << res.naive_us
                  << std::setw(14) << std::fixed << std::setprecision(2) << res.inferc_us
                  << std::setw(12) << std::fixed << std::setprecision(2) << (std::to_string(res.speedup).substr(0, 4) + "x")
                  << std::setw(14) << std::scientific << std::setprecision(2) << res.max_diff
                  << std::setw(10) << std::fixed << std::setprecision(2) << res.gflops
                  << "\n";
    }

    std::cout << "\n[VERIFIED] All numerical tolerances satisfied (epsilon < 1e-4).\n";
    return 0;
}
