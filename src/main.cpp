#include <iostream>
#include <fstream>
#include <vector>
#include "InferC.hpp"

using namespace inferc;

int main() {
    std::cout << "[INFO] Initializing InferC Engine...\n";

    // 1. Define Network Architecture
    // Example: Input(4) -> Dense(16) -> ReLU -> Dense(3) -> Softmax
    constexpr size_t INPUT_DIM = 4;
    constexpr size_t HIDDEN_DIM = 16;
    constexpr size_t OUTPUT_DIM = 3;

    try {
        DenseLayer layer1(INPUT_DIM, HIDDEN_DIM);
        DenseLayer layer2(HIDDEN_DIM, OUTPUT_DIM);

        // 2. Load Pre-trained Weights with Strict Protocol and Stream Verification
        std::cout << "[INFO] Loading weights from model_weights.bin...\n";
        std::ifstream weight_file("model_weights.bin", std::ios::binary);
        
        if (!weight_file.is_open()) {
            std::cerr << "[ERROR] Could not open model_weights.bin! Run the Python export script first.\n";
            return 1;
        }

        // Enable badbit exception to catch low-level hardware or filesystem corruption
        weight_file.exceptions(std::ifstream::badbit);

        layer1.load_weights(weight_file);
        layer2.load_weights(weight_file);

        // Verify clean EOF: ensure no extraneous, injected, or unparsed trailing bytes exist
        char trailing_byte = 0;
        if (weight_file.read(&trailing_byte, 1) || !weight_file.eof()) {
            throw std::runtime_error("Unexpected trailing data found in weights file: file size exceeds expected model payload.");
        }
        weight_file.close();
        std::cout << "[INFO] Weights verified and loaded successfully.\n";

        // 3. Create Input Tensor with Safe, Bounded Population
        constexpr size_t BATCH_SIZE = 2;
        Tensor input(BATCH_SIZE, INPUT_DIM);

        // Raw input sample matrix
        const std::vector<std::vector<float>> sample_data = {
            {1.0f,  0.5f, -1.2f, 0.1f}, // Sample 1
            {0.0f, -0.5f,  2.0f, 1.1f}  // Sample 2
        };

        // Bounded copy loop guarding against dimension mismatches
        for (size_t b = 0; b < input.rows(); ++b) {
            if (b < sample_data.size()) {
                for (size_t f = 0; f < input.cols(); ++f) {
                    if (f < sample_data[b].size()) {
                        input(b, f) = sample_data[b][f];
                    }
                }
            }
        }

        std::cout << "[INFO] Running Inference...\n";

        // 4. Forward Pass with Safe Non-Destructive / In-Place Operations
        // Layer 1: X * W1 + B1
        Tensor hidden = layer1.forward(input);
        
        // ReLU Activation (NaN-safe)
        activation::relu(hidden);
        
        // Layer 2: H * W2 + B2
        Tensor output = layer2.forward(hidden);
        
        // Softmax Probabilities (Division-by-Zero and IEEE 754 safe)
        activation::softmax(output);

        // 5. Output Results
        std::cout << "\n[RESULT] Inference Probabilities:\n";
        for (size_t i = 0; i < output.rows(); ++i) {
            std::cout << "Sample " << i << ": [ ";
            for (size_t j = 0; j < output.cols(); ++j) {
                std::cout << output(i, j) << " ";
            }
            std::cout << "]\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[CRITICAL ERROR] InferC execution halted: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
