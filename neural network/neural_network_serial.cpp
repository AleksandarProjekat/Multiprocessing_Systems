/*

g++ -std=c++17 -o neural_network_serial neural_network_serial.cpp
./neural_network_serial

*/

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

// Activation functions
inline float relu(float x) {
    return std::max(0.0f, x);
}

inline float softmax(const std::vector<float>& outputs, int index) {
    float max_val = *std::max_element(outputs.begin(), outputs.end());
    float sum = 0.0;
    for (float val : outputs) {
        sum += std::exp(val - max_val);
    }
    return std::exp(outputs[index] - max_val) / sum;
}

// Function to load MNIST dataset
void load_mnist(const std::string& image_file, const std::string& label_file, 
                std::vector<std::vector<float>>& images, std::vector<int>& labels, int num_samples) {
    std::ifstream img_stream(image_file, std::ios::binary);
    std::ifstream lbl_stream(label_file, std::ios::binary);

    if (!img_stream.is_open() || !lbl_stream.is_open()) {
        throw std::runtime_error("Failed to open dataset files!");
    }

    // Read image file headers
    int32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;
    img_stream.read(reinterpret_cast<char*>(&magic_number), 4);
    img_stream.read(reinterpret_cast<char*>(&num_images), 4);
    img_stream.read(reinterpret_cast<char*>(&rows), 4);
    img_stream.read(reinterpret_cast<char*>(&cols), 4);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    // Read label file headers
    int32_t label_magic = 0, num_labels = 0;
    lbl_stream.read(reinterpret_cast<char*>(&label_magic), 4);
    lbl_stream.read(reinterpret_cast<char*>(&num_labels), 4);

    label_magic = __builtin_bswap32(label_magic);
    num_labels = __builtin_bswap32(num_labels);

    if (num_samples > num_images || num_samples > num_labels) {
        throw std::runtime_error("Requested number of samples exceeds dataset size.");
    }

    images.resize(num_samples, std::vector<float>(rows * cols));
    labels.resize(num_samples);

    // Load image and label data
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < rows * cols; ++j) {
            unsigned char pixel = 0;
            img_stream.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = pixel / 255.0f; // Normalize pixel values
        }
        unsigned char label = 0;
        lbl_stream.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = label;
    }
}

// Implementation of the neural network
class NeuralNetwork {
public:
    NeuralNetwork(int input_size, const std::vector<int>& hidden_layers, int output_size)
        : input_size(input_size), hidden_layers(hidden_layers), output_size(output_size) {
        initialize_weights();
    }

    void train(const std::vector<std::vector<float>>& train_data,
               const std::vector<int>& train_labels,
               float learning_rate) {
        // Train for one epoch
        for (size_t i = 0; i < train_data.size(); ++i) {
            // Forward pass
            std::vector<float> outputs = forward(train_data[i]);

            // Backward pass
            backward(outputs, train_labels[i], learning_rate);
        }
    }

private:
    int input_size;
    std::vector<int> hidden_layers;
    int output_size;
    std::vector<std::vector<std::vector<float>>> weights;

    void initialize_weights() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5, 0.5);

        for (size_t i = 0; i < hidden_layers.size() + 1; ++i) {
            int rows = (i == 0) ? input_size : hidden_layers[i - 1];
            int cols = (i < hidden_layers.size()) ? hidden_layers[i] : output_size;
            weights.emplace_back(rows, std::vector<float>(cols));

            for (auto& row : weights.back()) {
                for (auto& weight : row) {
                    weight = dist(rng);
                }
            }
        }
    }

    std::vector<float> forward(const std::vector<float>& inputs) {
        std::vector<float> current_outputs = inputs;
        for (size_t layer = 0; layer < weights.size(); ++layer) {
            std::vector<float> next_outputs(weights[layer][0].size(), 0.0);

            for (size_t j = 0; j < weights[layer][0].size(); ++j) {
                for (size_t i = 0; i < current_outputs.size(); ++i) {
                    next_outputs[j] += current_outputs[i] * weights[layer][i][j];
                }
                next_outputs[j] = (layer == weights.size() - 1) ? softmax(next_outputs, j) : relu(next_outputs[j]);
            }
            current_outputs = next_outputs;
        }
        return current_outputs;
    }

    void backward(const std::vector<float>& outputs, int label, float learning_rate) {
        // Placeholder for backpropagation implementation
    }
};

int main() {
    int input_size = 784; // MNIST images are 28x28
    std::vector<int> hidden_layers = {128, 64};
    int output_size = 10; // 10 classes for classification

    NeuralNetwork nn(input_size, hidden_layers, output_size);

    std::cout << "Starting sequential training of the neural network...\n";

    // Load MNIST data
    std::vector<std::vector<float>> train_data;
    std::vector<int> train_labels;
    load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", train_data, train_labels, 10000);

    float learning_rate = 0.01;

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    nn.train(train_data, train_labels, learning_rate);

    // End time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;

    std::cout << "Total training time (1 epoch): " << elapsed_time.count() << " seconds.\n";

    return 0;
}

