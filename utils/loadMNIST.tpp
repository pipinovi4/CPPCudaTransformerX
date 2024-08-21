#ifndef LOADMNIST_TPP
#define LOADMNIST_TPP

#include "loadMNIST.h"

// Declare the function as inline to avoid multiple definitions across different translation units
std::vector<std::vector<std::uint8_t>> loadMNISTImages(const std::string& filename) {
    std::ifstream imageStream(filename, std::ios::binary);
    if (!imageStream.is_open()) {
        throw std::runtime_error("Could not open image file.");
    }

    // Read dimensions
    readUInt32(imageStream);
    const std::uint32_t num_images = readUInt32(imageStream);
    const std::uint32_t num_rows = readUInt32(imageStream);
    const std::uint32_t num_cols = readUInt32(imageStream);

    // Read image data
    std::vector<std::vector<std::uint8_t>> images(num_images, std::vector<std::uint8_t>(num_rows * num_cols));
    for (std::uint32_t i = 0; i < num_images; ++i) {
        imageStream.read(reinterpret_cast<char*>(images[i].data()), num_rows * num_cols);
    }
    imageStream.close();

    return images;
}

std::vector<std::uint8_t> loadMNISTLabels(const std::string& filename) {
    std::ifstream labelStream(filename, std::ios::binary);
    if (!labelStream.is_open()) {
        throw std::runtime_error("Could not open label file.");
    }

    // Read number of labels
    readUInt32(labelStream); // magic_number
    const std::uint32_t num_labels = readUInt32(labelStream);

    // Read label data
    std::vector<std::uint8_t> labels(num_labels);
    labelStream.read(reinterpret_cast<char*>(labels.data()), num_labels);
    labelStream.close();

    return labels;
}

#endif // LOADMNIST_TPP
