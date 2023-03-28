#include <cstddef>
#include <ctime>
#include <iostream>

#include "src/network.hpp"
#include "src/network_cuda.cuh"


const std::size_t PIXELS_PER_IMAGE = 784;
const std::size_t NUM_OF_CLASSES = 10;
const std::size_t TRAINING_DATA_SIZE = 60000;
const std::size_t TEST_DATA_SIZE = 10000;

const std::string TRAINING_IMAGE_FILE = "./data/mnist_training_images.txt";
const std::string TRAINING_LABEL_FILE = "./data/mnist_training_labels.txt";
const std::string TEST_IMAGE_FILE = "./data/mnist_test_images.txt";
const std::string TEST_LABEL_FILE = "./data/mnist_test_labels.txt";


int main() {
    std::size_t numOfNeuronsEachHiddenLayers[1] = {100};
    Network network(
        PIXELS_PER_IMAGE, NUM_OF_CLASSES, TRAINING_DATA_SIZE, TEST_DATA_SIZE,
        TRAINING_IMAGE_FILE, TRAINING_LABEL_FILE, TEST_IMAGE_FILE, TEST_LABEL_FILE,
        1, numOfNeuronsEachHiddenLayers, 30, 32, 3
    );

    NetworkCUDA networkCUDA(
        PIXELS_PER_IMAGE, NUM_OF_CLASSES, TRAINING_DATA_SIZE, TEST_DATA_SIZE,
        TRAINING_IMAGE_FILE, TRAINING_LABEL_FILE, TEST_IMAGE_FILE, TEST_LABEL_FILE,
        1, numOfNeuronsEachHiddenLayers, 30, 32, 3
    );

    std::clock_t start = clock();
    network.train();
    std::clock_t end = clock();
    std::cout << "Time without CUDA: " << (double) (end - start) / CLOCKS_PER_SEC << " sec" << std::endl;

    start = clock();
    networkCUDA.train();
    end = clock();
    std::cout << "Time with CUDA: " << (double) (end - start) / CLOCKS_PER_SEC << " sec" << std::endl;

    return 0;
}