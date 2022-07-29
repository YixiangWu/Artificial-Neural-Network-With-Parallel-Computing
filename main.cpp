#include <cstddef>

#include "src/network.hpp"


const std::size_t PIXELS_PER_IMAGE = 784;
const std::size_t NUM_OF_CLASSES = 10;
const std::size_t TRAINING_DATA_SIZE = 60000;
const std::size_t TEST_DATA_SIZE = 10000;

const std::string TRAINING_IMAGE_FILE = "./data/mnist_training_images.txt";
const std::string TRAINING_LABEL_FILE = "./data/mnist_training_labels.txt";
const std::string TEST_IMAGE_FILE = "./data/mnist_test_images.txt";
const std::string TEST_LABEL_FILE = "./data/mnist_test_labels.txt";


int main() {

    // Network Size, Number of Epochs, Mini-Batch Size, Learning Rate
    // where Network Size is {PIXELS_PER_IMAGE, ..., NUM_OF_CLASSES} and ... are hidden layers

    // {784, 100, 10}, Number of Epochs, 10, 3 -> can reach 96% within 5 Epochs

    // {784, 100, 10}, 30, 10, 3
    size_t numOfNeuronsEachHiddenLayers[1] = {100};
    Network network(
        PIXELS_PER_IMAGE, NUM_OF_CLASSES, TRAINING_DATA_SIZE, TEST_DATA_SIZE,
        TRAINING_IMAGE_FILE, TRAINING_LABEL_FILE, TEST_IMAGE_FILE, TEST_LABEL_FILE,
        1, numOfNeuronsEachHiddenLayers, 30, 10, 3
    );

    network.train();

    return 0;
}