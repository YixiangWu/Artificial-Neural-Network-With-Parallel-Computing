#ifndef CONSTANT
#define CONSTANT

#include <cstddef>


namespace constant {
    constexpr std::size_t PIXELS_PER_IMAGE = 784;
    constexpr std::size_t NUM_OF_CLASSES = 10;
    constexpr std::size_t TRAINING_DATA_SIZE = 60000;
    constexpr std::size_t TEST_DATA_SIZE = 10000;

    constexpr char TRAINING_IMAGE_FILE[] = "./data/mnist_training_images.txt";
    constexpr char TRAINING_LABEL_FILE[] = "./data/mnist_training_labels.txt";
    constexpr char TEST_IMAGE_FILE[] = "./data/mnist_test_images.txt";
    constexpr char TEST_LABEL_FILE[] = "./data/mnist_test_labels.txt";
}


#endif