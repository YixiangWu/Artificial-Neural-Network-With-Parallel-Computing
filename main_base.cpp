#include <cstddef>

#include "src/constant.hpp"
#include "src/network.hpp"


int main() {
    std::size_t numOfNeuronsEachHiddenLayers[1] = {100};
    Network network(
        constant::PIXELS_PER_IMAGE, constant::NUM_OF_CLASSES, constant::TRAINING_DATA_SIZE,
        constant::TEST_DATA_SIZE, constant::TRAINING_IMAGE_FILE, constant::TRAINING_LABEL_FILE,
        constant::TEST_IMAGE_FILE, constant::TEST_LABEL_FILE,
        1, numOfNeuronsEachHiddenLayers, 30, 24, 3
    );

    network.train();
    return 0;
}