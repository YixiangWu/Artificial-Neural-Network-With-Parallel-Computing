#ifndef NETWORK_CUDA
#define NETWORK_CUDA

#include <cstddef>
#include <string>

#include "network.hpp"


class NetworkCUDA : public Network {
public:
    NetworkCUDA(
        std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
        std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
        const std::string& testImageFile, const std::string& testLabelFile, std::size_t numOfHiddenLayers,
        const std::size_t* numOfNeuronsEachHiddenLayers, std::size_t numOfEpochs,
        std::size_t miniBatchSize, double learningRate
    ) : Network::Network(
        pixelsPerImage, numOfClasses, trainingSize, testSize, trainingImageFile, trainingLabelFile, testImageFile,
        testLabelFile, numOfHiddenLayers, numOfNeuronsEachHiddenLayers, numOfEpochs, miniBatchSize, learningRate
    ) { platform = "CUDA"; memoryAllocate(); }

    ~NetworkCUDA() { memoryFree(); }


private:
    double* label;
    double* helperMatrix;

    /** Allocates memory. */
    virtual void memoryAllocate();

    /** Frees memory. */
    virtual void memoryFree();

    /** Changes biases and weights repeatedly to achieve a minimum cost. */
    virtual void stochasticGradientDescent();

    /** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
    virtual void backpropagation(std::size_t dataPointIndex);

    /** Evaluates the network (biases and weights) with the test data. */
    virtual std::size_t evaluate() const;
};


#endif