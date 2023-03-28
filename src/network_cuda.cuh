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
        const std::size_t* numOfNeuronsEachHiddenLayers, unsigned int numOfEpochs,
        std::size_t miniBatchSize, double learningRate
    );

    ~NetworkCUDA();


protected:
    double* biases_;
    double* weights_;
    double* nablaBiases_;
    double* nablaWeights_;
    double* deltaNablaBiases_;
    double* deltaNablaWeights_;

    double* activations_;
    double* trainingLabels_;

    /** Changes biases and weights repeatedly to achieve a minimum cost. */
    virtual void stochasticGradientDescent();

    /** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
    void backpropagation();

    /** Evaluates the network (biases and weights) with the test data. */
    virtual std::size_t evaluate() const;
};


#endif