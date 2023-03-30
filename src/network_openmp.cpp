#include "network_openmp.hpp"
#include "operation.cpp"

#include <cstddef>
#include <iostream>
#include <omp.h>


/** Changes biases and weights repeatedly to achieve a minimum cost. */
void NetworkOpenMP::stochasticGradientDescent() {
    for (unsigned int e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {
            #pragma omp parallel for num_threads(omp_get_max_threads()) shared(biasesSize, nablaBiases)
            for (std::size_t j = 0; j < biasesSize; ++j)
                nablaBiases[j] = 0;  // initialize all elements to 0

            #pragma omp parallel for num_threads(omp_get_max_threads()) shared(weightsSize, nablaWeights)
            for (std::size_t j = 0; j < weightsSize; ++j)
                nablaWeights[j] = 0;  // initialize all elements to 0

            // look for the proper partial derivatives that reduce the cost
            #pragma omp parallel for num_threads(omp_get_max_threads()) \
                shared(miniBatchSize, biasesSize, weightsSize, nablaBiases, nablaWeights, \
                trainingDataIndices) private(deltaNablaBiases, deltaNablaWeights)
            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                deltaNablaBiases = new double[biasesSize];
                deltaNablaWeights = new double[weightsSize];

                backpropagation(trainingDataIndices[i * miniBatchSize + j], deltaNablaBiases, deltaNablaWeights);

                // update partial derivatives of the cost function with respect to biases
                for (std::size_t k = 0; k < biasesSize; ++k)
                    nablaBiases[k] += deltaNablaBiases[k];

                // update partial derivatives of the cost function with respect to weights
                for (std::size_t k = 0; k < weightsSize; ++k)
                    nablaWeights[k] += deltaNablaWeights[k];

                delete[] deltaNablaBiases;
                delete[] deltaNablaWeights;
            }

            // reduce the cost by changing biases
            #pragma omp parallel for num_threads(omp_get_max_threads()) \
                shared(miniBatchSize, learningRate, biasesSize, biases, nablaBiases)
            for (std::size_t j = 0; j < biasesSize; ++j)
                biases[j] -= (learningRate / (double) miniBatchSize) * nablaBiases[j];

            // reduce the cost by changing weights
            #pragma omp parallel for num_threads(omp_get_max_threads()) \
                shared(miniBatchSize, learningRate, weightsSize, weights, nablaWeights)
            for (std::size_t j = 0; j < weightsSize; ++j)
                weights[j] -= (learningRate / (double) miniBatchSize) * nablaWeights[j];
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t NetworkOpenMP::evaluate() const {
    std::size_t numOfPassedTests = 0;
    std::size_t prediction;
    std::size_t offset = biasesSize + numOfNeuronsEachLayer[0] - numOfNeuronsEachLayer[numOfLayers - 1];

    double* zs;
    double* activations;
    double* weightsDotActivations;  // helper array

    #pragma omp parallel for num_threads(omp_get_max_threads()) reduction(+:numOfPassedTests) \
        shared(pixelsPerImage, testSize, testImages, testLabels, numOfNeuronsEachLayer, numOfLayers, \
        weights, biases, biasesSize, offset) private(zs, activations, weightsDotActivations, prediction)
    for (std::size_t m = 0; m < testSize; ++m) {
        prediction = offset;

        zs = new double[biasesSize];
        activations = new double[biasesSize + numOfNeuronsEachLayer[0]];
        std::copy(testImages + m * pixelsPerImage, testImages + (m + 1) * pixelsPerImage, activations);

        // feed forward
        weightsDotActivations = new double[biasesSize];
        for (std::size_t l = 1, i = 0, j = 0, k = 0; l < numOfLayers; ++l) {
            // z = weights dot activations + bias
            dotMatrixVector(
                numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
                weights + j, activations + k, weightsDotActivations + i
            );
            add(numOfNeuronsEachLayer[l], weightsDotActivations + i, biases + i, zs + i);

            // new activation = sigmoid(z) = 1 / (1 + (e ^ -z))
            k += numOfNeuronsEachLayer[l - 1];
            sigmoid(numOfNeuronsEachLayer[l], zs + i, activations + k);

            i += numOfNeuronsEachLayer[l];
            j += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
        }

        for (std::size_t i = offset + 1; i < biasesSize + numOfNeuronsEachLayer[0]; ++i) {
            if (activations[i] > activations[prediction])
                prediction = i;
        }

        if (testLabels[(prediction - offset) + (numOfClasses * m)] == 1)
            ++numOfPassedTests;

        delete[] zs;
        delete[] activations;
        delete[] weightsDotActivations;
    }
    return numOfPassedTests;
}