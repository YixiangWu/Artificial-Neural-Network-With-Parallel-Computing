#include "network_openmp.hpp"
#include "operation.cpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <omp.h>


void NetworkOpenMP::setNumOfThreads(std::size_t numOfThreads) {
    this->numOfThreads = std::min(numOfThreads,  static_cast<size_t>(omp_get_max_threads()));
    printInfo();
}

/** Allocates memory. */
void NetworkOpenMP::memoryAllocate() {
    printInfo();
    generateBiasesAndWeights();

    nablaBiases = new double[biasesSize];
    nablaWeights = new double[weightsSize];

    deltaNablaBiases = new double[biasesSize * numOfThreads];
    deltaNablaWeights = new double[weightsSize * numOfThreads];

    zs = new double[biasesSize * numOfThreads];
    cost = new double[biasesSize * numOfThreads];
    zPrimes = new double[biasesSize * numOfThreads];
    activations = new double[activationsSize * numOfThreads];
    weightsDotActivations = new double[biasesSize * numOfThreads];
}

/** Frees memory. */
void NetworkOpenMP::memoryFree() {
    delete[] biases;
    delete[] weights;
    delete[] nablaBiases;
    delete[] nablaWeights;

    delete[] deltaNablaBiases;
    delete[] deltaNablaWeights;

    delete[] zs;
    delete[] cost;
    delete[] zPrimes;
    delete[] activations;
    delete[] weightsDotActivations;
}

/** Helps allocate new memory at the end of loading data. */
void NetworkOpenMP::loadDataHelper() {
    printInfo();
    delete[] activations;
    activations = new double[activationsSize * numOfThreads];
}

/** Changes biases and weights repeatedly to achieve a minimum cost. */
void NetworkOpenMP::stochasticGradientDescent() {
    for (std::size_t e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {
            #pragma omp parallel for num_threads(numOfThreads) shared(biasesSize, nablaBiases)
            for (std::size_t j = 0; j < biasesSize; ++j)
                nablaBiases[j] = 0;  // initialize all elements to 0

            #pragma omp parallel for num_threads(numOfThreads) shared(weightsSize, nablaWeights)
            for (std::size_t j = 0; j < weightsSize; ++j)
                nablaWeights[j] = 0;  // initialize all elements to 0

            // look for the proper partial derivatives that reduce the cost
            #pragma omp parallel for num_threads(numOfThreads) shared(miniBatchSize, \
                biasesSize, weightsSize, nablaBiases, nablaWeights, deltaNablaBiases, \
                deltaNablaWeights, trainingDataIndices) schedule(dynamic)
            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                backpropagation(trainingDataIndices[i * miniBatchSize + j]);

                // update partial derivatives of the cost function with respect to biases
                for (std::size_t k = 0; k < biasesSize; ++k)
                    nablaBiases[k] += deltaNablaBiases[k + omp_get_thread_num() * biasesSize];

                // update partial derivatives of the cost function with respect to weights
                for (std::size_t k = 0; k < weightsSize; ++k)
                    nablaWeights[k] += deltaNablaWeights[k + omp_get_thread_num() * weightsSize];
            }

            // reduce the cost by changing biases
            #pragma omp parallel for num_threads(numOfThreads) \
                shared(miniBatchSize, learningRate, biasesSize, biases, nablaBiases)
            for (std::size_t j = 0; j < biasesSize; ++j)
                biases[j] -= (learningRate / (double) miniBatchSize) * nablaBiases[j];

            // reduce the cost by changing weights
            #pragma omp parallel for num_threads(numOfThreads) \
                shared(miniBatchSize, learningRate, weightsSize, weights, nablaWeights)
            for (std::size_t j = 0; j < weightsSize; ++j)
                weights[j] -= (learningRate / (double) miniBatchSize) * nablaWeights[j];
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
void NetworkOpenMP::backpropagation(std::size_t dataPointIndex) {
    std::size_t a = 0, b = 0, w = 0;
    double* deltaNablaBiases_ = deltaNablaBiases + omp_get_thread_num() * biasesSize;
    double* deltaNablaWeights_ = deltaNablaWeights + omp_get_thread_num() * weightsSize;

    double* zs_ = zs + omp_get_thread_num() * biasesSize;
    double* cost_ = cost + omp_get_thread_num() * biasesSize;
    double* zPrimes_ = zPrimes + omp_get_thread_num() * biasesSize;
    double* activations_ = activations + omp_get_thread_num() * activationsSize;
    double* weightsDotActivations_ = weightsDotActivations + omp_get_thread_num() * biasesSize;

    std::copy(
        trainingImages + dataPointIndex * pixelsPerImage,
        trainingImages + (dataPointIndex + 1) * pixelsPerImage, activations_
    );

    // feed forward
    for (std::size_t l = 1; l < numOfLayers; ++l) {
        dotMatrixVector(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            weights + w, activations_ + a, weightsDotActivations_ + b
        );
        add(numOfNeuronsEachLayer[l], weightsDotActivations_ + b, biases + b, zs_ + b);

        a += numOfNeuronsEachLayer[l - 1];
        sigmoid(numOfNeuronsEachLayer[l], zs_ + b, activations_ + a);

        b += numOfNeuronsEachLayer[l];
        w += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    // backward pass for the output layer
    b -= numOfNeuronsEachLayer[numOfLayers - 1];
    w -= numOfNeuronsEachLayer[numOfLayers - 2] * numOfNeuronsEachLayer[numOfLayers - 1];
    subtract(
        numOfNeuronsEachLayer[numOfLayers - 1], activations_ + a,
        trainingLabels + dataPointIndex * numOfClasses, cost_ + b
    );
    sigmoidPrime(numOfNeuronsEachLayer[numOfLayers - 1], zs_ + b, zPrimes_ + b);
    multiply(numOfNeuronsEachLayer[numOfLayers - 1], cost_ + b, zPrimes_ + b, deltaNablaBiases_ + b);
    a -= numOfNeuronsEachLayer[numOfLayers - 2];
    dotVectorsWithMatrixOut(
        numOfNeuronsEachLayer[numOfLayers - 1], numOfNeuronsEachLayer[numOfLayers - 2],
        deltaNablaBiases_ + b, activations_ + a, deltaNablaWeights_ + w
    );

    // backward pass for the rest of the layers
    for (std::size_t l = numOfLayers - 2; l > 0; --l) {
        a -= numOfNeuronsEachLayer[l - 1];
        b -= numOfNeuronsEachLayer[l];

        dotVectorMatrix(
            numOfNeuronsEachLayer[l + 1], numOfNeuronsEachLayer[l],
            cost_ + b + numOfNeuronsEachLayer[l], weights + w, cost_ + b
        );
        sigmoidPrime(numOfNeuronsEachLayer[l], zs_ + b, zPrimes_ + b);
        multiply(numOfNeuronsEachLayer[l], cost_ + b, zPrimes_ + b, deltaNablaBiases_ + b);

        w -= numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];

        dotVectorsWithMatrixOut(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            deltaNablaBiases_ + b, activations_ + a, deltaNablaWeights_ + w
        );
    }
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t NetworkOpenMP::evaluate() const {
    std::size_t prediction;
    std::size_t numOfPassedTests = 0;
    std::size_t offset = activationsSize - numOfNeuronsEachLayer[numOfLayers - 1];
    double* zs_, * activations_, * weightsDotActivations_;

    #pragma omp parallel for num_threads(numOfThreads) reduction(+:numOfPassedTests) \
        shared(pixelsPerImage, testSize, testImages, testLabels, numOfNeuronsEachLayer, \
        numOfLayers, weights, biases, biasesSize, activationsSize, offset) \
        private(zs_, activations_, weightsDotActivations_, prediction) schedule(dynamic)
    for (std::size_t m = 0; m < testSize; ++m) {
        zs_ = new double[biasesSize];
        activations_ = new double[activationsSize];
        weightsDotActivations_ = new double[biasesSize];
        std::copy(testImages + m * pixelsPerImage, testImages + (m + 1) * pixelsPerImage, activations_);

        // feed forward
        for (std::size_t l = 1, i = 0, j = 0, k = 0; l < numOfLayers; ++l) {
            dotMatrixVector(
                numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
                weights + j, activations_ + k, weightsDotActivations_ + i
            );
            add(numOfNeuronsEachLayer[l], weightsDotActivations_ + i, biases + i, zs_ + i);

            k += numOfNeuronsEachLayer[l - 1];
            sigmoid(numOfNeuronsEachLayer[l], zs_ + i, activations_ + k);

            i += numOfNeuronsEachLayer[l];
            j += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
        }

        prediction = offset;
        for (std::size_t i = offset + 1; i < activationsSize; ++i)
            if (activations_[i] > activations_[prediction])
                prediction = i;

        if (testLabels[(prediction - offset) + (numOfClasses * m)] == 1)
            ++numOfPassedTests;

        delete[] zs_;
        delete[] activations_;
        delete[] weightsDotActivations_;
    }
    return numOfPassedTests;
}