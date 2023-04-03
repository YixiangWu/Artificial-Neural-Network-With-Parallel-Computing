#include "network_base.hpp"
#include "operation.cpp"

#include <algorithm>
#include <cstddef>
#include <iostream>


/** Allocates memory. */
void NetworkBase::memoryAllocate() {
    printInfo();
    generateBiasesAndWeights();

    nablaBiases = new double[biasesSize];
    nablaWeights = new double[weightsSize];
    deltaNablaBiases = new double[biasesSize];
    deltaNablaWeights = new double[weightsSize];

    zs = new double[biasesSize];
    cost = new double[biasesSize];
    zPrimes = new double[biasesSize];
    weightsDotActivations = new double[biasesSize];

    activations = new double[activationsSize];
}

/** Frees memory. */
void NetworkBase::memoryFree() {
    delete[] biases;
    delete[] weights;
    delete[] nablaBiases;
    delete[] nablaWeights;
    delete[] deltaNablaBiases;
    delete[] deltaNablaWeights;

    delete[] zs;
    delete[] cost;
    delete[] zPrimes;
    delete[] weightsDotActivations;

    delete[] activations;
}

/** Helps allocate new memory at the end of loading data. */
void NetworkBase::loadDataHelper() {
    printInfo();
    delete[] activations;
    activations = new double[activationsSize];
}

/** Changes biases and weights repeatedly to achieve a minimum cost. */
void NetworkBase::stochasticGradientDescent() {
    for (std::size_t e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {

            // initialize batches
            for (std::size_t j = 0; j < biasesSize; ++j)
                nablaBiases[j] = 0;  // initialize all elements to 0

            for (std::size_t j = 0; j < weightsSize; ++j)
                nablaWeights[j] = 0;  // initialize all elements to 0

            // look for the proper partial derivatives that reduce the cost
            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                // std::cout << e << " " << i << " " << j << std::endl;

                backpropagation(trainingDataIndices[i * miniBatchSize + j]);

                // update partial derivatives of the cost function with respect to biases
                for (std::size_t k = 0; k < biasesSize; ++k)
                    nablaBiases[k] += deltaNablaBiases[k];

                // update partial derivatives of the cost function with respect to weights
                for (std::size_t k = 0; k < weightsSize; ++k)
                    nablaWeights[k] += deltaNablaWeights[k];
            }
            // reduce the cost by changing biases
            for (std::size_t j = 0; j < biasesSize; ++j)
                biases[j] -= (learningRate / (double) miniBatchSize) * nablaBiases[j];

            // reduce the cost by changing weights
            for (std::size_t j = 0; j < weightsSize; ++j)
                weights[j] -= (learningRate / (double) miniBatchSize) * nablaWeights[j];
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
void NetworkBase::backpropagation(std::size_t dataPointIndex) {
    std::size_t a = 0, b = 0, w = 0; // a -> activationsSize, b -> biasesSize, w -> weightsSize
    std::copy(
        trainingImages + dataPointIndex * pixelsPerImage,
        trainingImages + (dataPointIndex + 1) * pixelsPerImage, activations
    );

    // feed forward
    for (std::size_t l = 1; l < numOfLayers; ++l) {
        // z = weights dot activations + bias
        dotMatrixVector(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            weights + w, activations + a, weightsDotActivations + b
        );
        add(numOfNeuronsEachLayer[l], weightsDotActivations + b, biases + b, zs + b);

        // new activation = sigmoid(z) = 1 / (1 + (e ^ -z))
        a += numOfNeuronsEachLayer[l - 1];
        sigmoid(numOfNeuronsEachLayer[l], zs + b, activations + a);

        b += numOfNeuronsEachLayer[l];
        w += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    // backward pass for the output layer
    b -= numOfNeuronsEachLayer[numOfLayers - 1];
    w -= numOfNeuronsEachLayer[numOfLayers - 2] * numOfNeuronsEachLayer[numOfLayers - 1];
    // cost for the output layer = (predicted<y> - observed<y>)
    subtract(
        numOfNeuronsEachLayer[numOfLayers - 1], activations + a,
        trainingLabels + dataPointIndex * numOfClasses, cost + b
    );

    // error[l] = cost[l] HadamardProduct sigmoid'(z[l])
    // deltaNablaBiases[l] = error[l]
    sigmoidPrime(numOfNeuronsEachLayer[numOfLayers - 1], zs + b, zPrimes + b);
    multiply(numOfNeuronsEachLayer[numOfLayers - 1], cost + b, zPrimes + b, deltaNablaBiases + b);
    a -= numOfNeuronsEachLayer[numOfLayers - 2];

    // deltaNablaWeights[l] = error[l] dot activations[l - 1]
    dotVectorsWithMatrixOut(
        numOfNeuronsEachLayer[numOfLayers - 1], numOfNeuronsEachLayer[numOfLayers - 2],
        deltaNablaBiases + b, activations + a, deltaNablaWeights + w
    );

    // backward pass for the rest of the layers
    for (std::size_t l = numOfLayers - 2; l > 0; --l) {
    // l = numOfLayers - 2 -> start from the back right before the output layer

        a -= numOfNeuronsEachLayer[l - 1];
        b -= numOfNeuronsEachLayer[l];

        // cost[l] = transpose(weights[l]) dot cost[l + 1]
        dotVectorMatrix(
            numOfNeuronsEachLayer[l + 1], numOfNeuronsEachLayer[l],
            cost + b + numOfNeuronsEachLayer[l], weights + w, cost + b
        );

        // error[l] = cost[l] HadamardProduct sigmoid'(z[l])
        // deltaNablaBiases[l] = error[l]
        sigmoidPrime(numOfNeuronsEachLayer[l], zs + b, zPrimes + b);
        multiply(numOfNeuronsEachLayer[l], cost + b, zPrimes + b, deltaNablaBiases + b);

        // deltaNablaWeights[l] = error[l] dot activations[l - 1]
        w -= numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
        dotVectorsWithMatrixOut(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            deltaNablaBiases + b, activations + a, deltaNablaWeights + w
        );
    }
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t NetworkBase::evaluate() const {
    std::size_t prediction;
    std::size_t numOfPassedTests = 0;
    std::size_t offset = activationsSize - numOfNeuronsEachLayer[numOfLayers - 1];
    auto zs_ = new double[biasesSize];
    auto activations_ = new double[activationsSize];
    auto weightsDotActivations_ = new double[biasesSize];

    for (std::size_t m = 0; m < testSize; ++m) {
        std::copy(testImages + m * pixelsPerImage, testImages + (m + 1) * pixelsPerImage, activations_);

        // feed forward
        for (std::size_t l = 1, i = 0, j = 0, k = 0; l < numOfLayers; ++l) {
            // z = weights dot activations + bias
            dotMatrixVector(
                numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
                weights + j, activations_ + k, weightsDotActivations_ + i
            );
            add(numOfNeuronsEachLayer[l], weightsDotActivations_ + i, biases + i, zs_ + i);

            // new activation = sigmoid(z) = 1 / (1 + (e ^ -z))
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
    }
    delete[] zs_;
    delete[] activations_;
    delete[] weightsDotActivations_;
    return numOfPassedTests;
}