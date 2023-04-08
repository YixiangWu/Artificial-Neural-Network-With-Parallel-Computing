#include "network_cuda.cuh"
#include "operation.cu"

#include <algorithm>
#include <cstddef>
#include <iostream>


/** Allocates memory. */
void NetworkCUDA::memoryAllocate() {
    printInfo();
    generateBiasesAndWeights();

    auto biasesTemp = biases;
    auto weightsTemp = weights;

    cudaMalloc(&biases, biasesSize * sizeof(double));
    cudaMalloc(&weights, weightsSize * sizeof(double));
    cudaMemcpy(biases, biasesTemp, biasesSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weights, weightsTemp, weightsSize * sizeof(double), cudaMemcpyHostToDevice);

    delete[] biasesTemp;
    delete[] weightsTemp;

    cudaMalloc(&nablaBiases, biasesSize * sizeof(double));
    cudaMalloc(&nablaWeights, weightsSize * sizeof(double));
    cudaMalloc(&deltaNablaBiases, biasesSize * sizeof(double));
    cudaMalloc(&deltaNablaWeights, weightsSize * sizeof(double));

    cudaMalloc(&zs, biasesSize * sizeof(double));
    cudaMalloc(&cost, biasesSize * sizeof(double));
    cudaMalloc(&zPrimes, biasesSize * sizeof(double));
    cudaMalloc(&weightsDotActivations, biasesSize * sizeof(double));

    std::size_t helperMatrixSize = numOfNeuronsEachLayer[0] * numOfNeuronsEachLayer[1];
    for (std::size_t i = 2; i < numOfLayers; ++i)
        std::max(helperMatrixSize, helperMatrixSize / numOfNeuronsEachLayer[i - 2] * numOfNeuronsEachLayer[i]);
    cudaMalloc(&helperMatrix, helperMatrixSize * sizeof(double));

    cudaMalloc(&activations, activationsSize * sizeof(double));
    cudaMalloc(&label, numOfClasses * sizeof(double));
}

/** Frees memory. */
void NetworkCUDA::memoryFree() {
    cudaFree(biases);
    cudaFree(weights);
    cudaFree(nablaBiases);
    cudaFree(nablaWeights);
    cudaFree(deltaNablaBiases);
    cudaFree(deltaNablaWeights);

    cudaFree(zs);
    cudaFree(cost);
    cudaFree(zPrimes);
    cudaFree(weightsDotActivations);
    cudaFree(helperMatrix);

    cudaFree(activations);
    cudaFree(label);
}

/** Changes biases and weights repeatedly to achieve a minimum cost. */
void NetworkCUDA::stochasticGradientDescent() {
    for (std::size_t e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {

            // initialize batches
            zero(biasesSize, nablaBiases);
            zero(weightsSize, nablaWeights);

            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                backpropagation(trainingDataIndices[i * miniBatchSize + j]);

                // update partial derivatives of the cost function with respect to biases
                add(biasesSize, nablaBiases, deltaNablaBiases, nablaBiases);

                // update partial derivatives of the cost function with respect to weights
                add(weightsSize, nablaWeights, deltaNablaWeights, nablaWeights);
            }
            // reduce the cost by changing biases
            reduceCost(biasesSize, learningRate, miniBatchSize, nablaBiases, biases);

            // reduce the cost by changing weights
            reduceCost(weightsSize, learningRate, miniBatchSize, nablaWeights, weights);
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
void NetworkCUDA::backpropagation(std::size_t dataPointIndex) {
    std::size_t a = 0, b = 0, w = 0;
    cudaMemcpy(
        activations, trainingImages + dataPointIndex * pixelsPerImage,
        pixelsPerImage * sizeof(double), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        label, trainingLabels + dataPointIndex * numOfClasses,
        numOfClasses * sizeof(double), cudaMemcpyHostToDevice
    );

    // feed forward
    for (std::size_t l = 1; l < numOfLayers; ++l) {
        dotMatrixVector(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1], weights + w,
            activations + a, helperMatrix, weightsDotActivations + b
        );
        add(numOfNeuronsEachLayer[l], weightsDotActivations + b, biases + b, zs + b);

        a += numOfNeuronsEachLayer[l - 1];
        sigmoid(numOfNeuronsEachLayer[l], zs + b, activations + a);

        b += numOfNeuronsEachLayer[l];
        w += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    // backward pass for the output layer
    b -= numOfNeuronsEachLayer[numOfLayers - 1];
    w -= numOfNeuronsEachLayer[numOfLayers - 2] * numOfNeuronsEachLayer[numOfLayers - 1];
    subtract(numOfNeuronsEachLayer[numOfLayers - 1], activations + a, label, cost + b);
    sigmoidPrime(numOfNeuronsEachLayer[numOfLayers - 1], zs + b, zPrimes + b);
    multiply(numOfNeuronsEachLayer[numOfLayers - 1], cost + b, zPrimes + b, deltaNablaBiases + b);
    a -= numOfNeuronsEachLayer[numOfLayers - 2];
    dotVectorsWithMatrixOut(
        numOfNeuronsEachLayer[numOfLayers - 1], numOfNeuronsEachLayer[numOfLayers - 2],
        deltaNablaBiases + b, activations + a, deltaNablaWeights + w
    );

    // backward pass for the rest of the layers
    for (std::size_t l = numOfLayers - 2; l > 0; --l) {
        a -= numOfNeuronsEachLayer[l - 1];
        b -= numOfNeuronsEachLayer[l];

        dotVectorMatrix(
            numOfNeuronsEachLayer[l + 1], numOfNeuronsEachLayer[l],
            cost + b + numOfNeuronsEachLayer[l], weights + w, cost + b
        );
        sigmoidPrime(numOfNeuronsEachLayer[l], zs + b, zPrimes + b);
        multiply(numOfNeuronsEachLayer[l], cost + b, zPrimes + b, deltaNablaBiases + b);

        w -= numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];

        dotVectorsWithMatrixOut(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            deltaNablaBiases + b, activations + a, deltaNablaWeights + w
        );
    }
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t NetworkCUDA::evaluate() const {
    std::size_t prediction;
    std::size_t numOfPassedTests = 0;
    auto predictions = new double[numOfClasses];
    double* zs_, * activations_, * weightsDotActivations_;
    cudaMalloc(&zs_, biasesSize * sizeof(double));
    cudaMalloc(&activations_, activationsSize * sizeof(double));
    cudaMalloc(&weightsDotActivations_, biasesSize * sizeof(double));

    for (std::size_t m = 0; m < testSize; ++m) {
        cudaMemcpy(
            activations_, testImages + m * pixelsPerImage,
            pixelsPerImage * sizeof(double), cudaMemcpyHostToDevice
        );

        // feed forward
        for (std::size_t l = 1, i = 0, j = 0, k = 0; l < numOfLayers; ++l) {
            dotMatrixVector(
                numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1], weights + j,
                activations_ + k, helperMatrix, weightsDotActivations_ + i
            );
            add(numOfNeuronsEachLayer[l], weightsDotActivations_ + i, biases + i, zs_ + i);

            k += numOfNeuronsEachLayer[l - 1];
            sigmoid(numOfNeuronsEachLayer[l], zs_ + i, activations_ + k);

            i += numOfNeuronsEachLayer[l];
            j += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
        }

        cudaMemcpy(
            predictions, activations_ + activationsSize - numOfClasses,
            numOfClasses * sizeof(double), cudaMemcpyDeviceToHost
        );

        prediction = 0;
        for (std::size_t i = 1; i < numOfClasses; ++i)
            if (predictions[i] > predictions[prediction])
                prediction = i;

        if (testLabels[prediction + (numOfClasses * m)] == 1)
            ++numOfPassedTests;
    }
    cudaFree(zs_);
    cudaFree(activations_);
    cudaFree(weightsDotActivations_);
    delete[] predictions;
    return numOfPassedTests;
}