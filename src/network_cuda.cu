#include "network_cuda.cuh"
#include "operation.cu"

#include <cstddef>
#include <iostream>


NetworkCUDA::NetworkCUDA(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile, std::size_t numOfHiddenLayers,
    const std::size_t* numOfNeuronsEachHiddenLayers, unsigned int numOfEpochs,
    std::size_t miniBatchSize, double learningRate
) : Network::Network(
    pixelsPerImage, numOfClasses, trainingSize, testSize, trainingImageFile, trainingLabelFile, testImageFile,
    testLabelFile, numOfHiddenLayers, numOfNeuronsEachHiddenLayers, numOfEpochs, miniBatchSize, learningRate
) {
    cudaMalloc(&biases_, biasesSize * sizeof(double));
    cudaMalloc(&weights_, weightsSize * sizeof(double));
    cudaMalloc(&nablaBiases_, biasesSize * sizeof(double));
    cudaMalloc(&nablaWeights_, weightsSize * sizeof(double));
    cudaMalloc(&deltaNablaBiases_, biasesSize * sizeof(double));
    cudaMalloc(&deltaNablaWeights_, weightsSize * sizeof(double));
    cudaMalloc(&activations_, (pixelsPerImage + biasesSize) * sizeof(double));
    cudaMalloc(&trainingLabel_, numOfClasses * sizeof(double));

    cudaMemcpy(biases_, biases, biasesSize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weights_, weights, weightsSize * sizeof(double), cudaMemcpyHostToDevice);
}

NetworkCUDA::~NetworkCUDA() {
    cudaFree(biases_);
    cudaFree(weights_);
    cudaFree(nablaBiases_);
    cudaFree(nablaWeights_);
    cudaFree(deltaNablaBiases_);
    cudaFree(deltaNablaWeights_);
    cudaFree(activations_);
    cudaFree(trainingLabel_);
}

/** Changes biases and weights repeatedly to achieve a minimum cost. */
void NetworkCUDA::stochasticGradientDescent() {
    for (unsigned int e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {

            // initialize batches
            zero(biasesSize, nablaBiases_);
            zero(weightsSize, nablaWeights_);

            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                backpropagation(trainingDataIndices[i * miniBatchSize + j]);

                // update partial derivatives of the cost function with respect to biases
                add(biasesSize, nablaBiases_, deltaNablaBiases_, nablaBiases_);

                // update partial derivatives of the cost function with respect to weights
                add(weightsSize, nablaWeights_, deltaNablaWeights_, nablaWeights_);
            }
            // reduce the cost by changing biases
            reduceCost(biasesSize, learningRate, miniBatchSize, nablaBiases_, biases_);

            // reduce the cost by changing weights
            reduceCost(weightsSize, learningRate, miniBatchSize, nablaWeights_, weights_);
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
void NetworkCUDA::backpropagation(std::size_t dataPointIndex) {
    double* zs, * weightsDotActivations, * cost, * zPrimes;
    cudaMalloc(&zs, biasesSize * sizeof(double));
    cudaMalloc(&weightsDotActivations, biasesSize * sizeof(double));
    cudaMalloc(&cost, biasesSize * sizeof(double));
    cudaMalloc(&zPrimes, biasesSize * sizeof(double));
    cudaMemcpy(
        activations_, trainingImages + dataPointIndex * pixelsPerImage,
        pixelsPerImage * sizeof(double), cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        trainingLabel_, trainingLabels + dataPointIndex * numOfClasses,
        numOfClasses * sizeof(double), cudaMemcpyHostToDevice
    );
    std::size_t a = 0, b = 0, w = 0;

    // feed forward
    for (std::size_t l = 1; l < numOfLayers; ++l) {
        dotMatrixVector(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            weights_ + w, activations_ + a, weightsDotActivations + b
        );
        add(numOfNeuronsEachLayer[l], weightsDotActivations + b, biases_ + b, zs + b);

        a += numOfNeuronsEachLayer[l - 1];
        sigmoid(numOfNeuronsEachLayer[l], zs + b, activations_ + a);

        b += numOfNeuronsEachLayer[l];
        w += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    // backward pass for the output layer
    b -= numOfNeuronsEachLayer[numOfLayers - 1];
    w -= numOfNeuronsEachLayer[numOfLayers - 2] * numOfNeuronsEachLayer[numOfLayers - 1];
    subtract(numOfNeuronsEachLayer[numOfLayers - 1], activations_ + a, trainingLabel_, cost + b);
    sigmoidPrime(numOfNeuronsEachLayer[numOfLayers - 1], zs + b, zPrimes + b);
    multiply(numOfNeuronsEachLayer[numOfLayers - 1], cost + b, zPrimes + b, deltaNablaBiases_ + b);
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
            cost + b + numOfNeuronsEachLayer[l], weights_ + w, cost + b
        );
        sigmoidPrime(numOfNeuronsEachLayer[l], zs + b, zPrimes + b);
        multiply(numOfNeuronsEachLayer[l], cost + b, zPrimes + b, deltaNablaBiases_ + b);

        w -= numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];

        dotVectorsWithMatrixOut(
            numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
            deltaNablaBiases_ + b, activations_ + a, deltaNablaWeights_ + w
        );
    }

    cudaFree(zs);
    cudaFree(weightsDotActivations);
    cudaFree(cost);
    cudaFree(zPrimes);
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t NetworkCUDA::evaluate() const {
    std::size_t numOfPassedTests = 0;
    double* zs;
    double* weightsDotActivations;
    cudaMalloc(&zs, biasesSize * sizeof(double));
    cudaMalloc(&weightsDotActivations, biasesSize * sizeof(double));
    auto predictions = new double[numOfClasses];

    for (std::size_t m = 0; m < testSize; ++m) {
        cudaMemcpy(
            activations_, testImages + m * pixelsPerImage,
            pixelsPerImage * sizeof(double), cudaMemcpyHostToDevice
        );

        // feed forward
        for (std::size_t l = 1, i = 0, j = 0, k = 0; l < numOfLayers; ++l) {
            dotMatrixVector(
                numOfNeuronsEachLayer[l], numOfNeuronsEachLayer[l - 1],
                weights_ + j, activations_ + k, weightsDotActivations + i
            );
            add(numOfNeuronsEachLayer[l], weightsDotActivations + i, biases_ + i, zs + i);

            k += numOfNeuronsEachLayer[l - 1];
            sigmoid(numOfNeuronsEachLayer[l], zs + i, activations_ + k);

            i += numOfNeuronsEachLayer[l];
            j += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
        }

        cudaMemcpy(
            predictions, activations_ + pixelsPerImage + biasesSize - numOfClasses,
            numOfClasses * sizeof(double), cudaMemcpyDeviceToHost
        );

        std::size_t prediction = 0;
        for (std::size_t i = 1; i < numOfClasses; ++i) {
            if (predictions[i] > predictions[prediction])
                prediction = i;
        }

        if (testLabels[prediction + (numOfClasses * m)] == 1)
            ++numOfPassedTests;
    }

    cudaFree(zs);
    cudaFree(weightsDotActivations);
    delete[] predictions;
    return numOfPassedTests;
}