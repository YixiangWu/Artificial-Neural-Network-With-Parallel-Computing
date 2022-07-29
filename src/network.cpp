#include "network.hpp"
#include "operation.cpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <string>


Network::Network(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile, std::size_t numOfHiddenLayers,
    const std::size_t* numOfNeuronsEachHiddenLayers, unsigned int numOfEpochs,
    std::size_t miniBatchSize, double learningRate
) : numOfEpochs(numOfEpochs), miniBatchSize(miniBatchSize), learningRate(learningRate) {
    loadTrainingAndTest(
        pixelsPerImage, numOfClasses, trainingSize, testSize,
        trainingImageFile, trainingLabelFile, testImageFile, testLabelFile
    );

    initializeNetwork(numOfHiddenLayers, numOfNeuronsEachHiddenLayers);

    mt = std::mt19937(std::random_device()());
}

Network::~Network() {
    delete[] trainingImages;
    delete[] trainingLabels;
    delete[] testImages;
    delete[] testLabels;
    delete[] numOfNeuronsEachLayer;
    delete[] biases;
    delete[] weights;
}

void Network::loadData(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile
) {
    delete[] trainingImages;
    delete[] trainingLabels;
    delete[] testImages;
    delete[] testLabels;

    loadTrainingAndTest(
        pixelsPerImage, numOfClasses, trainingSize, testSize,
        trainingImageFile, trainingLabelFile, testImageFile, testLabelFile
    );
}

/** Loads both training and test data. */
void Network::loadTrainingAndTest(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile
) {
    this->pixelsPerImage = pixelsPerImage;
    this->numOfClasses = numOfClasses;

    // load training data
    this->trainingSize = trainingSize;
    trainingImages = new double[trainingSize * pixelsPerImage];
    trainingLabels = new double[trainingSize * numOfClasses];
    loadImagesAndLabels(trainingImageFile, trainingLabelFile);

    // load test data
    this->testSize = testSize;
    testImages = new double[testSize * pixelsPerImage];
    testLabels = new double[testSize * numOfClasses];
    loadImagesAndLabels(testImageFile, testLabelFile, true);
}

/** Loads images and labels from text files. */
void Network::loadImagesAndLabels(const std::string& imageFile, const std::string& labelFile, bool isTestData) {
    std::size_t size;
    double* images;
    double* labels;

    if (!isTestData) {
        size = trainingSize;
        images = trainingImages;
        labels = trainingLabels;
    }
    else {
        size = testSize;
        images = testImages;
        labels = testLabels;
    }

    std::fstream imagesIn(imageFile);
    std::fstream labelsIn(labelFile);
    std::string label, pixels;

    for (std::size_t i = 0; i < size; ++i) {
        // load images
        std::getline(imagesIn, pixels);  // an image in each line
        for (std::size_t j = 0, k = 0; pixels[j] != '\0'; ++j)
            if (pixels[j] == ' ')  // pixels are separated with a space
                // normalization before loading the next pixel (k++)
                images[i * pixelsPerImage + k++] /= 255;
            else
                images[i * pixelsPerImage + k] = 10 * images[i * pixelsPerImage + k] + (pixels[j] - 48);

        // load labels
        std::getline(labelsIn, label);  // a label in each line
        for (std::size_t j = 0; j < numOfClasses; ++j)
            labels[i * numOfClasses + j] = ((std::size_t) std::stoi(label) == j) ? 1 : 0;
    }

    imagesIn.close();
    labelsIn.close();
}

void Network::setHiddenLayers(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers) {
    delete[] numOfNeuronsEachLayer;
    delete[] biases;
    delete[] weights;

    initializeNetwork(numOfHiddenLayers, numOfNeuronsEachHiddenLayers);
}

/** Initializes the neural network based on hidden layers. */
void Network::initializeNetwork(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers) {
    numOfLayers = numOfHiddenLayers + 2;
    numOfNeuronsEachLayer = new size_t[numOfLayers];
    numOfNeuronsEachLayer[0] = pixelsPerImage;
    numOfNeuronsEachLayer[numOfLayers - 1] = numOfClasses;
    for (size_t l = 0; l < numOfHiddenLayers; ++l)
        numOfNeuronsEachLayer[l + 1] = numOfNeuronsEachHiddenLayers[l];

    biasesSize = 0;
    weightsSize = 0;
    for (std::size_t l = 1; l < numOfLayers; ++l) {
    // l = 1 -> start from the second layer (excluding the input layer)

        // one random bias for each neuron
        biasesSize += numOfNeuronsEachLayer[l];

        // the amount of weights for each neuron in the l-th layer is the number of neurons in the (l - 1)-th layer
        weightsSize += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    std::normal_distribution<double> dist(0, 1);

    biases = new double[biasesSize];
    for (std::size_t i = 0; i < biasesSize; ++i)
        biases[i++] = dist(mt);

    weights = new double[weightsSize];
    for (std::size_t i = 0; i < weightsSize; ++i)
        weights[i++] = dist(mt);
}

/** Changes biases and weights repeatedly to achieve a minimum cost. */
void Network::stochasticGradientDescent() {
    for (unsigned int e = 0; e < numOfEpochs; ++e) {
        std::cout << "Epoch " << e << " ..." << std::endl;

        // partial derivatives of the cost function with respect to biases
        double* nablaBiases;

        // partial derivatives of the cost function with respect to weights
        double* nablaWeights;

        // split training data into mini batches
        shuffleTrainingData();
        for (std::size_t i = 0; i < trainingSize / miniBatchSize; ++i) {
            nablaBiases = new double[biasesSize];
            for (std::size_t j = 0; j < biasesSize; ++j)
                nablaBiases[j] = 0;  // initialize all elements to 0

            nablaWeights = new double[weightsSize];
            for (std::size_t j = 0; j < weightsSize; ++j)
                nablaWeights[j] = 0;  // initialize all elements to 0

            // look for the proper partial derivatives that reduce the cost
            double* deltaNablaBiases;
            double* deltaNablaWeights;
            for (std::size_t j = 0; j < miniBatchSize; ++j) {
                // std::cout << e << " " << i << " " << j << std::endl;

                deltaNablaBiases = new double[biasesSize];
                deltaNablaWeights = new double[weightsSize];

                backpropagation(i * miniBatchSize + j, deltaNablaBiases, deltaNablaWeights);

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
            for (std::size_t j = 0; j < biasesSize; ++j)
                biases[j] -= (learningRate / (double) miniBatchSize) * nablaBiases[j];

            // reduce the cost by changing weights
            for (std::size_t j = 0; j < weightsSize; ++j)
                weights[j] -= (learningRate / (double) miniBatchSize) * nablaWeights[j];

            delete[] nablaBiases;
            delete[] nablaWeights;
        }
        std::cout << "Epoch " << e << " " << evaluate() << "/" << testSize << std::endl;
    }
}

/** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
void Network::backpropagation(std::size_t dataPointIndex, double* deltaNablaBiases, double* deltaNablaWeights) {
    // store activations, layer by layer
    auto activations = new double[biasesSize + numOfNeuronsEachLayer[0]];
    std::copy(
        trainingImages + dataPointIndex * pixelsPerImage,
        trainingImages + (dataPointIndex + 1) * pixelsPerImage, activations
    );

    // store z vectors, layer by layer
    auto zs = new double[biasesSize];

    std::size_t a = 0, b = 0, w = 0;
    // a -> activationsSize (biasesSize + numOfNeuronsEachLayer[0]), b -> biasesSize, w -> weightsSize

    // feed forward
    auto weightsDotActivations = new double[biasesSize];  // helper array
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

    delete [] weightsDotActivations;

    // backward pass for the output layer
    b -= numOfNeuronsEachLayer[numOfLayers - 1];
    w -= numOfNeuronsEachLayer[numOfLayers - 2] * numOfNeuronsEachLayer[numOfLayers - 1];
    auto cost = new double[biasesSize];
    // cost for the output layer = (predicted<y> - observed<y>)
    subtract(
        numOfNeuronsEachLayer[numOfLayers - 1], activations + a,
        trainingLabels + dataPointIndex * numOfClasses, cost + b
    );

    // error[l] = cost[l] HadamardProduct sigmoid'(z[l])
    // deltaNablaBiases[l] = error[l]
    auto zPrimes = new double[biasesSize];  // store z' vectors, layer by layer
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

    delete[] activations;
    delete[] zs;
    delete[] cost;
    delete[] zPrimes;
}

/** Shuffles the training data. */
void Network::shuffleTrainingData() {
    std::size_t newIndex;

    for (std::size_t i = 0; i < trainingSize; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, trainingSize - 1);
        newIndex = dist(mt);

        // images swapping
        std::swap_ranges(
            trainingImages + i * pixelsPerImage, trainingImages + (i + 1) * pixelsPerImage,
            trainingImages + newIndex * pixelsPerImage
        );

        // labels swapping
        std::swap_ranges(
            trainingLabels + i * numOfClasses, trainingLabels + (i + 1) * numOfClasses,
            trainingLabels + newIndex * numOfClasses
        );
    }
}

/** Helps evaluate the network (biases and weights) with the test data. */
std::size_t Network::evaluate() const {
    std::size_t numOfPassedTests = 0;
    std::size_t prediction;
    std::size_t offset = biasesSize + numOfNeuronsEachLayer[0] - numOfNeuronsEachLayer[numOfLayers - 1];

    double* zs;
    double* activations;
    double* weightsDotActivations;  // helper array

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