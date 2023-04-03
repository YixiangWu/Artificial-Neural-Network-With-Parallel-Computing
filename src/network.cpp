#include "network.hpp"
#include "operation.cpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <string>


Network::Network(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile, std::size_t numOfHiddenLayers,
    const std::size_t* numOfNeuronsEachHiddenLayers, std::size_t numOfEpochs,
    std::size_t miniBatchSize, double learningRate
) : numOfThreads(0), numOfEpochs(numOfEpochs), miniBatchSize(miniBatchSize), learningRate(learningRate) {
    loadTrainingAndTest(
        pixelsPerImage, numOfClasses, trainingSize, testSize,
        trainingImageFile, trainingLabelFile, testImageFile, testLabelFile
    );

    initializeNetwork(numOfHiddenLayers, numOfNeuronsEachHiddenLayers);

    mt = std::mt19937(std::random_device()());
}

Network::~Network() {
    delete[] trainingDataIndices;
    delete[] trainingImages;
    delete[] trainingLabels;
    delete[] testImages;
    delete[] testLabels;
    delete[] numOfNeuronsEachLayer;
}

void Network::loadData(
    std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
    std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
    const std::string& testImageFile, const std::string& testLabelFile
) {
    delete[] trainingDataIndices;
    delete[] trainingImages;
    delete[] trainingLabels;
    delete[] testImages;
    delete[] testLabels;

    loadTrainingAndTest(
        pixelsPerImage, numOfClasses, trainingSize, testSize,
        trainingImageFile, trainingLabelFile, testImageFile, testLabelFile
    );

    activationsSize = pixelsPerImage + biasesSize;
    loadDataHelper();
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
    trainingDataIndices = new std::size_t[trainingSize];
    for (std::size_t i = 0; i < trainingSize; ++i)
        trainingDataIndices[i] = i;
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
    memoryFree();
    initializeNetwork(numOfHiddenLayers, numOfNeuronsEachHiddenLayers);
    memoryAllocate();
}

/** Initializes the neural network based on hidden layers. */
void Network::initializeNetwork(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers) {
    numOfLayers = numOfHiddenLayers + 2;
    numOfNeuronsEachLayer = new std::size_t[numOfLayers];
    numOfNeuronsEachLayer[0] = pixelsPerImage;
    numOfNeuronsEachLayer[numOfLayers - 1] = numOfClasses;
    for (std::size_t l = 0; l < numOfHiddenLayers; ++l)
        numOfNeuronsEachLayer[l + 1] = numOfNeuronsEachHiddenLayers[l];

    biasesSize = 0; weightsSize = 0;
    for (std::size_t l = 1; l < numOfLayers; ++l) {
    // l = 1 -> start from the second layer (excluding the input layer)

        // one random bias for each neuron
        biasesSize += numOfNeuronsEachLayer[l];

        // the amount of weights for each neuron in the l-th layer is the number of neurons in the (l - 1)-th layer
        weightsSize += numOfNeuronsEachLayer[l - 1] * numOfNeuronsEachLayer[l];
    }

    activationsSize = pixelsPerImage + biasesSize;
}

/** Prints info of the network. */
void Network::printInfo() {
    std::cout << "Network " << platform;
    if (numOfThreads > 0) std::cout << " with " << numOfThreads << " Treads";
    std::cout << std::endl << "Network Size: {";
    for (std::size_t i = 0; i < numOfLayers; ++i) {
        std::cout << numOfNeuronsEachLayer[i];
        if (i != numOfLayers - 1) std::cout << ", ";
    }
    std::cout << "}" << std::endl;
    std::cout << "Number of Epochs: " << numOfEpochs << std::endl;
    std::cout << "Mini-Batch Size: " << miniBatchSize << std::endl;
    std::cout << "Learning Rate: " << learningRate << std::endl << std::endl;
}

/** Generates random biases and weights from normal distribution. */
void Network::generateBiasesAndWeights() {
    std::normal_distribution<double> dist(0, 1);

    biases = new double[biasesSize];
    for (std::size_t i = 0; i < biasesSize; ++i)
        biases[i] = dist(mt);

    weights = new double[weightsSize];
    for (std::size_t i = 0; i < weightsSize; ++i)
        weights[i] = dist(mt);
}

/** Shuffles the training data. */
void Network::shuffleTrainingData() {
    for (std::size_t i = 0; i < trainingSize; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, trainingSize - 1);
        std::swap(trainingDataIndices[i], trainingDataIndices[dist(mt)]);
    }
}