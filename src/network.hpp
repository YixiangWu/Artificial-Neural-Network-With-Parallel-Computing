#ifndef NETWORK
#define NETWORK

#include <cstddef>
#include <random>
#include <string>


class Network {
public:
    Network(
        std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
        std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
        const std::string& testImageFile, const std::string& testLabelFile, std::size_t numOfHiddenLayers,
        const std::size_t* numOfNeuronsEachHiddenLayers, unsigned int numOfEpochs,
        std::size_t miniBatchSize, double learningRate
    );

    ~Network();

    void loadData(
        std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
        std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
        const std::string& testImageFile, const std::string& testLabelFile
    );

    void setHiddenLayers(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers);

    void setNumOfEpochs(unsigned int numOfEpochs) { this->numOfEpochs = numOfEpochs; }

    void setMiniBatchSize(std::size_t miniBatchSize) { this->miniBatchSize = miniBatchSize; }

    void setLearningRate(double learningRate) { this->learningRate = learningRate; }

    void train() { stochasticGradientDescent(); }


protected:
    std::size_t pixelsPerImage;
    std::size_t numOfClasses;

    std::size_t trainingSize;
    std::size_t testSize;

    double* trainingImages;
    double* trainingLabels;
    double* testImages;
    double* testLabels;

    std::size_t numOfLayers;
    std::size_t* numOfNeuronsEachLayer;

    std::size_t biasesSize;
    std::size_t weightsSize;

    std::mt19937 mt;

    // biases[l][j] is bias of the (j + 1)-th neuron in the (l + 2)-th layer
    // assuming 1-base indexing for neurons and layers
    double* biases;

    // weights[l][j][k] is the weight connecting from the (k + 1)-th neuron in the (l + 1)-th layer
    // to the (j + 1)-th neuron in the (l + 2)-th layer
    // assuming 1-base indexing for neurons and layers
    double* weights;

    unsigned int numOfEpochs;
    std::size_t miniBatchSize;
    double learningRate;

    /** Loads both training and test data. */
    void loadTrainingAndTest(
        std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
        std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
        const std::string& testImageFile, const std::string& testLabelFile
    );

    /** Loads images and labels from text files. */
    void loadImagesAndLabels(const std::string& imageFile, const std::string& labelFile, bool isTestData=false);

    /** Initializes the neural network based on hidden layers. */
    void initializeNetwork(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers);

    /** Changes biases and weights repeatedly to achieve a minimum cost. */
    virtual void stochasticGradientDescent();

    /** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
    void backpropagation(std::size_t dataPointIndex, double* deltaNablaBiases, double* deltaNablaWeights);

    /** Shuffles the training data. */
    void shuffleTrainingData();

    /** Evaluates the network (biases and weights) with the test data. */
    virtual std::size_t evaluate() const;
};


#endif