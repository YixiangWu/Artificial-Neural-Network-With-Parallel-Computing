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
        const std::size_t* numOfNeuronsEachHiddenLayers, std::size_t numOfEpochs,
        std::size_t miniBatchSize, double learningRate
    );

    ~Network();

    void loadData(
        std::size_t pixelsPerImage, std::size_t numOfClasses, std::size_t trainingSize,
        std::size_t testSize, const std::string& trainingImageFile, const std::string& trainingLabelFile,
        const std::string& testImageFile, const std::string& testLabelFile
    );

    void setHiddenLayers(std::size_t numOfHiddenLayers, const std::size_t* numOfNeuronsEachHiddenLayers);

    void setNumOfEpochs(std::size_t numOfEpochs) { this->numOfEpochs = numOfEpochs; printInfo(); }

    void setMiniBatchSize(std::size_t miniBatchSize) { this->miniBatchSize = miniBatchSize; printInfo(); }

    void setLearningRate(double learningRate) { this->learningRate = learningRate; printInfo(); }

    void train() { stochasticGradientDescent(); }


protected:
    std::string platform;
    std::size_t numOfThreads;

    std::size_t pixelsPerImage;
    std::size_t numOfClasses;

    std::size_t trainingSize;
    std::size_t testSize;

    std::size_t* trainingDataIndices;
    double* trainingImages;
    double* trainingLabels;
    double* testImages;
    double* testLabels;

    std::size_t numOfLayers;
    std::size_t* numOfNeuronsEachLayer;

    std::size_t biasesSize;
    std::size_t weightsSize;
    std::size_t activationsSize;  // activationsSize = pixelsPerImage + biasesSize

    std::mt19937 mt;

    // biases[l][j] is bias of the (j + 1)-th neuron in the (l + 2)-th layer
    // assuming 1-base indexing for neurons and layers
    double* biases;

    // weights[l][j][k] is the weight connecting from the (k + 1)-th neuron in the (l + 1)-th layer
    // to the (j + 1)-th neuron in the (l + 2)-th layer
    // assuming 1-base indexing for neurons and layers
    double* weights;

    double* nablaBiases;  // partial derivatives of the cost function with respect to biases
    double* nablaWeights;  // partial derivatives of the cost function with respect to weights
    double* deltaNablaBiases;
    double* deltaNablaWeights;

    // helper arrays
    double* zs;  // store z vectors, layer by layer
    double* cost;
    double* zPrimes;  // store z' vectors, layer by layer
    double* activations;  // store activations, layer by layer
    double* weightsDotActivations;

    std::size_t numOfEpochs;
    std::size_t miniBatchSize;
    double learningRate;

    /** Prints hyperparameters info of the network. */
    void printInfo();

    /** Generates random biases and weights from normal distribution. */
    void generateBiasesAndWeights();

    /** Shuffles the training data. */
    void shuffleTrainingData();

    /** Allocates memory. */
    virtual void memoryAllocate() = 0;

    /** Frees memory. */
    virtual void memoryFree() = 0;

    /** Helps allocate new memory at the end of loading data. */
    virtual void loadDataHelper() = 0;

    /** Changes biases and weights repeatedly to achieve a minimum cost. */
    virtual void stochasticGradientDescent() = 0;

    /** Helps compute partial derivatives of the cost function with respect to any weight or bias in the network. */
    virtual void backpropagation(std::size_t dataPointIndex) = 0;

    /** Evaluates the network (biases and weights) with the test data. */
    virtual std::size_t evaluate() const = 0;


private:
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
};


#endif