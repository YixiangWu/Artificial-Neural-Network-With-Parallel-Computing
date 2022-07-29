#include <cmath>
#include <cstddef>


/** Vector(Matrix) Addition */
template<typename T>
void add(std::size_t size, const T* vector1, const T* vector2, T* vectorOut) {
    for (std::size_t i = 0; i < size; ++i)
        vectorOut[i] = vector1[i] + vector2[i];
}

/** Vector(Matrix) Subtraction */
template<typename T>
void subtract(std::size_t size, const T* vector1, const T* vector2, T* vectorOut) {
    for (std::size_t i = 0; i < size; ++i)
        vectorOut[i] = vector1[i] - vector2[i];
}

/** Vector(Matrix) Multiplication (Hadamard Product) */
template<typename T>
void multiply(std::size_t size, const T* vector1, const T* vector2, T* vectorOut) {
    for (std::size_t i = 0; i < size; ++i)
        vectorOut[i] = vector1[i] * vector2[i];
}

/** Dot Product */
template<typename T>
void dotVectorsWithMatrixOut(std::size_t vector1Size, std::size_t vector2Size, const T* vector1, const T* vector2, T* matrixOut) {
    for (std::size_t i = 0; i < vector1Size; ++i)
        for (std::size_t j = 0; j < vector2Size; ++j)
            matrixOut[i * vector2Size + j] = vector1[i] * vector2[j];
}

/** Dot Product */
template<typename T>
void dotMatrixVector(std::size_t numOfMatrixRows, std::size_t numOfMatrixCols, const T* matrix, const T* vector, T* vectorOut) {
    for (std::size_t i = 0; i < numOfMatrixRows; ++i) {
        vectorOut[i] = 0;  // initialize all elements to 0
        for (std::size_t j = 0; j < numOfMatrixCols; ++j)
            vectorOut[i] += matrix[i * numOfMatrixCols + j] * vector[j];
    }
}

/** Dot Product */
template<typename T>
void dotVectorMatrix(std::size_t numOfMatrixRows, std::size_t numOfMatrixCols, const T* vector, const T* matrix, T* vectorOut) {
    for (std::size_t i = 0; i < numOfMatrixCols; ++i) {
        vectorOut[i] = 0;  // initialize all elements to 0
        for (std::size_t j = 0; j < numOfMatrixRows; ++j)
            vectorOut[i] += matrix[j * numOfMatrixCols + i] * vector[j];
    }
}

/** Sigmoid Function: sigmoid(z) = 1 / (1 + (e ^ -z)) */
template<typename T>
void sigmoid(std::size_t size, const T* z, T* functionOut) {
    for (std::size_t i = 0; i < size; ++i)
        functionOut[i] = 1 / (1 + std::exp(-z[i]));
}

/** Derivative of Sigmoid Function: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) */
template<typename T>
void sigmoidPrime(std::size_t size, const T* z, T* functionOut) {
    auto sigmoidZ = new double[size];
    sigmoid(size, z, sigmoidZ);
    for (std::size_t i = 0; i < size; ++i)
        functionOut[i] = sigmoidZ[i] * (1 - sigmoidZ[i]);
    delete[] sigmoidZ;
}