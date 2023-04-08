#include <cmath>
#include <cstddef>


const std::size_t THREADS_PER_1D_BLOCK = 1024;
const std::size_t THREADS_PER_2D_BLOCK = 32;


__global__ void copy_(unsigned int size, const double* arrayIn, double* arrayOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    arrayOut[index] = arrayIn[index];
}


/** Initializes an array with zeros. */
__global__ void zero_(unsigned int size, double* array) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    array[index] = 0;
}

void zero(std::size_t size, double* array) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    zero_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, array);
}


/** Vector(Matrix) Addition */
__global__ void add_(unsigned int size, const double* vector1, const double* vector2, double* vectorOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    vectorOut[index] = vector1[index] + vector2[index];
}

void add(std::size_t size, const double* vector1, const double* vector2, double* vectorOut) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    add_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, vector1, vector2, vectorOut);
}


/** Vector(Matrix) Subtraction */
__global__ void subtract_(unsigned int size, const double* vector1, const double* vector2, double* vectorOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    vectorOut[index] = vector1[index] - vector2[index];
}

void subtract(std::size_t size, const double* vector1, const double* vector2, double* vectorOut) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    subtract_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, vector1, vector2, vectorOut);
}


/** Vector(Matrix) Multiplication (Hadamard Product) */
__global__ void multiply_(unsigned int size, const double* vector1, const double* vector2, double* vectorOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    vectorOut[index] = vector1[index] * vector2[index];
}

void multiply(std::size_t size, const double* vector1, const double* vector2, double* vectorOut) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    multiply_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, vector1, vector2, vectorOut);
}


/** Dot Product */
__global__ void dotVectorsWithMatrixOut_(
    unsigned int vector1Size, unsigned int vector2Size,
    const double* vector1, const double* vector2, double* matrixOut
) {
    const unsigned int index1 = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int index2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (index1 >= vector1Size || index2 >= vector2Size) return;
    matrixOut[index1 * vector2Size + index2] = vector1[index1] * vector2[index2];
}

void dotVectorsWithMatrixOut(
    std::size_t vector1Size, std::size_t vector2Size,
    const double* vector1, const double* vector2, double* matrixOut
) {
    dim3 gridDim((vector1Size + THREADS_PER_2D_BLOCK - 1) / THREADS_PER_2D_BLOCK,
                 (vector2Size + THREADS_PER_2D_BLOCK - 1) / THREADS_PER_2D_BLOCK);
    dim3 blockDim(THREADS_PER_2D_BLOCK, THREADS_PER_2D_BLOCK);
    dotVectorsWithMatrixOut_<<<gridDim, blockDim>>>(vector1Size, vector2Size, vector1, vector2, matrixOut);
}


__global__ void dotMatrixVectorMultiply_(
    unsigned int numOfMatrixRows, unsigned int numOfMatrixCols,
    const double* matrix, const double* vector, double* matrixOut
) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= numOfMatrixRows || col >= numOfMatrixCols) return;
    matrixOut[row * numOfMatrixCols + col] = matrix[row * numOfMatrixCols + col] * vector[col];
}

__global__ void dotMatrixVectorSumReduction_(unsigned int numOfMatrixRows, unsigned int numOfMatrixCols, double* matrix) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int sharedDataIndex = threadIdx.x * blockDim.y + threadIdx.y;
    if (row >= numOfMatrixRows) return;

    extern __shared__ double sharedData[];
    sharedData[sharedDataIndex] = (col < numOfMatrixCols) ? matrix[row * numOfMatrixCols + col] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.y / 2; stride > 0; stride >>= 1) {
        if (threadIdx.y < stride)
            sharedData[sharedDataIndex] += sharedData[sharedDataIndex + stride];
        __syncthreads();
    }

    if (threadIdx.y == 0) matrix[row * gridDim.y + blockIdx.y] = sharedData[sharedDataIndex];
}

void dotMatrixVector(
    std::size_t numOfMatrixRows, std::size_t numOfMatrixCols, const double* matrix,
    const double* vector, double* helperMatrix, double* vectorOut
) {
    dim3 gridDim((numOfMatrixRows + THREADS_PER_2D_BLOCK - 1) / THREADS_PER_2D_BLOCK,
                 (numOfMatrixCols + THREADS_PER_2D_BLOCK - 1) / THREADS_PER_2D_BLOCK);
    dim3 blockDim(THREADS_PER_2D_BLOCK, THREADS_PER_2D_BLOCK);
    std::size_t sharedMemSize = blockDim.x * blockDim.y * sizeof(double);

    dotMatrixVectorMultiply_<<<gridDim, blockDim>>>(numOfMatrixRows, numOfMatrixCols, matrix, vector, helperMatrix);

    while (numOfMatrixCols > 1) {
        dotMatrixVectorSumReduction_<<<gridDim, blockDim, sharedMemSize>>>(numOfMatrixRows, numOfMatrixCols, helperMatrix);
        numOfMatrixCols = gridDim.y;
        gridDim.y = (numOfMatrixCols + THREADS_PER_2D_BLOCK - 1) / THREADS_PER_2D_BLOCK;
    }

    std::size_t gridSize = (numOfMatrixRows + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    copy_<<<gridSize, THREADS_PER_1D_BLOCK>>>(numOfMatrixRows, helperMatrix, vectorOut);
}


/** Dot Product */
__global__ void dotVectorMatrix_(
    unsigned int numOfMatrixRows, unsigned int numOfMatrixCols,
    const double* vector, const double* matrix, double* vectorOut
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numOfMatrixCols) return;
    vectorOut[index] = 0;  // initialize all elements to 0
    for (unsigned int i = 0; i < numOfMatrixRows; ++i)
        vectorOut[index] += matrix[i * numOfMatrixCols + index] * vector[i];
}

void dotVectorMatrix(
    std::size_t numOfMatrixRows, std::size_t numOfMatrixCols,
    const double* vector, const double* matrix, double* vectorOut
) {
    std::size_t gridSize = (numOfMatrixCols + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    dotVectorMatrix_<<<gridSize, THREADS_PER_1D_BLOCK>>>(numOfMatrixRows, numOfMatrixCols, vector, matrix, vectorOut);
}


/** Sigmoid Function: sigmoid(z) = 1 / (1 + (e ^ -z)) */
__global__ void sigmoid_(unsigned int size, const double* z, double* functionOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    functionOut[index] = 1 / (1 + std::exp(-z[index]));
}

void sigmoid(std::size_t size, const double* z, double* functionOut) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    sigmoid_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, z, functionOut);
}


/** Derivative of Sigmoid Function: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)) */
__global__ void sigmoidPrime_(unsigned int size, const double* z, double* functionOut) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    double sigmoidZ = 1 / (1 + std::exp(-z[index]));
    functionOut[index] = sigmoidZ * (1 - sigmoidZ);
}

void sigmoidPrime(std::size_t size, const double* z, double* functionOut) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    sigmoidPrime_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, z, functionOut);
}


/** Reduces the cost with specified learning rate. */
__global__ void reduceCost_(
    unsigned int size, double learningRate,
    double miniBatchSize, const double* nabla, double* out
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size) return;
    out[index] -= (learningRate / miniBatchSize) * nabla[index];
}

void reduceCost(
    std::size_t size, double learningRate,
    double miniBatchSize, const double* nabla, double* out
) {
    std::size_t gridSize = (size + THREADS_PER_1D_BLOCK - 1) / THREADS_PER_1D_BLOCK;
    reduceCost_<<<gridSize, THREADS_PER_1D_BLOCK>>>(size, learningRate, miniBatchSize, nabla, out);
}