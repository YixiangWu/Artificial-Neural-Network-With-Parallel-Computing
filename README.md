# Artificial-Neural-Network-With-Parallel-Computing

SHREC-SURG-2022 FIRST-PLACE UNDERGRADUATE RESEARCH PROJECT

## Neural Network

### Network Hyperparameters

* Network Size: {`PIXELS_PER_IMAGE`, ... (Hidden Layers), `NUM_OF_CLASSES`}
* Number of Epochs
* Mini-Batch Size
* Learning Rate

### Network Performances

|   #  |     Network Size    | Number of Epochs | Mini-Batch Size | Learning Rate |  Accuracy  |
| :--: | :-----------------: | :--------------: | :-------------: | :-----------: | :--------: |
|  --  |    {784, 100, 10}   |         5        |        10       |       3       | 9608/10000 |
| BASE |    {784, 100, 10}   |        30        |        10       |       3       | 9721/10000 |
|   1  |    {784, 100, 10}   |        30        |      **4**      |       3       | 9434/10000 |
|   2  |    {784, 100, 10}   |        30        |      **24**     |       3       | 9739/10000 |
|   3  |    {784, 100, 10}   |        30        |        10       |    **0.5**    | 9717/10000 |
|   4  |    {784, 100, 10}   |        30        |        10       |     **5**     | 9601/10000 |
|  --  |  {784, **800**, 10} |        30        |        24       |       5       | 9805/10000 |
|  --  | {784, **2000**, 10} |        30        |        24       |       5       | 9813/10000 |

---

## Neural Network Taking Advantage of Batch Gradient Descent Utilizing OpenMP

* Network Size: {784, 100, 10}
* Number of Epochs: 30
* Mini-Batch Size: 24
* Learning Rate: 3

### OpenMP System Specifications

* CPU: Intel Xeon Gold 6126 (2.60 GHz Base, 3.70 GHz Max Turbo) (24 Threads)
* RAM: 192 GB (8 GB/Thread)

### OpenMP Performances

| Number of Threads | Elapsed Time (s) | Speed Up | Efficiency |
| :---------------: | :--------------: | :------: | :--------: |
|         1         |     1164.735     |    1X    |    100%    |
|         4         |      329.691     |   3.53X  |    88.3%   |
|         8         |      175.763     |   6.63X  |    82.8%   |
|         12        |      125.060     |   9.31X  |    77.6%   |
|         24        |      75.865      |  15.35X  |    64.0%   |

### OpenMP In-Depth Statistics

| Number of Threads | Elapsed Time (s) | CPU Time (s) | Effective Time (s) | Spin Time (s) | Overhead Time (s) |
| :---------------: | :--------------: | :----------: | :----------------: | :-----------: | :---------------: |
|         1         |     1164.735     |   1164.168   |      1164.168      |       0       |         0         |
|         4         |      329.691     |   1311.540   |      1275.831      |     34.239    |       1.470       |
|         8         |      175.763     |   1397.568   |      1346.258      |     49.130    |       2.180       |
|         12        |      125.060     |   1486.421   |      1425.547      |     58.634    |       2.240       |
|         24        |      75.865      |   1785.050   |      1667.276      |    114.774    |       3.000       |

---

## Neural Network with Tensor-Operation Accelerations Utilizing CUDA

* Network Size: {784, 100, 10}
* Number of Epochs: 30
* Mini-Batch Size: 24
* Learning Rate: 3

### CUDA System Specifications

* CPU: Intel Xeon Gold 6130 (2.10 GHz Base, 3.70 GHz Max Turbo)
* GPU: NVIDIA V100 32GB PCIe

### CUDA Performances

| GPU  | Elapsed Time (s) | Speed Up |
| :--: | :--------------: | :------: |
| BASE |     2347.436     |    1X    |
| V100 |      184.621     |  12.71X  |

### CUDA API Statistics

| Time (%) | Total Time (ns) |    Calls   |  Avg (ns) | Min (ns) |   Max (ns)  |         Name         |
| :------: | :-------------: | :--------: | :-------: | :------: | :---------: | :------------------: |
|  83.180  | 205,113,606,186 | 43,500,000 |   4,715   |   3,586  | 101,588,538 |   cudaLaunchKernel   |
|  16.669  |  41,103,870,378 |  4,200,002 |   9,786   |   3,228  | 147,246,447 |      cudaMemcpy      |
|   0.149  |   367,148,951   |     103    | 3,564,552 |   2,872  | 364,186,172 |      cudaMalloc      |
|   0.002  |    4,107,400    |     103    |   39,877  |   2,738  |  2,289,262  |       cudaFree       |
|   0.000  |     535,072     |      1     |  535,072  |  535,072 |   535,072   |  cuDeviceGetPCIBusId |
|   0.000  |     495,980     |     101    |   4,910   |    191   |   222,688   | cuDeviceGetAttribute |
|   0.000  |      80,871     |      1     |   80,871  |  80,871  |    80,871   |    cuDeviceGetName   |
|   0.000  |      2,326      |      3     |    775    |    237   |    1,808    |   cuDeviceGetCount   |
|   0.000  |       845       |      2     |    422    |    197   |     648     |      cuDeviceGet     |
|   0.000  |       405       |      1     |    405    |    405   |     405     |   cuDeviceTotalMem   |
|   0.000  |       317       |      1     |    317    |    317   |     317     |    cuDeviceGetUuid   |

### CUDA Kernel Statistics

| Time (%) | Total Time (ns) |   Calls   | Avg (ns) | Min (ns) | Max (ns) |                                               Name                                               |
| :------: | :-------------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: |
|  33.919  |  40,850,857,062 | 8,400,000 |   4,863  |   2,431  |  23,584  |               `dotMatrixVectorSumReduction_(unsigned int, unsigned int, double *)`               |
|  15.917  |  19,170,215,054 | 4,200,000 |   4,564  |   1,599  |  23,871  | `dotMatrixVectorMultiply_(unsigned int, unsigned int, double const *, double const *, double *)` |
|  11.993  |  14,443,978,099 | 3,600,000 |   4,012  |   1,599  |  13,888  | `dotVectorsWithMatrixOut_(unsigned int, unsigned int, double const *, double const *, double *)` |
|   8.693  |  10,469,156,974 | 7,800,000 |   1,342  |   1,119  |  21,632  |                  `add_(unsigned int, double const *, double const *, double *)`                  |
|   5.617  |  6,764,913,111  | 3,900,002 |   1,734  |   1,311  |  54,880  |                                         CUDA memcpy HtoD                                         |
|   4.964  |  5,978,685,407  | 4,200,000 |   1,423  |   1,375  |  14,016  |                        `sigmoid_(unsigned int, double const *, double *)`                        |
|   4.751  |  5,721,753,457  | 1,800,000 |   3,178  |   3,135  |  13,696  |     `dotVectorMatrix_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   4.194  |  5,050,999,439  | 3,600,000 |   1,403  |   1,343  |  13,952  |                      `sigmoidPrime_(unsigned int, double const *, double *)`                     |
|   3.993  |  4,809,167,630  | 4,200,000 |   1,145  |   1,087  |  16,000  |                          `copy_(unsigned int, double const *, double *)`                         |
|   3.471  |  4,180,108,410  | 3,600,000 |   1,161  |   1,119  |  14,208  |                `multiply_(unsigned int, double const *, double const *, double *)`               |
|   1.771  |  2,132,782,477  | 1,800,000 |   1,184  |   1,151  |  13,920  |                `subtract_(unsigned int, double const *, double const *, double *)`               |
|   0.376  |   452,682,466   |  300,000  |   1,508  |   1,471  |  13,952  |                                         CUDA memcpy DtoH                                         |
|   0.200  |   240,337,545   |  150,000  |   1,602  |   1,279  |  13,344  |               `reduceCost_(unsigned int, double, double, double const *, double *)`              |
|   0.141  |   169,265,994   |  150,000  |   1,128  |    927   |  13,664  |                                  `zero_(unsigned int, double *)`                                 |
