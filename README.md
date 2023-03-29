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
|         1         |     1174.212     |    1X    |    100%    |
|         4         |      338.590     |   3.47X  |    86.7%   |
|         8         |      196.882     |   5.96X  |    74.6%   |
|         12        |      148.534     |   7.91X  |    65.9%   |
|         24        |      100.301     |  11.71X  |    48.8%   |

### OpenMP In-Depth Statistics

| Number of Threads | Elapsed Time (s) | CPU Time (s) | Effective Time (s) | Spin Time (s) | Overhead Time (s) |
| :---------------: | :--------------: | :----------: | :----------------: | :-----------: | :---------------: |
|         1         |     1174.212     |   1173.948   |      1173.948      |       0       |         0         |
|         4         |      338.590     |   1300.818   |      1251.058      |     49.049    |       0.710       |
|         8         |      196.882     |   1449.643   |      1356.866      |     91.047    |       1.730       |
|         12        |      148.534     |   1579.944   |      1457.995      |    119.449    |       2.500       |
|         24        |      100.301     |   1997.491   |      1689.278      |    303.223    |       4.990       |

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
| BASE |     2367.721     |    1X    |
| V100 |      378.529     |   6.26X  |

### CUDA API Statistics

| Time (%) | Total Time (ns) |    Calls   |  Avg (ns) |  Min (ns) |   Max (ns)  |         Name         |
| :------: | :-------------: | :--------: | :-------: | :-------: | :---------: | :------------------: |
|  41.475  | 158,271,032,774 |  7,200,068 |   21,981  |   1,635   |  7,057,387  |       cudaFree       |
|  37.678  | 143,778,265,179 | 30,900,000 |   4,653   |   3,703   |  8,750,435  |   cudaLaunchKernel   |
|  16.427  |  62,685,997,897 |  4,200,002 |   14,925  |   3,361   |  94,404,515 |      cudaMemcpy      |
|   4.419  |  16,864,055,729 |  7,200,068 |   2,342   |   1,782   | 373,557,410 |      cudaMalloc      |
|   0.000  |    1,735,639    |      1     | 1,735,639 | 1,735,639 |  1,735,639  |    cuDeviceGetName   |
|   0.000  |     472,616     |     101    |   4,679   |    177    |   213,346   | cuDeviceGetAttribute |
|   0.000  |      9,433      |      1     |   9,433   |   9,433   |    9,433    |  cuDeviceGetPCIBusId |
|   0.000  |      1,941      |      3     |    647    |    233    |    1,454    |   cuDeviceGetCount   |
|   0.000  |       661       |      2     |    330    |    193    |     468     |      cuDeviceGet     |
|   0.000  |       402       |      1     |    402    |    402    |     402     |   cuDeviceTotalMem   |
|   0.000  |       277       |      1     |    277    |    277    |     277     |    cuDeviceGetUuid   |

### CUDA Kernel Statistics

| Time (%) | Total Time (ns) |   Calls   | Avg (ns) | Min (ns) | Max (ns) |                                               Name                                               |
| :------: | :-------------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: |
|  80.056  | 225,417,034,828 | 4,200,000 |  53,670  |   8,959  |  151,070 |     `dotMatrixVector_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   5.154  |  14,512,906,935 | 3,600,000 |   4,031  |   1,631  |  19,743  | `dotVectorsWithMatrixOut_(unsigned int, unsigned int, double const *, double const *, double *)` |
|   3.745  |  10,546,105,650 | 7,800,000 |   1,352  |   1,119  |  25,375  |                  `add_(unsigned int, double const *, double const *, double *)`                  |
|   2.399  |  6,754,057,086  | 3,900,002 |   1,731  |   1,311  |  54,464  |                                         CUDA memcpy HtoD                                         |
|   2.161  |  6,085,614,644  | 4,200,000 |   1,448  |   1,407  |  20,544  |                        `sigmoid_(unsigned int, double const *, double *)`                        |
|   2.058  |  5,794,810,606  | 1,800,000 |   3,219  |   3,104  |  17,984  |     `dotVectorMatrix_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   1.847  |  5,201,910,607  | 3,600,000 |   1,444  |   1,375  |  21,696  |                      `sigmoidPrime_(unsigned int, double const *, double *)`                     |
|   1.520  |  4,280,726,856  | 3,600,000 |   1,189  |   1,151  |  23,168  |                `multiply_(unsigned int, double const *, double const *, double *)`               |
|   0.751  |  2,114,655,756  | 1,800,000 |   1,174  |   1,151  |  19,616  |                `subtract_(unsigned int, double const *, double const *, double *)`               |
|   0.161  |   454,256,510   |  300,000  |   1,514  |    768   |  13,984  |                                         CUDA memcpy DtoH                                         |
|   0.085  |   239,532,726   |  150,000  |   1,596  |   1,279  |  13,407  |               `reduceCost_(unsigned int, double, double, double const *, double *)`              |
|   0.061  |   171,900,756   |  150,000  |   1,146  |    960   |  13,344  |                                  `zero_(unsigned int, double *)`                                 |
