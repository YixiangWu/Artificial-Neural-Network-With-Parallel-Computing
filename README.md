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
|         4         |      366.520     |   3.18X  |    79.4%   |
|         8         |      209.620     |   5.56X  |    69.5%   |
|         12        |      156.799     |   7.43X  |    61.9%   |
|         24        |      100.296     |  11.61X  |    48.4%   |

### OpenMP In-Depth Statistics

| Number of Threads | Elapsed Time (s) | CPU Time (s) | Effective Time (s) | Spin Time (s) | Overhead Time (s) |
| :---------------: | :--------------: | :----------: | :----------------: | :-----------: | :---------------: |
|         1         |     1164.735     |   1164.168   |      1164.168      |       0       |         0         |
|         4         |      366.520     |   1461.790   |      1402.811      |     58.119    |       0.860       |
|         8         |      209.620     |   1668.173   |      1540.179      |    125.804    |       2.190       |
|         12        |      156.799     |   1867.312   |      1537.380      |    325.582    |       4.350       |
|         24        |      100.296     |   2356.824   |      1895.827      |    455.367    |       5.630       |

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
| V100 |      331.285     |   7.09X  |

### CUDA API Statistics

| Time (%) | Total Time (ns) |    Calls   |  Avg (ns) | Min (ns) |   Max (ns)  |         Name         |
| :------: | :-------------: | :--------: | :-------: | :------: | :---------: | :------------------: |
|  58.049  | 197,582,887,601 |  4,200,002 |   47,043  |   3,445  | 106,218,542 |      cudaMemcpy      |
|  41.834  | 142,391,642,138 | 30,900,000 |   4,608   |   3,467  |  10,034,909 |   cudaLaunchKernel   |
|   0.109  |   369,942,697   |     102    | 3,626,889 |   2,469  | 360,667,929 |      cudaMalloc      |
|   0.008  |    27,288,052   |     102    |  267,529  |   2,287  |  13,413,120 |       cudaFree       |
|   0.000  |     460,039     |     101    |   4,554   |    176   |   205,218   | cuDeviceGetAttribute |
|   0.000  |      89,309     |      1     |   89,309  |  89,309  |    89,309   |    cuDeviceGetName   |
|   0.000  |      8,483      |      1     |   8,483   |   8,483  |    8,483    |  cuDeviceGetPCIBusId |
|   0.000  |      2,814      |      3     |    938    |    244   |    2,316    |   cuDeviceGetCount   |
|   0.000  |       796       |      2     |    398    |    182   |     614     |      cuDeviceGet     |
|   0.000  |       563       |      1     |    563    |    563   |     563     |   cuDeviceTotalMem   |
|   0.000  |       287       |      1     |    287    |    287   |     287     |    cuDeviceGetUuid   |

### CUDA Kernel Statistics

| Time (%) | Total Time (ns) |   Calls   | Avg (ns) | Min (ns) | Max (ns) |                                               Name                                               |
| :------: | :-------------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: |
|  80.510  | 229,366,243,782 | 4,200,000 |  54,611  |   8,959  |  151,423 |     `dotMatrixVector_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   5.111  |  14,560,673,129 | 3,600,000 |   4,044  |   1,631  |  17,279  | `dotVectorsWithMatrixOut_(unsigned int, unsigned int, double const *, double const *, double *)` |
|   3.665  |  10,441,792,595 | 7,800,000 |   1,338  |   1,119  |  13,920  |                  `add_(unsigned int, double const *, double const *, double *)`                  |
|   2.363  |  6,733,294,310  | 3,900,002 |   1,726  |   1,311  |  54,784  |                                         CUDA memcpy HtoD                                         |
|   2.061  |  5,870,939,237  | 4,200,000 |   1,397  |   1,343  |  13,728  |                        `sigmoid_(unsigned int, double const *, double *)`                        |
|   1.985  |  5,656,231,909  | 1,800,000 |   3,142  |   3,072  |  22,528  |     `dotVectorMatrix_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   1.769  |  5,039,408,309  | 3,600,000 |   1,399  |   1,344  |  13,919  |                      `sigmoidPrime_(unsigned int, double const *, double *)`                     |
|   1.484  |  4,228,892,912  | 3,600,000 |   1,174  |   1,120  |  13,760  |                `multiply_(unsigned int, double const *, double const *, double *)`               |
|   0.748  |  2,130,368,085  | 1,800,000 |   1,183  |   1,151  |  13,632  |                `subtract_(unsigned int, double const *, double const *, double *)`               |
|   0.158  |   450,674,475   |  300,000  |   1,502  |    800   |  13,504  |                                         CUDA memcpy DtoH                                         |
|   0.085  |   243,036,832   |  150,000  |   1,620  |   1,279  |  13,344  |               `reduceCost_(unsigned int, double, double, double const *, double *)`              |
|   0.060  |   171,811,490   |  150,000  |   1,145  |    991   |  13,120  |                                  `zero_(unsigned int, double *)`                                 |
