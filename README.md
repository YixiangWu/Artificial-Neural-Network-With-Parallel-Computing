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
|         1         |     1164.952     |    1X    |    100%    |
|         4         |      321.102     |   3.63X  |    90.7%   |
|         8         |      183.791     |   6.34X  |    79.2%   |
|         12        |      135.429     |   8.60X  |    71.7%   |
|         24        |      87.440      |  13.32X  |    55.5%   |

### OpenMP In-Depth Statistics

| Number of Threads | Elapsed Time (s) | CPU Time (s) | Effective Time (s) | Spin Time (s) | Overhead Time (s) |
| :---------------: | :--------------: | :----------: | :----------------: | :-----------: | :---------------: |
|         1         |     1164.952     |   1164.916   |      1164.916      |       0       |         0         |
|         4         |      321.102     |   1280.628   |      1238.830      |     40.978    |       0.820       |
|         8         |      183.791     |   1460.160   |      1354.762      |    103.498    |       1.900       |
|         12        |      135.429     |   1606.428   |      1449.708      |    153.619    |       3.100       |
|         24        |      87.440      |   2066.756   |      1687.112      |    374.203    |       5.440       |

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
| BASE |     2333.864     |    1X    |
| V100 |      347.389     |   6.72X  |

### CUDA API Statistics

| Time (%) | Total Time (ns) |    Calls   | Avg (ns) | Min (ns) |   Max (ns)  |         Name         |
| :------: | :-------------: | :--------: | :------: | :------: | :---------: | :------------------: |
|  41.096  | 155,045,597,732 |  7,200,068 |  21,533  |   1,669  |  6,812,344  |       cudaFree       |
|  38.700  | 146,005,921,896 | 30,900,000 |   4,725  |   3,803  |  6,718,662  |   cudaLaunchKernel   |
|  15.700  |  59,232,850,946 |  4,200,002 |  14,103  |   3,286  | 114,582,772 |      cudaMemcpy      |
|   4.505  |  16,995,959,581 |  7,200,068 |   2,360  |   1,805  | 349,176,716 |      cudaMalloc      |
|   0.000  |     447,248     |     101    |   4,428  |    173   |   196,895   | cuDeviceGetAttribute |
|   0.000  |      73,431     |      1     |  73,431  |  73,431  |    73,431   |    cuDeviceGetName   |
|   0.000  |      7,511      |      1     |   7,511  |   7,511  |    7,511    |  cuDeviceGetPCIBusId |
|   0.000  |      2,000      |      3     |    666   |    228   |    1,539    |   cuDeviceGetCount   |
|   0.000  |       745       |      1     |    745   |    745   |     745     |   cuDeviceTotalMem   |
|   0.000  |       672       |      2     |    336   |    170   |     502     |      cuDeviceGet     |
|   0.000  |       272       |      1     |    272   |    272   |     272     |    cuDeviceGetUuid   |

### CUDA Kernel Statistics

| Time (%) | Total Time (ns) |   Calls   | Avg (ns) | Min (ns) | Max (ns) |                                               Name                                               |
| :------: | :-------------: | :-------: | :------: | :------: | :------: | :----------------------------------------------------------------------------------------------: |
|  79.986  | 223,956,621,061 | 4,200,000 |  53,323  |   8,959  |  149,727 |     `dotMatrixVector_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   5.209  |  14,586,170,034 | 3,600,000 |   4,051  |   1,631  |  13,888  | `dotVectorsWithMatrixOut_(unsigned int, unsigned int, double const *, double const *, double *)` |
|   3.749  |  10,497,297,719 | 7,800,000 |   1,345  |   1,119  |  24,704  |                  `add_(unsigned int, double const *, double const *, double *)`                  |
|   2.412  |  6,752,526,514  | 3,900,002 |   1,731  |   1,311  |  55,039  |                                         CUDA memcpy HtoD                                         |
|   2.154  |  6,030,767,262  | 4,200,000 |   1,435  |   1,375  |  16,064  |                        `sigmoid_(unsigned int, double const *, double *)`                        |
|   2.052  |  5,744,499,605  | 1,800,000 |   3,191  |   3,072  |  13,728  |     `dotVectorMatrix_(unsigned int, unsigned int, double const *, double const *, double *)`     |
|   1.871  |  5,239,536,150  | 3,600,000 |   1,455  |   1,407  |  19,232  |                      `sigmoidPrime_(unsigned int, double const *, double *)`                     |
|   1.527  |  4,275,448,111  | 3,600,000 |   1,187  |   1,151  |  23,263  |                `multiply_(unsigned int, double const *, double const *, double *)`               |
|   0.731  |  2,045,859,482  | 1,800,000 |   1,136  |   1,088  |  20,224  |                `subtract_(unsigned int, double const *, double const *, double *)`               |
|   0.163  |   455,400,053   |  300,000  |   1,518  |    800   |  13,920  |                                         CUDA memcpy DtoH                                         |
|   0.086  |   239,676,104   |  150,000  |   1,597  |   1,279  |  12,512  |               `reduceCost_(unsigned int, double, double, double const *, double *)`              |
|   0.061  |   172,124,025   |  150,000  |   1,147  |    960   |  13,504  |                                  `zero_(unsigned int, double *)`                                 |
