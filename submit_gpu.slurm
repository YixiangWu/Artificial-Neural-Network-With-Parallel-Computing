#!/bin/bash
#SBATCH --output=output.txt
#SBATCH --time=12:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


# Purge modules and load correct ones
module purge
module load gcc/10.2.0
module load cuda/11.5.0
module load vtune/2019.4  # commandline interface: amplxe-cl

# Compile
nvcc main_base.cpp src/network.cpp -o main_base
nvcc main_cuda.cu src/network.cpp src/network_cuda.cu -o main_cuda

# Run
amplxe-cl -collect hotspots ./main_base
amplxe-cl -collect hotspots ./main_cuda
nvprof --normalized-time-unit ns --csv ./main_cuda


# Tip 1: submit this script by running > sbatch submit_gpu.slurm
# Tip 2: check the job status by running > squeue -u $USER -M gpu
# Tip 3: check the partition status by running > sinfo -M gpu
