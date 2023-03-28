#!/bin/bash
#SBATCH --output=output.txt
#SBATCH --time=01:00:00
#SBATCH --cluster=gpu
#SBATCH --partition=gtx1080
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


# Purge modules and load correct ones
module purge
module load gcc/10.2.0
module load cuda/11.5.0

# Compile
nvcc main.cu src/network.cpp src/network_cuda.cu -o main

# Run
nvprof ./main


# Tip 1: submit this script by running > sbatch submit_gpu.slurm
# Tip 2: check the job status by running > squeue -u $USER -M gpu
# Tip 3: check the partition status by running > sinfo -M gpu
