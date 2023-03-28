#!/bin/bash
#SBATCH --output=output.txt
#SBATCH --time=01:00:00
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32


# Purge modules and load correct ones
module purge
module load gcc/10.2.0

# Compile
g++ main.cpp src/network.cpp src/network_openmp.cpp -o main -fopenmp

# Run
./main


# Tip 1: submit this script by running > sbatch submit_smp.slurm
# Tip 2: check the job status by running > squeue -u $USER -M smp
# Tip 3: check the partition status by running > sinfo -M smp
