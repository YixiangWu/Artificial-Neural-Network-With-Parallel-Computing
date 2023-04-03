#!/bin/bash
#SBATCH --output=output.txt
#SBATCH --time=12:00:00
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24


# Purge modules and load correct ones
module purge
module load gcc/10.2.0
module load vtune/2019.4  # commandline interface: amplxe-cl

# Compile
g++ main_base.cpp src/network.cpp src/network_base.cpp -o main_base
g++ main_openmp.cpp src/network.cpp src/network_openmp.cpp -o main_openmp -fopenmp

# Run
amplxe-cl -collect hotspots ./main_base
amplxe-cl -collect hotspots ./main_openmp


# Tip 1: submit this script by running > sbatch submit_smp.slurm
# Tip 2: check the job status by running > squeue -u $USER -M smp
# Tip 3: check the partition status by running > sinfo -M smp
