#!/bin/bash
#SBATCH --job-name=k_means_c
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
echo "k_means_c started"
srun --mpi=pmix_v4 python k_means_c.py
echo "k_means_c finished"
