#!/bin/bash
#SBATCH --job-name=multi_machine_multi_core
#SBATCH --partition=partition
#SBATCH --mem=1000mb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2


echo "###INFO"
echo "Nodes: "$SLURM_JOB_NODELIST
echo "Total nodes: "$SLURM_JOB_NUM_NODES
echo "tasks per node: "$SLURM_NTASKS_PER_NODE
echo "rdzv: " 172.18.35.205

export MASTER_PORT=$(shuf -i 2000-65000 -n 1)
export RENDEZVOUS_ID=1114
export WORLD_SIZE=2
export MASTER_ADDR=172.18.35.205
export OMP_NUM_THREADS=1

###RUN
srun torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=$RENDEZVOUS_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT  MMMC.py
