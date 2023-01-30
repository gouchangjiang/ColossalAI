#!/bin/bash
#SBATCH --partition=llm
#SBATCH --job-name="test-colossalAI"
#SBATCH --nodes=4
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=128G

export DATA=/mnt/petrelfs/share_data/cjgou/small-gpt-dataset.json

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

srun python \
   train_gpt.py \
   --config 'configs/gpt2_small_zero3_pp1d.py' \
   --host $head_node
