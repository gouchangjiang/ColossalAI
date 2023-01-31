#!/bin/bash
#SBATCH --partition=caif_rd
#SBATCH --job-name="test-nlp"
#SBATCH --nodes=8
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G

# export DATA=/mnt/petrelfs/share_data/cjgou/small-gpt-dataset.json
export DATA=/mnt/petrelfs/share_data/cjgou/proxy_raw.json

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

MODEL_PARA_CONF=$1 

srun python \
   train_gpt.py \
   --config ${MODEL_PARA_CONF} \
   --host $head_node
