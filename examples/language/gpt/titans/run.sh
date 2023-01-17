#!/bin/bash
#SBATCH --partition=llm
#SBATCH --job-name="test-colossalAI"
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G

export DATA=/mnt/petrelfs/share_data/small-gpt-dataset.json

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

NUM_NODES=4
nodes_list=""
for i in $(seq 0 $(($NUM_NODES>0? $NUM_NODES-1:0))); do
    node_name=${nodes_array[i]}
    nodes_list="${nodes_list}${nodes_list:+,}$node_name"
done

echo $nodes_list

# bug, 16 Jan 2023
# colossalai run --nproc_per_node=16 \
#    --master_addr $head_node \
#    --master_port "5877" \
#    --host $nodes_list \
#    train_gpt.py \
#    --config configs/gpt3_zero3_pp1d.py \
#    --from_torch

srun python \
   train_gpt.py \
   --config configs/gpt3_zero3_pp1d.py \
   --host $head_node