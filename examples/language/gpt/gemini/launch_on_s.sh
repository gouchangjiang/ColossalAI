#!/bin/bash
#SBATCH --partition=caif_rd
#SBATCH --job-name="GE"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=profile_gpt6b_tp8_ZeROtwo8_BS32.log

# distplan in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]
export DISTPLAN=${DISTPLAN:-"CAI_ZeRO2"}

# The following options only valid when DISTPLAN="CAI_Gemini"
export GPUNUM=${GPUNUM:-8}
export TPDEGREE=${TPDEGREE:-8}
export PLACEMENT=${PLACEMENT:-"cuda"} # or cuda
export USE_SHARD_INIT=${USE_SHARD_INIT:-False}
export BATCH_SIZE=${BATCH_SIZE:-4} # batch size per DP degree
export MODEL_TYPE=${MODEL_TYPE:-"gpt2_6b"}
export TRAIN_STEP=${TRAIN_STEP:-100}
export DATASET_PATH=${DATASET_PATH:-"/mnt/petrelfs/share_data/cjgou/small-gpt-dataset.json"}
# export PYTHONPATH=$PWD:$PYTHONPATH

if [ ${USE_SHARD_INIT} = "True" ]; then
  USE_SHARD_INIT="--shardinit"
else
  USE_SHARD_INIT=""
fi

mkdir -p gemini_logs

srun sh ./run_gemini_multi_node.sh ${USE_SHARD_INIT}
