set -x

USE_SHARD_INIT=$1
NUM_NODES=$SLURM_JOB_NUM_NODES
HEAD_NODE_ADDR=`scontrol show hostname $SLURM_NODELIST | head -n1`

torchrun --nnodes=${NUM_NODES} \
--nproc_per_node=${GPUNUM} \
--rdzv_id=5877 \
--rdzv_backend=c10d \
--rdzv_endpoint=${HEAD_NODE_ADDR}:20311 \
./train_gpt_demo.py \
--tp_degree=${TPDEGREE} \
--model_type=${MODEL_TYPE} \
--batch_size=${BATCH_SIZE} \
--placement=${PLACEMENT} ${USE_SHARD_INIT} \
--distplan=${DISTPLAN} \
--train_step=${TRAIN_STEP}
