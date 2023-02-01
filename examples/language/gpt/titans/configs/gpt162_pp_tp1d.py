from model import GPT162_pipeline_hybrid

from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.shard_utils import TensorShardStrategy

BATCH_SIZE = 128 # global batch size
NUM_EPOCHS = 1
SEQ_LEN = 2048
NUM_MICRO_BATCHES = 128 # so micro batch size is 1
HIDDEN_SIZE = 20480
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

model = dict(type=GPT162_pipeline_hybrid, checkpoint=True, num_chunks=1)

parallel = dict(
    pipeline=8, # Pipeline-parallel size options = [2, 4, 8, 16, 32]
    tensor=dict(size=8, mode='1d'),    # for the current model implementation, mode can only be 1D or None
)
