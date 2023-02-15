import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter

_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_STEP_COUNTER = 0


class TensorboardManager:

    def __init__(self, location, name=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), config=None):
        if not os.path.exists(location):
            os.mkdir(location)
        self.writer = SummaryWriter(location, comment=name)

    def log(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f'{k}', v, step)

    def log_time(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f'time/{k}', v, step)

    def log_train(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f'{k}/train', v, step)

    def log_eval(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f'{k}/eval', v, step)

    def log_zeroshot(self, result, step):
        for k, v in result.items():
            self.writer.add_scalar(f'{k}_acc/eval', v, step)


def set_tb_manager(launch_time, tensorboard_path):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, 'tensorboard writer')
    if torch.distributed.get_rank() == 0:
        _GLOBAL_TENSORBOARD_WRITER = TensorboardManager(tensorboard_path + f'/{launch_time}', launch_time)


def get_tb_manager() -> TensorboardManager:
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    assert _GLOBAL_TENSORBOARD_WRITER, "{} is None".format('tensorboard writer')
    return _GLOBAL_TENSORBOARD_WRITER


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def get_global_counter() -> int:
    global _GLOBAL_STEP_COUNTER
    return _GLOBAL_STEP_COUNTER


def set_global_counter(step) -> int:
    global _GLOBAL_STEP_COUNTER
    _GLOBAL_STEP_COUNTER = step
