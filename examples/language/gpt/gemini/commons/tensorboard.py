import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter

_GLOBAL_TENSORBOARD_WRITER = None


class TensorboardLog:

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


def set_tensorboard_writer(launch_time, tensorboard_path):
    """Set tensorboard writer."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER, 'tensorboard writer')
    if torch.distributed.get_rank() == 0:
        _GLOBAL_TENSORBOARD_WRITER = TensorboardLog(tensorboard_path + f'/{launch_time}', launch_time)


def get_tensorboard_writer():
    """Return tensorboard writer. It can be None so no need
    to check if it is initialized."""
    return _GLOBAL_TENSORBOARD_WRITER


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)
