import time
from contextlib import nullcontext

import torch
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from colossalai.utils import get_current_device
from colossalai.utils.profiler.legacy import CommProfiler, ProfilerContext


class DummyProfiler:

    def __init__(self):
        self.step_number = 0

    def step(self):
        self.step_number += 1


# Randomly Generated Data
def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def get_real_data(data_iter):
    # Copy from colossalai/engine/schedule/_base_schedule.py:29
    def _move_tensor(element):
        if torch.is_tensor(element):
            if not element.is_cuda:
                return element.to(get_current_device()).detach()
        return element

    # Copy from colossalai/engine/schedule/_base_schedule.py:35
    def _move_to_device(data):
        if isinstance(data, torch.Tensor):
            data = data.to(get_current_device())
        elif isinstance(data, (list, tuple)):
            data_to_return = []
            for element in data:
                if isinstance(element, dict):
                    data_to_return.append({k: _move_tensor(v) for k, v in element.items()})
                else:
                    data_to_return.append(_move_tensor(element))
            data = data_to_return
        elif isinstance(data, dict):
            data = {k: _move_tensor(v) for k, v in data.items()}
        else:
            raise TypeError(
                f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
        return data

    batch_data = next(data_iter)
    batch_data = _move_to_device(batch_data[0])
    input_ids, attn_mask = batch_data['input_ids'], batch_data['attention_mask']

    return input_ids, attn_mask


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_comm_profile_context(enable_flag):
    if enable_flag:
        return ProfilerContext([CommProfiler()])
    else:
        return nullcontext(DummyProfiler())


def get_torch_profile_context(enable_flag, warmup_steps, active_steps, save_dir):
    if enable_flag:
        return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=schedule(wait=15, warmup=warmup_steps, active=active_steps),
                       on_trace_ready=tensorboard_trace_handler(save_dir),
                       record_shapes=True,
                       profile_memory=True)
    else:
        return nullcontext(DummyProfiler())


def get_time_stamp():
    cur_time = time.strftime("%d-%H:%M", time.localtime())
    return cur_time
