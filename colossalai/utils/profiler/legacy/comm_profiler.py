import inspect
from pathlib import Path
from functools import partial
import torch
from torch.autograd.profiler import profile
import torch.distributed as dist
from torch.distributed import ReduceOp, get_backend
from colossalai.utils import get_current_device
from .prof_utils import BaseProfiler, _format_time, _format_memory, _format_bandwidth
from typing import List, Optional
from torch.autograd.profiler_util import FunctionEvent
from dataclasses import dataclass
from colossalai.logging import get_dist_logger
from colossalai.utils.profiler.legacy.tb_writer import get_global_counter, get_tb_manager
from torch.utils.tensorboard import SummaryWriter


def _get_code_location(depth: int):
    ret = []
    length = min(len(inspect.stack()), depth + 1)
    for i in range(3, length):
        upper_frame = inspect.stack()[i]
        function_name = inspect.stack()[i - 1].function
        ret.append(upper_frame.filename)
        ret.append('(')
        ret.append(str(upper_frame.lineno))
        ret.append('): ')
        ret.append(function_name)
        if i != length - 1:
            ret.append('\n')

    return ''.join(ret)


torch_all_reduce = dist.all_reduce
torch_all_gather = dist.all_gather
torch_reduce_scatter = dist.reduce_scatter
torch_broadcast = dist.broadcast
torch_reduce = dist.reduce


# @dataclass
class CommEvent:
    """Communication Event. Used for communication time and communication
    volume recording.
    """

    def __init__(self):
        self.self_count = 0
        self.self_comm_vol = 0.0
        self.self_cuda_time = 0.0
        self.max_self_cuda_time = 0
        self.min_self_cuda_time = float('inf')

        self.copy_cuda_time = 0.0
        self.h2d_time = 0.0
        self.h2d_count = 0
        self.d2h_time = 0.0
        self.d2h_count = 0

    def add(self, rhs):
        self.self_count += rhs.self_count
        self.self_comm_vol += rhs.self_comm_vol
        self.self_cuda_time += rhs.self_cuda_time

        self.max_self_cuda_time = max(self.max_self_cuda_time, rhs.self_cuda_time)
        self.min_self_cuda_time = min(self.min_self_cuda_time, rhs.self_cuda_time)

        self.copy_cuda_time = rhs.copy_cuda_time
        self.h2d_count = rhs.h2d_count
        self.h2d_time = rhs.h2d_time
        self.d2h_count = rhs.d2h_count
        self.d2h_time = rhs.d2h_time

    def __str__(self) -> str:
        return "self_count:{}, self_comm_vol:{}, self_cuda_time:{}, copy_cuda_time:{}".format(
            self.self_count, self.self_comm_vol, self.self_cuda_time, self.copy_cuda_time)


class CommProfiler(BaseProfiler):
    """Communication profiler. Records all communication events.
    """
    PCIE_KN_SET = {"Memcpy HtoD", "Memcpy DtoH", "aten::copy_"}

    def __init__(self,
                 depth: int = 0,
                 total_count: int = 0,
                 total_comm_vol: float = 0,
                 total_cuda_time: int = 0,
                 writer: SummaryWriter = None):
        super().__init__(profiler_name="Collective_Communication", priority=0)
        self.depth = 3 + depth
        self.total_count = total_count
        self.total_comm_vol = total_comm_vol
        self.total_cuda_time = total_cuda_time

        self.total_copy_cuda_time = 0
        self.total_h2d_count = 0
        self.total_h2d_time = 0
        self.total_d2h_count = 0
        self.total_d2h_time = 0

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

        self.logger = get_dist_logger()
        if dist.get_rank() == 0:
            self.writer = writer if writer is not None else get_tb_manager().writer
        else:
            self.write = None

    def reset(self):
        self.total_count = 0
        self.total_comm_vol = 0
        self.total_cuda_time = 0

        self.ops_record = dict()
        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None
        self.warn_flag = False

        self.total_copy_cuda_time = 0
        self.total_h2d_count = 0
        self.total_h2d_time = 0
        self.total_d2h_count = 0
        self.total_d2h_time = 0

    def enable(self):
        dist.all_reduce = partial(all_reduce, profiler=self)
        dist.all_gather = partial(all_gather, profiler=self)
        dist.reduce_scatter = partial(reduce_scatter, profiler=self)
        dist.broadcast = partial(broadcast, profiler=self)
        dist.reduce = partial(reduce, profiler=self)

    def disable(self):
        dist.all_reduce = torch_all_reduce
        dist.all_gather = torch_all_gather
        dist.reduce_scatter = torch_reduce_scatter
        dist.broadcast = torch_broadcast
        dist.reduce = torch_reduce

    def step_to_tensorboard(self, kn, time, count):
        # Watch out tensorboard webpage overload
        self.writer.add_scalar(f'Collective Communication/{kn}', time, count)
        self.writer.add_scalar(f'Collective Communication/step', count, get_global_counter())

    def to_tensorboard(self, writer=None):
        if self.writer:
            self.writer.add_text(tag="Collective Communication", text_string=self.result_str())

    def to_file(self, filename: Path):
        with open(filename, "w") as f:
            f.write(self.result_str())

    def show(self):
        print(self.result_str())

    def result_str(self, sep: str = "\n"):
        res = []

        def append(s: str = None):
            if s is not None:
                res.append(s)
            res.append(sep)

        if self.warn_flag:
            append("Warnning: there exists multiple communication operations in the same time. As a result, "
                   "the profiling result is not accurate.")

        if self.total_cuda_time == 0:
            return "No collective communication has been called yet!"

        append("Collective communication profiling result:")
        append("total cuda time: {}".format(_format_time(self.total_cuda_time)))
        append("average bandwidth: {}".format(_format_bandwidth(self.total_comm_vol, self.total_cuda_time)))
        append("total number of calls: {}".format(self.total_count))
        append("time of data transmission (CPU -> GPU): {}".format(_format_time(self.total_h2d_time)))
        append("number of transmission (CPU -> GPU): {}".format(self.total_h2d_count))
        append("time of data transmission (GPU -> CPU): {}".format(_format_time(self.total_d2h_time)))
        append("number of transmission (GPU -> CPU): {}".format(self.total_d2h_count))

        append("All events:")

        seperation = '-' * 74
        row_format = '{:^10}' + '{:^12}' * 2 + '{:^16}' + '{:^12}' * 3

        append(seperation)
        append(
            row_format.format('Location', 'GPU time', 'Percentage', 'Comm volume', 'Bandwidth', 'PCIe BW',
                              'Num of calls'))
        append(seperation)

        show_list = sorted(self.ops_record.items(), key=lambda kv: -kv[1].self_cuda_time)
        for location, event in show_list:
            event: CommEvent
            append(location)
            append(
                row_format.format('', _format_time(event.self_cuda_time),
                                  '{:.1f}%'.format(event.self_cuda_time / self.total_cuda_time * 100.0),
                                  _format_memory(event.self_comm_vol),
                                  _format_bandwidth(event.self_comm_vol, event.self_cuda_time),
                                  _format_bandwidth(event.self_comm_vol, event.copy_cuda_time), event.self_count))
            append()

        return ''.join(res)

    @property
    def has_aync_op(self):
        return self.pending_op is not None

    def activate_profiler(self, kn: str, vol: float, backend: str = "nccl", async_op: bool = False):
        self.pending_metadata = (kn, _get_code_location(self.depth), vol, backend, async_op)
        self.profiler = profile(enabled=True, use_cuda=True, use_cpu=True, use_kineto=True)
        self.profiler.__enter__()

    def close_profiler(self, group=None):
        assert self.profiler is not None, "There is no running dist op"
        kernel_name, code_location, vol, backend, async_op = self.pending_metadata
        self.profiler.__exit__(None, None, None)
        sync_time = 0
        now_kn_count = 0

        if self.profiler.enabled and dist.get_world_size(group) > 1:
            curr_event = CommEvent()
            for event in self.profiler.function_events:
                event: FunctionEvent
                if event.name == "cudaDeviceSynchronize":
                    sync_time = event.self_cpu_time_total
                    continue
                elif event.name == "Memcpy HtoD":
                    curr_event.h2d_time += event.cuda_time_total
                    curr_event.h2d_count += 1
                elif event.name == "Memcpy DtoH":
                    curr_event.d2h_count += 1
                    curr_event.d2h_time += event.cuda_time_total
                elif event.name == "aten::copy_":
                    if len(event.input_shapes) == 0 or len(
                            event.input_shapes[0]) == 0 or event.cuda_time_total == 0 or len(event.stack) == 0:
                        continue
                    curr_event.copy_cuda_time = event.cuda_time_total

                if kernel_name in event.name:
                    curr_event.self_count = 1
                    curr_event.self_comm_vol = vol
                    curr_event.self_cuda_time = event.self_cuda_time_total

            if curr_event.self_count == 0:
                curr_event.self_count = 1
                curr_event.self_comm_vol = vol
                curr_event.self_cuda_time = sync_time
            else:
                buffer = torch.tensor([curr_event.self_cuda_time], device=get_current_device())
                torch_all_reduce(buffer, op=ReduceOp.MIN, group=group)
                curr_event.self_cuda_time = buffer.item()

            if curr_event.self_count != 1:
                if dist.get_rank() == 0:
                    print("dist op num is not equal with 1", flush=True)
                    print("kernel_name:{}, code_location:{}, vol:{}, backend:{}, async_op:{}".format(
                        kernel_name, code_location, vol, backend, async_op),
                          flush=True)
                    print(self.profiler.function_events, flush=True)
                self.logger.error("The number of communication primitives != 1.")

            self.total_count += curr_event.self_count
            self.total_comm_vol += curr_event.self_comm_vol
            self.total_cuda_time += curr_event.self_cuda_time

            self.total_copy_cuda_time += curr_event.copy_cuda_time
            self.total_h2d_count += curr_event.h2d_count
            self.total_h2d_time = +curr_event.h2d_time
            self.total_d2h_count += curr_event.d2h_count
            self.total_d2h_time += curr_event.d2h_time

            if code_location in self.ops_record:
                self.ops_record[code_location].add(curr_event)
            else:
                self.ops_record[code_location] = curr_event

            if dist.get_rank() == 0:
                self.step_to_tensorboard(code_location, curr_event.self_cuda_time,
                                         self.ops_record[code_location].self_count)

        self.profiler = None
        self.pending_op = None
        self.pending_metadata = None

    def wait_async_op(self):
        if self.pending_op is not None:
            op = self.pending_op
            op.wait()
            self.close_profiler()


class CommHandler(object):
    """Communication handler. A dummy handler to wait aync operations.
    """

    def __init__(self, profiler: CommProfiler):
        super().__init__()
        self.prof = profiler

    def wait(self):
        self.prof.wait_async_op()


def async_check(profiler: CommProfiler):
    if profiler.pending_op is not None:
        profiler.warn_flag = True
        profiler.wait_async_op()


def all_reduce(tensor: torch.Tensor,
               op: ReduceOp = ReduceOp.SUM,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = 2 * (comm_size - 1) / comm_size
    comm_vol = correction * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_AllReduce_", comm_vol)
    profiler.pending_op = torch_all_reduce(tensor, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce_scatter(output: torch.Tensor,
                   input_list: List[torch.Tensor],
                   op: ReduceOp = ReduceOp.SUM,
                   group=None,
                   async_op: bool = False,
                   profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for tensor in input_list:
        comm_vol += tensor.element_size() * tensor.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_ReduceScatter_", comm_vol)
    profiler.pending_op = torch_reduce_scatter(output, input_list, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def all_gather(tensor_list: List[torch.Tensor],
               tensor: torch.Tensor,
               group=None,
               async_op: bool = False,
               profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_size = dist.get_world_size(group)
    correction = (comm_size - 1) / comm_size
    comm_vol = 0
    for ten in tensor_list:
        comm_vol += ten.element_size() * ten.numel()
    comm_vol *= correction
    profiler.activate_profiler("ncclKernel_AllGather_", comm_vol)
    profiler.pending_op = torch_all_gather(tensor_list, tensor, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def broadcast(tensor: torch.Tensor,
              src: int,
              group=None,
              async_op: bool = False,
              profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    backend = get_backend(group)
    if get_backend(group) != "nccl":
        profiler.logger.warrning("Can't profile non-nccl backend communication op.",)
    profiler.activate_profiler("ncclKernel_Broadcast_", comm_vol, backend, async_op)
    profiler.pending_op = torch_broadcast(tensor, src, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)


def reduce(tensor: torch.Tensor,
           dst: int,
           op: ReduceOp = ReduceOp.SUM,
           group=None,
           async_op: bool = False,
           profiler: CommProfiler = None) -> Optional[CommHandler]:
    async_check(profiler)

    comm_vol = 1.0 * tensor.element_size() * tensor.numel()
    profiler.activate_profiler("ncclKernel_Reduce_", comm_vol)
    profiler.pending_op = torch_reduce(tensor, dst, op, group, async_op)

    if async_op:
        return CommHandler(profiler)

    profiler.close_profiler(group)
