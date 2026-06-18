import torch

from .device_utils import get_free_memory_bytes


class LayeredProfiler:
    def __init__(self, print_memory=False, device: str = "cuda"):
        self.profiling_time_dict = {}
        self.print_memory = print_memory
        self.device = device
        self.min_free_mem = 1024 * 1024 * 1024 * 1024

    def add_profiling_time(self, item, time):

        if item not in self.profiling_time_dict:
            self.profiling_time_dict[item] = []

        self.profiling_time_dict[item].append(time)

        if self.print_memory:
            free_mem = get_free_memory_bytes(self.device)
            if free_mem >= 0:
                self.min_free_mem = min(self.min_free_mem, free_mem)
                print(f"free vmem @{item}: {free_mem/1024/1024/1024:.02f}GB, min free: {self.min_free_mem/1024/1024/1024:.02f}GB")
            else:
                print(f"free vmem @{item}: n/a (device '{self.device}' does not expose memory info)")

    def clear_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            self.profiling_time_dict[item] = []

    def print_profiling_time(self):
        for item in self.profiling_time_dict.keys():
            print(f"total time for {item}: {sum(self.profiling_time_dict[item])}")

