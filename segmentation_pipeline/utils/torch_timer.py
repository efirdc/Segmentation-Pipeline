import time

import torch


class TorchTimer:
    def __init__(self, device):
        self.device = device

        self.start_time = 0
        self.last_time = 0
        self.timestamps = {}

    def start(self):
        self.start_time = self.last_time = time.time()
        self.timestamps = {stamp: 0.0 for stamp in self.timestamps.keys()}

    def stamp(self, name=None, from_start=False):
        if self.device.type != "cpu":
            torch.cuda.current_stream().synchronize()

        new_time = time.time()
        if not from_start:
            dt = new_time - self.last_time
        else:
            dt = time.time() - self.start_time
        self.last_time = new_time
        if name:
            self.timestamps[name] = dt
        return dt
