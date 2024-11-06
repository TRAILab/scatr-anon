
import math
import pdb
import sys

import numpy as np
from mmdet3d.registry import DATA_SAMPLERS
from mmengine.dist import get_dist_info
from torch.utils.data.sampler import Sampler


class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def set_trace():
    ForkedPdb().set_trace(sys._getframe().f_back)


@DATA_SAMPLERS.register_module()
class DistributedSampler(Sampler):
    def __init__(
        self, dataset=None, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last: bool = False
    ):
        rank, num_replicas = get_dist_info()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        assert not self.shuffle
        self.seed = seed
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0
        
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        self.groups_num = len(self.group_sizes)
        self.group_idx_to_sample_idxs = {
            group_idx: np.where(self.flag == group_idx)[0].tolist()
            for group_idx in range(self.groups_num)
        }

    def __iter__(self):
        # if "data_infos" in dir(self.dataset):
        #     timestamps = [
        #         x["timestamp"] / 1e6 for x in self.dataset.data_infos
        #     ]
        #     vehicle_idx = [
        #         x["lidar_path"].split("/")[-1][:4]
        #         if "lidar_path" in x
        #         else None
        #         for x in self.dataset.data_infos
        #     ]
        # else:
        #     timestamps = [
        #         x["timestamp"] / 1e6
        #         for x in self.dataset.datasets[0].data_infos
        #     ] * len(self.dataset.datasets)
        #     vehicle_idx = [
        #         x["lidar_path"].split("/")[-1][:4]
        #         if "lidar_path" in x
        #         else None
        #         for x in self.dataset.datasets[0].data_infos
        #     ] * len(self.dataset.datasets)

        # sequence_splits = []
        # for i in range(len(timestamps)):
        #     if i == 0 or (  # first sample
        #         # timestamp difference is larger than 4s
        #         abs(timestamps[i] - timestamps[i - 1]) > 4
        #         or vehicle_idx[i] != vehicle_idx[i - 1]  # different vehicle
        #     ):
        #         # start a new sequence
        #         sequence_splits.append([i])
        #     else:  # same sequence
        #         sequence_splits[-1].append(i)

        indices = []
        perfix_sum = 0  # what index is the current sample
        split_length = len(self.dataset) // self.num_replicas
        for i in range(self.groups_num):
            if perfix_sum >= (self.rank + 1) * split_length:  # rank has enough samples?
                break
            elif perfix_sum >= self.rank * split_length:
                indices.extend(self.group_idx_to_sample_idxs[i])
            # perfix_sum += len(self.group_idx_to_sample_idxs[i])
            perfix_sum += self.group_sizes[i]

        self.num_samples = len(indices)
        return iter(indices)
