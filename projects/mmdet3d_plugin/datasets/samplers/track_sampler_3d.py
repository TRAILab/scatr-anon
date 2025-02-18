import copy
import math
import random
from typing import Optional

import numpy as np
from mmdet3d.registry import DATA_SAMPLERS
from mmdet.datasets.samplers import TrackImgSampler
from mmengine.dist import get_dist_info, sync_random_seed

from projects.mmdet3d_plugin.datasets import NuScenesTrackingDataset


@DATA_SAMPLERS.register_module()
class TrackSampler3D(TrackImgSampler):
    def __init__(
        self,
        sampler,
        batch_size: int,
        drop_last: bool = False,
        seed: Optional[int] = None,
        shuffle: bool = True,
        clip_len: Optional[int] = None,
        num_splits: Optional[int] = None,
        use_CBGS: bool = False,
        seq_flip_prob:float = 0.1,
    ) -> None:
        self.sampler = sampler
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        if seed is None:
            self.seed = sync_random_seed()
        else:
            self.seed = seed

        self.batch_size = batch_size
        self.global_batch_size = self.batch_size * self.world_size
        self.drop_last = drop_last
        self.dataset = sampler.dataset
        self.shuffle = shuffle
        self.seq_flip_prob = seq_flip_prob

        # Hard code here to handle different dataset wrapper
        assert isinstance(
            self.dataset, NuScenesTrackingDataset
        ), f"TrackImgSampler is only supported in NuScenesTrackingDataset but got {type(self.dataset)} "
        # TODO support CBGS wrapper
        self.test_mode = self.dataset.test_mode
        if self.test_mode and self.shuffle:
            raise ValueError(
                "Test mode is not compatible with shuffle=True in TrackSampler3D")
        # if self.test_mode and num_splits != 1:
        #     raise ValueError(
        #         "Test mode is not compatible with num_splits != 1 in TrackSampler3D")
        self.scene_tokens = self.dataset.get_all_scene_tokens()
        self.scene_indices = [self.dataset.get_scene_token_indices(
            scene_token) for scene_token in self.scene_tokens]
        self.group_indices = []

        assert (clip_len is not None) or (num_splits is not None), (
            "Either clip_len or num_splits must be specified.")
        
        assert (clip_len is None) or (num_splits is None), (
            "Only one of clip_len or num_splits can be specified.")
        
        if num_splits is not None:
            assert num_splits > 0, f"num_splits must be greater than 0, but got {num_splits}"
            for scene_indices in self.scene_indices:
                # split the sequence into num_splits parts
                self.group_indices.extend(np.array_split(
                    scene_indices, num_splits))
            self.group_indices = [x.tolist() for x in self.group_indices]
        elif clip_len == 1:  # split the scenes into individual frames
            for scene_indices in self.scene_indices:
                self.group_indices.extend(np.array_split(
                    scene_indices, len(scene_indices)))
            self.group_indices = [x.tolist() for x in self.group_indices]
        elif clip_len == -1:  # don't split the scenes
            self.group_indices = self.scene_indices
        else: # split the scenes into clips of length clip_len
            assert clip_len > 0, f"clip_len must be greater than -1, but got {clip_len}"
            # TODO support randomly skipping frames
            for scene_indices in self.scene_indices:
                # self.group_indices.extend(
                #     np.array_split(scene_indices, 10))
                self.group_indices.extend(
                    [scene_indices[i:i + clip_len]
                        for i in range(len(scene_indices)-clip_len)]
                )
            # self.group_indices = [x.tolist() for x in self.group_indices]
        self.classes = self.dataset.metainfo['classes']
        if use_CBGS:
            self.group_indices = self.get_CBGS_sample_indices(
                self.group_indices)
        self.num_groups = len(self.group_indices)
        assert self.num_groups >= self.global_batch_size, (
            f"only {self.num_groups} clips loaded but {self.world_size} gpus were given, each with a batch size of {self.batch_size}.")

    def get_CBGS_sample_indices(self, group_indices):
        class_sample_idxs = [[] for cat in self.classes]
        # iterate through each group
        for i, group_idxs_single in enumerate(group_indices):
            # get total categories present in the group
            cat_ids = set()
            for sample_idx in group_idxs_single:  # iterate through each sample in the group
                cat_ids = cat_ids.union(self.dataset.get_cat_ids(sample_idx))
            # note group idx if it contains a given category
            for cat_id in cat_ids:
                if cat_id != -1:
                    # Filter categories that do not need to be cared.
                    # -1 indicates dontcare in MMDet3D.
                    class_sample_idxs[cat_id].append(i)
        # calculate the distribution of each class in all groups
        duplicated_samples = sum([len(v) for v in class_sample_idxs])
        class_distribution = [
            max(1, len(v)) / duplicated_samples for v in class_sample_idxs]

        balanced_group_indices = []
        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution]
        rng = np.random.default_rng(self.seed)
        for cls_inds, ratio in zip(class_sample_idxs, ratios):
            while ratio > 1:
                balanced_group_indices += cls_inds
                ratio -= 1
            if ratio < 1 and ratio > 0:
                balanced_group_indices += rng.choice(
                    cls_inds, int(len(cls_inds) * ratio), replace=False).tolist()

        balanced_group_indices = [copy.deepcopy(group_indices[i])
                                  for i in balanced_group_indices]
        return balanced_group_indices

    def __iter__(self):
        if self.shuffle:  # shuffle the groups before dividing them between GPUs
            rng = random.Random(self.epoch + self.seed)
            groups_queue = rng.sample(
                self.group_indices, len(self.group_indices))
        else:
            groups_queue = self.group_indices
        # split groups between GPUs
        group_split_idxs = np.array_split(
            np.arange(len(groups_queue)), self.world_size)

        # split each GPU's groups into batches
        batched_group_split_idxs = [np.array_split(
            group_split_rank_idxs, self.batch_size) for group_split_rank_idxs in group_split_idxs]
        batched_group_split = [[[groups_queue[i] for i in batch_gs_idxs]  # each group
                                for batch_gs_idxs in batched_gs_idxs_rank]  # each batch
                               for batched_gs_idxs_rank in batched_group_split_idxs]  # each GPU
        # batched_group_split = [[groups_queue[batch_gs_ixs] for batch_gs_ixs in batched_gs_idxs_rank]
        #    for batched_gs_idxs_rank in batched_group_split_idxs]

        group_split_rank = batched_group_split[self.rank]
        group_split_lengths = [sum(len(group) for group in batch)
                               for batched_gs_idxs_rank in batched_group_split for batch in batched_gs_idxs_rank]
        # set every group split to be the same length
        if self.drop_last:  # crop to shortest batch
            min_group_len = min(group_split_lengths)
            for batch_idx, gs_rank_batch in enumerate(group_split_rank):
                while sum(len(group) for group in gs_rank_batch) > min_group_len:
                    if len(gs_rank_batch[-1]) > 1:
                        gs_rank_batch[-1].pop(-1)
                    else:
                        gs_rank_batch.pop(-1)
            padding_array = [[False] * sum(len(group) for group in group_batch)
                             for group_batch in group_split_rank]
        else:  # pad to longest batch
            max_group_len = max(group_split_lengths)
            padding_array = [[False] * sum(len(group) for group in group_batch)
                             for group_batch in group_split_rank]
            for batch_idx, gs_rank_batch in enumerate(group_split_rank):
                pad_idx = 0
                while sum(len(group) for group in gs_rank_batch) < max_group_len:
                    padding_group = gs_rank_batch[pad_idx]
                    # case 1: append the entire padding group
                    # case 2: append subset up to max_group_len
                    amount_padded = min(max_group_len - sum(len(group)
                                        for group in gs_rank_batch), len(padding_group))
                    gs_rank_batch.append(padding_group[:amount_padded])
                    padding_array[batch_idx].extend([True] * amount_padded)
                    pad_idx = (pad_idx + 1) % len(gs_rank_batch)
        assert all(sum(len(group) for group in group_split_batch) == sum(len(group)
                   for group in group_split_rank[0]) for group_split_batch in group_split_rank)
        assert all(sum(len(group) for group in group_split_batch) == len(padding_array_i)
                   for group_split_batch, padding_array_i in zip(group_split_rank, padding_array))
        active_groups = [[] for _ in range(self.batch_size)]
        while any(len(group_batch) > 0 for group_batch in group_split_rank) or any(len(group) > 0 for group in active_groups):
            # iterate until no more groups in group_split_rank and all active groups are empty
            curr_batch = []
            # construct batch
            for batch_idx in range(self.batch_size):
                # refill with next group when empty
                if len(active_groups[batch_idx]) == 0:
                    next_group = group_split_rank[batch_idx].pop(0)
                    if np.random.uniform() < self.seq_flip_prob:
                        # flip the sequence
                        next_group = next_group[::-1]
                    group_aug = self.dataset.get_augmentation(next_group)
                    active_groups[batch_idx] = [
                        {
                            "index": index,
                            "aug_config": group_aug_i,
                            "new_scene": i == 0,
                            "padding": padding_array[batch_idx].pop(0),
                        } for i, (index, group_aug_i) in enumerate(zip(next_group, group_aug))
                    ]
                # pop from active group
                curr_batch.append(active_groups[batch_idx].pop(0))
            if len(curr_batch) == self.batch_size:
                yield curr_batch

    def __len__(self) -> int:
        return math.ceil(sum(len(group) for group in self.group_indices) / self.batch_size / self.world_size)
