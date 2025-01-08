import os
from typing import Dict, List, Optional, Union

import mmengine
import numpy as np
from mmdet3d.datasets.transforms import data_augment_utils
from mmdet3d.datasets.transforms.dbsampler import BatchSampler, DataBaseSampler
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops
from mmengine.fileio import get_local_path


@TRANSFORMS.register_module()
class TrackDBSampler(DataBaseSampler):
    """
    db = {
        class1: {
            track1: [info1, info2, ...],
            track2: [info1, info2, ...],
            ...
        },
        class2: {
            track1: [info1, info2, ...],
            track2: [info1, info2, ...],
            ...
        },
        ...
    }

    Unlike the original DBSampler, this sampler will sample a track at a time,
    instead of a single sample at a time.
    """

    def __init__(
        self,
        info_path: str,
        data_root: str,
        rate: float,
        prepare: dict,
        sample_groups: dict,
        classes: List[str],
        points_loader: dict = dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=5,
            use_dim=5,
            backend_args=None,
        ),
        backend_args: Optional[dict] = None,
    ) -> None:
        super(DataBaseSampler).__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.points_loader = TRANSFORMS.build(points_loader)
        self.backend_args = backend_args

        # load data base infos
        with get_local_path(info_path, backend_args=self.backend_args) as local_path:
            # loading data from a file-like object needs file format
            db_infos = mmengine.load(open(local_path, "rb"), file_format="pkl")

        # filter database infos
        from mmengine.logging import MMLogger

        logger: MMLogger = MMLogger.get_current_instance()
        for class_name, track_dict in db_infos.items():
            logger.info(
                f"load {len(track_dict)} {class_name} database tracks in TrackDBSampler"
            )
            total_instances = sum([len(track)
                                  for track in track_dict.values()])
            logger.info(
                f"load {total_instances} {class_name} instances in TrackDBSampler"
            )
            logger.info(
                f"longest tracks: {max([len(track) for track in track_dict.values()])} \
                shortest tracks: {min([len(track) for track in track_dict.values()])}"
            )
        # filter based on self.classes
        db_infos = {
            class_name: track_dict
            for class_name, track_dict in db_infos.items()
            if class_name in self.classes
        }
        # filter based on prepare
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info("After filter database:")
        for class_name, track_dict in db_infos.items():
            logger.info(
                f"load {len(track_dict)} {class_name} database tracks in TrackDBSampler"
            )
            # sum valid instances
            total_instances = sum(
                [sum([det["valid"] for det in track])
                 for track in track_dict.values()]
            )
            logger.info(
                f"load {total_instances} {class_name} instances in TrackDBSampler"
            )
        self.db_infos = db_infos

        # load sample groups
        self.sample_groups = sample_groups

        self.sampler_dict = {}
        self.group_db_infos = self.db_infos  # just use db_infos
        for class_name, track_dict in self.group_db_infos.items():
            self.sampler_dict[class_name] = TrackBatchSampler(
                track_dict, class_name)

    def get_samples(self, cls_distr, scene_token: str):
        """Get samples for a clip.
        Args:
            num_frames (int): Number of frames to sample.
        Returns:
            list[dict]: Sampled data dicts, corresponding to each frame.
                Each dict contains the following keys:
                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        clip_len = len(cls_distr)
        all_samples = [[] for _ in range(clip_len)]
        class_2_sampled_num = {}
        avg_cls_distr = np.mean(cls_distr, axis=0)
        assert len(avg_cls_distr) == len(
            self.classes), "avg_cls_distr should have the same length as classes"
        for class_name, max_sample_num in self.sample_groups.items():
            class_label = self.cat2label[class_name]
            sampled_num = max_sample_num - avg_cls_distr[class_label]
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            if sampled_num > 0:  # skip if no samples
                class_2_sampled_num[class_name] = sampled_num
        # order the sampled classes by the number of samples in descending order
        sampled_classes = sorted(
            class_2_sampled_num.keys(),
            key=lambda x: class_2_sampled_num[x],
            reverse=True,
        )
        for class_name in sampled_classes:
            sampled_num = class_2_sampled_num[class_name]  # nonzero guaranteed
            sample_tracks = self.sampler_dict[class_name].sample(
                sampled_num, clip_len, scene_token)
            for frame_idx, sample_tracks_i in enumerate(sample_tracks):
                all_samples[frame_idx] += sample_tracks_i
        return all_samples

    @staticmethod
    def filter_by_difficulty(db_infos: dict, removed_difficulty: list) -> dict:
        """Flag ground truths by difficulties.

        Do not filter for track db sampler to keep the track complete and
        have the correct spacing between valid samples. If the infos are filtered,
        the tracks will not be temporally consistent.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after flagging.
        """
        for class_name, track_dict in db_infos.items():
            for instance_id, infos in track_dict.items():
                for infos_idx, infos_i in enumerate(infos):
                    db_infos[class_name][instance_id][infos_idx]["valid"] = \
                        db_infos[class_name][instance_id][infos_idx]["valid"] and \
                        (infos_i["difficulty"] not in removed_difficulty)
        return db_infos

    @staticmethod
    def filter_by_min_points(db_infos: dict, min_gt_points_dict: dict) -> dict:
        """Flag ground truths by number of points in the bbox.
        Do not filter for track db sampler to keep the track complete and
        have the correct spacing between valid samples. If the infos are filtered,
        the tracks will not be temporally consistent.
        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after flagging.
        """
        for class_name, track_dict in db_infos.items():
            min_num = int(min_gt_points_dict[class_name])
            if min_num <= 0:
                continue
            for instance_id, infos in track_dict.items():
                for infos_idx, infos_i in enumerate(infos):
                    db_infos[class_name][instance_id][infos_idx]["valid"] = \
                        db_infos[class_name][instance_id][infos_idx]["valid"] and \
                        (infos_i["num_points_in_gt"] >= min_num)
        return db_infos

    def sample_all(self, data: dict, sample_tracks: list) -> Union[None, Dict]:
        """Sampling all categories of bboxes.
        Takes a list of data_queue, each element is a dict of data.
        Sample tracks based on the ground truths in the first frame.
        If there are no sampled tracks, return None in the list.

        Returns:
            list[dict]: Sampled data dicts, corresponding to each element in data_queue.
                Each dict contains the following keys:
                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths

        """

        # prune sampled tracks that collide with gt_bboxes
        pruned_samples = self.prune_collisions(
            data["gt_bboxes_3d"].numpy(),
            data["instance_inds"],
            sample_tracks,
        )

        # convert list of samples into dict
        return self.convert_samples(pruned_samples)

    def prune_collisions(
        self, gt_bboxes_3d: np.ndarray, gt_instance_inds: np.ndarray, sampled: List[Dict]
    ) -> List[Dict]:
        """Prune the sampled bboxes that collide with gt_bboxes.
        Args:
            gt_bboxes_3d (np.ndarray): Ground truth bboxes.
            sampled (list[dict]): Sampled bboxes.
        Returns:
            list[dict]: Pruned sampled bboxes.
        """
        if len(sampled) == 0:
            return sampled
        # convert gt_bboxes_3d to corner_box2d
        num_gt_bboxes = gt_bboxes_3d.shape[0]
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes_3d[:, 0:2], gt_bboxes_3d[:, 3:5], gt_bboxes_3d[:, 6]
        )

        # convert sampled to corner_box2d
        sampled_bboxes_3d = np.stack(
            [s["box3d_lidar"] for s in sampled], axis=0)
        sampled_bboxes_bv = box_np_ops.center_to_corner_box2d(
            sampled_bboxes_3d[:, 0:2],
            sampled_bboxes_3d[:, 3:5],
            sampled_bboxes_3d[:, 6],
        )

        all_bboxes_bv = np.concatenate(
            [gt_bboxes_bv, sampled_bboxes_bv], axis=0)

        # create collision matrix, where rows are sampled_bboxes_bv and columns are all_bboxes_bv
        coll_mat = data_augment_utils.box_collision_test(
            sampled_bboxes_bv, all_bboxes_bv
        )
        # set collisions on identical boxes to False
        diag = np.arange(len(sampled))
        # coll_mat[:, -len(sampled_bboxes_bv):] = False
        coll_mat[diag, diag + num_gt_bboxes] = False

        # set collisions between identical IDs to be true
        for i, s in enumerate(sampled):
            if s["instance_ind"] in gt_instance_inds:
                # get index of the colliding gt bbox
                matched_idx = np.where(
                    gt_instance_inds == s["instance_ind"])[0]
                coll_mat[i, matched_idx] = True

        # select valid samples based on collisions
        valid_samples = []
        for i, s in enumerate(sampled):
            if not coll_mat[i].any() and s["valid"]:  # if no collisions
                valid_samples.append(s)
            else:
                # do not append and set possible collisions to False
                coll_mat[:, i + num_gt_bboxes] = False
        return valid_samples

    def convert_samples(self, samples: list) -> Union[None, Dict]:
        if len(samples) == 0:
            return None
        ret = {}
        ret["gt_bboxes_3d"] = np.stack(
            [s["box3d_lidar"] for s in samples], axis=0)
        ret["gt_labels_3d"] = [self.cat2label[s["name"]] for s in samples]
        ret["instance_inds"] = [s["instance_ind"] for s in samples]
        
        if 'forecasting_locs' in samples[0]:
            ret["gt_forecasting_locs"] = np.stack(
                [s["forecasting_locs"] for s in samples], axis=0
            )
            ret["gt_forecasting_masks"] = np.stack(
                [s["forecasting_masks"] for s in samples], axis=0
            )
            ret["gt_forecasting_types"] = np.stack(
                [s["forecasting_types"] for s in samples], axis=0
            )
        # load points
        sampled_points = []
        for s in samples:
            file_path = (
                os.path.join(self.data_root,
                             s["path"]) if self.data_root else s["path"]
            )
            results = dict(lidar_points=dict(lidar_path=file_path))
            s_points = self.points_loader(results)["points"]
            # translate points to corresponding centers
            s_points.translate(s["box3d_lidar"][:3])
            sampled_points.append(s_points)
        ret["points"] = sampled_points[0].cat(sampled_points)

        return ret


class TrackBatchSampler(BatchSampler):
    def __init__(self, track_dict: Dict[str, List], name: str, shuffle: bool = True, rand_crop: bool = True) -> None:
        self.track_dict = track_dict
        self.track_ids = list(track_dict.keys())
        self.num_tracks = len(self.track_ids)
        self._idx = 0
        self.shuffle = shuffle
        self.indices = np.arange(self.num_tracks)
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.rand_crop = rand_crop  # cut the tracks to a random length
        self.name = name

    def sample(self, num_sample_tracks: int, num_frames: int, scene_token: str) -> List[List[dict]]:
        """
        Sample a set of <num_tracks> tracks across <num_frames> frames
        Return a list of length <num_frames>, where each element is a list of dicts with most
        num_tracks entries.
        TODO pull the shuffling out of the loop since it doesn't have to be run every time. See BatchSampler code.
        This ^ should speed up the sampling process, reducing the number of calls to np.random.shuffle
        """
        num_sample_tracks = min(num_sample_tracks, self.num_tracks)
        # sample tracks based on probabilities without replacement
        # sampled_track_ids = np.random.choice(
        #     self.track_ids, num_sample_tracks, replace=False, p=self.track_probabilities
        # )
        out = [[] for _ in range(num_frames)]
        # keep track of the sampled track ids to avoid double sampling the same track
        potential_tracks = [
            track_id for track_id in self.track_ids if track_id != scene_token]
        np.random.shuffle(potential_tracks)
        for frame_idx in range(num_frames):
            num_sample_tracks_rem = max(
                num_sample_tracks - len(out[frame_idx]), 0)
            # check if we need to sample more frames
            for sample_idx in range(num_sample_tracks_rem):
                if len(potential_tracks) == 0:  # no more potential tracks to insert
                    break
                sampled_track_id = potential_tracks.pop()
                sampled_track = self.track_dict[sampled_track_id]
                # check the length of the sampled track
                if self.rand_crop and len(sampled_track) > 1:
                    sampled_track_len = np.random.randint(
                        1, len(sampled_track))
                    sampled_track_len = min(
                        sampled_track_len, num_frames - frame_idx)
                    sampled_track_start = np.random.randint(
                        0, len(sampled_track) - sampled_track_len)
                    sampled_track = sampled_track[sampled_track_start:
                                                  sampled_track_start + sampled_track_len]
                # add sampled track to corresponding frames
                for j, (sampled_track_inst) in enumerate(sampled_track):
                    out[frame_idx + j].append(sampled_track_inst)
            if len(potential_tracks) == 0:  # no more potential tracks to insert
                break

        return out
