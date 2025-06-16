import copy
import os
from typing import Dict, List, Optional, Union

import mmcv
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
        sample_2d: bool = False,
        min_pixels:int = 1,
        mixup:float=0.7, # see AutoAlignv2
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
        self.mixup = mixup  # mixing factor for pasting objects
        assert 0 < self.mixup <= 1, "mixup should be in range (0, 1]"

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
            
        self.min_pixels = min_pixels
        self.sample_2d = sample_2d

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
        return self.convert_samples(pruned_samples, data)

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

    def convert_samples(self, samples: list, data) -> Union[None, Dict]:
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

        # paste img mask
        if self.sample_2d:
            ret["img"] = self.paste_objects(
                samples,
                data,
            )

        return ret


    def paste_objects(self, samples: list, data):
        """Paste sampled objects to the image.
        TODO support blending of pasted objects (see seamlessClone in OpenCV, or gaussian blur)
        """
        img = data['img']
        # TODO remove this deepcopy when validated
        orig_img = copy.deepcopy(img)  # keep original image for debugging
        # append the gt 3d boxes to the sampled 3d boxes
        centers = [s['box3d_lidar'][:2] for s in samples] + data["gt_bboxes_3d"].numpy()[:, :2].tolist()
        centers = np.array(centers)
        distances = np.linalg.norm(centers, axis=1)

        obj_cutout = [
            # iterate over each view of the sample
            [mmcv.imread(os.path.join(self.data_root, s_i)) if s_i is not None else None for s_i in s["img_path"]]
            # iterate over each sample
            for s in samples]

        # get gt cutouts from the image based on gt masks
        gt_cutout = []
        for i, (pos, mask) in enumerate(zip(data["gt_mask_pos"], data["gt_masks"])):
            gt_cutout_views = []
            # iterate through each view of the gt cutout
            for view_i, pos_i in enumerate(pos):
                if pos_i is None:
                    gt_cutout_views.append(None)
                    continue
                (x1, y1, w, h) = pos_i
                if w == 0 or h == 0:
                    gt_cutout_views.append(None)
                    continue
                # extract the region from the original image
                roi = orig_img[view_i][y1:y1+h, x1:x1+w]
                # apply the binary mask to get only the object pixels
                binary_mask = (mask[view_i] > 0).astype(np.float32)
                cutout = roi * binary_mask
                gt_cutout_views.append(cutout)
            gt_cutout.append(gt_cutout_views)

        # append gt cutouts to the sampled cutouts
        obj_cutout.extend(gt_cutout)

        # append gt mask positions to the sampled mask positions
        mask_pos = [s['box2d_camera'] for s in samples] + data["gt_mask_pos"]

        # Sort indices by farthest to closest (descending order)
        sorted_distance_inds = np.argsort(distances)[::-1]
        sorted_cutout = [obj_cutout[i] for i in sorted_distance_inds]
        sorted_mask_pos = [mask_pos[i] for i in sorted_distance_inds]

        for cutout_i_views, pos_i in zip(sorted_cutout, sorted_mask_pos):
            # paste the mask to the image
            # iterate over each cam view
            for view_i, cutout_i in enumerate(cutout_i_views):
                if cutout_i is None:
                    continue
                assert pos_i is not None, "Mask position must be provided"
                # get the position of the mask
                x1, y1, w, h = pos_i[view_i]
                if w * h < self.min_pixels: # skip if the mask is too small
                    continue

                # create a binary mask where non-zero values are 1
                binary_mask = (cutout_i > 0).astype(np.float32)
                mixup_mask = binary_mask * self.mixup
                # get the region of interest in the image
                roi = img[view_i][y1:y1 + h, x1:x1 + w]
                # blend the mask with the image, ignoring zero values
                img[view_i][y1:y1+h, x1:x1+w] = \
                    (roi * (1 - mixup_mask) + cutout_i * mixup_mask).astype(roi.dtype)
        # compute distance of each box from ego. insert in reverse order
        # the bbox might exceed the img size because the img is different

        # # choose a blend option
        # if not self.blending_type:
        #     blending_op = 'none'

        # else:
        #     blending_choice = np.random.randint(len(self.blending_type))
        #     blending_op = self.blending_type[blending_choice]

        # if blending_op.find('poisson') != -1:
        #     # options: cv2.NORMAL_CLONE=1, or cv2.MONOCHROME_TRANSFER=3
        #     # cv2.MIXED_CLONE mixed the texture, thus is not used.
        #     if blending_op == 'poisson':
        #         mode = np.random.choice([1, 3], 1)[0]
        #     elif blending_op == 'poisson_normal':
        #         mode = cv2.NORMAL_CLONE
        #     elif blending_op == 'poisson_transfer':
        #         mode = cv2.MONOCHROME_TRANSFER
        #     else:
        #         raise NotImplementedError
        #     center = (int(x1 + w / 2), int(y1 + h / 2))
        #     img = cv2.seamlessClone(obj_img, img, obj_mask * 255, center, mode)
        # else:
        #     if blending_op == 'gaussian':
        #         obj_mask = cv2.GaussianBlur(
        #             obj_mask.astype(np.float32), (5, 5), 2)
        #     elif blending_op == 'box':
        #         obj_mask = cv2.blur(obj_mask.astype(np.float32), (3, 3))
        #     paste_mask = 1 - obj_mask
        #     img[y1:y1 + h,
        #         x1:x1 + w] = (img[y1:y1 + h, x1:x1 + w].astype(np.float32) *
        #                       paste_mask[..., None]).astype(np.uint8)
        #     img[y1:y1 + h, x1:x1 + w] += (obj_img.astype(np.float32) *
        #                                   obj_mask[..., None]).astype(np.uint8)

        return img


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
