import mmcv
import numpy as np
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TrackLoadAnnotations3D(LoadAnnotations3D):

    def __init__(self, with_forecasting: bool = False, num_cams: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.with_forecasting = with_forecasting
        self.num_cams = num_cams # default is 6 for nuScenes dataset

    def _load_track_ids(self, results):
        if not "instance_inds" in results["ann_info"]:
            assert len(results["ann_info"]["gt_labels_3d"]) == 0, f"{results['ann_info']}"
        results["instance_inds"] = results["ann_info"].get("instance_inds", np.array([]))
        return results

    def _load_forecasting(self, results):
        """Private function to load forecasting annotations"""
        results["gt_forecasting_locs"] = results["ann_info"].get(
            "forecasting_locs", np.zeros((0, 13, 3))
        )
        results["gt_forecasting_masks"] = results["ann_info"].get(
            "forecasting_masks", np.zeros((0, 13))
        )
        results["gt_forecasting_types"] = results["ann_info"].get(
            "forecasting_types", np.zeros((0, 13))
        )
        return results

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        results = self._load_track_ids(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_forecasting:
            results = self._load_forecasting(results)
        return results

    def _load_masks(self, results: dict) -> None:
        gt_masks = []
        gt_mask_pos = []
        # iterate through each instance
        for mask_info in results['ann_info']['mask_info']:
            # mask_info = ann_info.get('mask_info', None)
            # if mask_info is None or len(mask_info) == 0:
            #     gt_masks.append([None] * self.num_cams)
            #     gt_mask_pos.append([None] * self.num_cams)
            #     continue
            mask = [
                None if (x is None or 'mask_crop_path' not in x) 
                else mmcv.imread(x['mask_crop_path'])
                for x in mask_info
            ]
            gt_masks.append(mask)
            mask_pos = [
                None if (x is None) 
                else x.get('mask_crop_box', None)
                for x in mask_info]
            gt_mask_pos.append(mask_pos)
        results['gt_masks'] = gt_masks
        results['gt_mask_pos'] = gt_mask_pos

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = "    "
        repr_str = (
            super().__repr__()
            + f"{indent_str}with_forecasting={self.with_forecasting}, "
        )

        return repr_str
