from mmdet3d.registry import HOOKS
from mmengine.hooks import Hook
from typing import Optional

@HOOKS.register_module()
class DisableGroupAugHook(Hook):
    """The hook of disabling group augmentations during training.

    Args:
        disable_after_epoch (int): The number of epochs after which
            the group augmentations will be closed in the training.
            Defaults to 15.
    """

    def __init__(self, disable_after_epoch: Optional[int] = None, disable_after_iter: Optional[int] = None):
        raise NotImplementedError(
            "This hook is not implemented yet. Breaks when used in distributed training. need to validate how to " \
            "disable group augmentations in distributed training properly."
        )
        self.disable_after_epoch = disable_after_epoch
        self.disable_after_iter = disable_after_iter
        assert disable_after_epoch is not None or disable_after_iter is not None, \
            'At least one of `disable_after_epoch` or `disable_after_iter` must be set.'
        if disable_after_epoch is not None:
            assert disable_after_epoch > 0, \
                f'`disable_after_epoch` must be greater than 0, but got {disable_after_epoch}.'
        if disable_after_iter is not None:
            assert disable_after_iter > 0, \
                f'`disable_after_iter` must be greater than 0, but got {disable_after_iter}.'

    def before_train_epoch(self, runner):
        """Close group augmentations."""
        if self.disable_after_epoch is None:
            return
        epoch = runner.epoch
        if epoch == self.disable_after_epoch:
            self.disable_group_aug(runner)

    def before_train_iter(self, runner, batch_idx: int, data_batch: dict | tuple | list | None = None) -> None:
        """Close group augmentations."""
        if self.disable_after_iter is None:
            return
        iter = runner.iter
        if iter == self.disable_after_iter:
            self.disable_group_aug(runner)

    def disable_group_aug(self, runner):
        """Close group augmentations."""
        runner.logger.info('Disable Group Augmentation')
        runner.model.pts_bbox_head.disable_group_aug()