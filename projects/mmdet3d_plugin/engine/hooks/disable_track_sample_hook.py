from mmdet3d.engine.hooks import DisableObjectSampleHook
from mmdet3d.registry import HOOKS
from mmengine.dataset import BaseDataset
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from projects.mmdet3d_plugin.datasets.transforms.track_transforms_3d import TrackSample, GLOBAL_DISABLE_TRACK_SAMPLE


@HOOKS.register_module()
class DisableTrackSampleHook(DisableObjectSampleHook):
    def __init__(self, disable_after_epoch:int = None, disable_after_iter:int = None):
        super().__init__(disable_after_epoch)
        self.disable_after_iter = disable_after_iter

    def before_train_epoch(self, runner: Runner):
        """Close augmentation.

        Args:
            runner (Runner): The runner.
        """
        if self.disable_after_epoch is None:
            return
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if epoch == self.disable_after_epoch:
            runner.logger.info('Disable ObjectSample')
            dataset = runner.train_dataloader.dataset
            # handle dataset wrapper
            if not isinstance(dataset, BaseDataset):
                dataset = dataset.dataset
            for transform in dataset.pipeline.transforms:  # noqa: E501
                if isinstance(transform, TrackSample):
                    assert hasattr(transform, 'disabled')
                    GLOBAL_DISABLE_TRACK_SAMPLE.value = True
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

    def before_train_iter(self, runner, batch_idx, data_batch = None):
        """Close augmentation.

        Args:
            runner (Runner): The runner.
        """
        if self.disable_after_iter is None:
            return
        iter = runner.iter
        train_loader = runner.train_dataloader
        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module
        if iter == self.disable_after_iter:
            runner.logger.info('Disable ObjectSample')
            dataset = runner.train_dataloader.dataset
            # handle dataset wrapper
            if not isinstance(dataset, BaseDataset):
                dataset = dataset.dataset
            for transform in dataset.pipeline.transforms:  # noqa: E501
                if isinstance(transform, TrackSample):
                    assert hasattr(transform, 'disabled')
                    GLOBAL_DISABLE_TRACK_SAMPLE.value = True
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True