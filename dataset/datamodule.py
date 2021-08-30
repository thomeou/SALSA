import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from dataset.dataloader import SeldDataset
from utilities.transforms import ComposeTransformNp, CompositeCutout, RandomShiftUpDownNp, \
    ComposeMapTransform, TfmapRandomSwapChannelFoa, TfmapRandomSwapChannelMic, GccRandomSwapChannelMic


class SeldDataModule(pl.LightningDataModule):
    """
    DataModule that group train and validation data for SELD task loader under on hood.
    """
    def __init__(self, feature_db, split_meta_dir: str = 'meta/dcase2021/original/', train_batch_size: int = 32,
                 val_batch_size: int = 32, mode: str = 'crossval', inference_split: str = None,
                 feature_type: str = 'salsa', audio_format: str = None):
        super().__init__()
        self.feature_db = feature_db
        self.split_meta_dir = split_meta_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.inference_split = inference_split
        self.feature_type = feature_type
        assert audio_format in ['foa', 'mic'], 'audio format {} is not valid'.format(audio_format)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.test_batch_size = None
        self.lit_logger = logging.getLogger('lightning')
        self.lit_logger.info('Create DataModule using train val split at {}.'.format(split_meta_dir))
        if mode == 'crossval':
            self.train_split = 'train'
            self.val_split = 'val'
            self.test_split = 'test'
        elif mode == 'eval':
            self.train_split = 'dev'
            self.val_split = 'test'  # actually not used during eval
            self.test_split = 'test'  # actually not used during eval
        else:
            raise NotImplementedError('Mode {} is not implemented!'.format(mode))

        # Data augmentation
        if audio_format == 'foa':
            if self.feature_type == 'salsa':
                self.train_joint_transform = ComposeMapTransform([
                    TfmapRandomSwapChannelFoa(n_classes=feature_db.n_classes),
                    ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10),  # apply across all channels
                ])
            elif self.feature_type == 'linspeciv':
                self.train_joint_transform = ComposeMapTransform([
                    TfmapRandomSwapChannelFoa(n_classes=feature_db.n_classes),
                ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10),  # apply across all channels
                    CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len / 200,
                                    n_zero_channels=3),  # n_zero_channels: these last channels will be replaced with 0
                ])
            elif self.feature_type == 'melspeciv':
                self.train_joint_transform = ComposeMapTransform([
                    TfmapRandomSwapChannelFoa(n_classes=feature_db.n_classes),
                ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10),  # apply across all channels
                    CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len / 128,
                                    n_zero_channels=3),  # n_zero_channels: these last channels will be replaced with 0
                ])
            else:
                raise NotImplementedError('aug not implemented for {} {}'.format(audio_format, self.feature_type))
        elif audio_format == 'mic':
            if self.feature_type == 'salsa':
                self.train_joint_transform = ComposeMapTransform([
                    TfmapRandomSwapChannelMic(n_classes=feature_db.n_classes),
                ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10),  # apply across all channels
                    CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len / 200,
                                    n_zero_channels=4),  # n_zero_channels: these last channels will be replaced with 0
                ])
            elif self.feature_type == 'linspecgcc':
                self.train_joint_transform = ComposeMapTransform([
                    GccRandomSwapChannelMic(n_classes=feature_db.n_classes),
                ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10, n_last_channels=6),  # apply across all channels
                    CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len / 200,
                                    n_zero_channels=6),  # n_zero_channels: these last channels will be replaced with 0
                ])
            elif self.feature_type == 'melspecgcc':
                self.train_joint_transform = ComposeMapTransform([
                    GccRandomSwapChannelMic(n_classes=feature_db.n_classes),
                ])
                self.train_transform = ComposeTransformNp([
                    RandomShiftUpDownNp(freq_shift_range=10, n_last_channels=6),  # apply across all channels
                    CompositeCutout(image_aspect_ratio=self.feature_db.train_chunk_len / 128,
                                    n_zero_channels=6),  # n_zero_channels: these last channels will be replaced with 0
                ])

    def setup(self, stage: str = None):
        """
        :param stage: can be 'fit' (default for training by lightning), 'test' (default for testing by lightning).
                      'inference': custom setup for inference of any data split.
        """
        # Get train and val data during training
        if stage == 'fit':
            train_db = self.feature_db.get_split(split=self.train_split, split_meta_dir=self.split_meta_dir,
                                                 stage='fit')
            self.train_dataset = SeldDataset(db_data=train_db, joint_transform=self.train_joint_transform,
                                             transform=self.train_transform)
            val_db = self.feature_db.get_split(split=self.val_split, split_meta_dir=self.split_meta_dir,
                                               stage='inference')
            self.val_dataset = SeldDataset(db_data=val_db)
            self.val_batch_size = val_db['test_batch_size']
            self.lit_logger.info('In datamodule: val batch size = {}'.format(self.val_batch_size))
            self.lit_logger.info('train dataset size: {} - val dataset size: {}'.format(
                len(self.train_dataset), len(self.val_dataset)))
        elif stage == 'test':
            test_db = self.feature_db.get_split(split=self.test_split, split_meta_dir=self.split_meta_dir,
                                                stage='inference')
            self.test_dataset = SeldDataset(db_data=test_db)
            self.test_batch_size = test_db['test_batch_size']
            self.lit_logger.info('Setup test stage in data module')
            self.lit_logger.info('In datamodule: test batch size = {}'.format(self.test_batch_size))
        elif stage == 'inference':
            inference_db = self.feature_db.get_split(split=self.inference_split, split_meta_dir=self.split_meta_dir,
                                                     stage='inference')
            self.test_dataset = SeldDataset(db_data=inference_db)
            self.test_batch_size = inference_db['test_batch_size']
            self.lit_logger.info('Setup inference stage in datamodule')
            self.lit_logger.info('In datamodule: test batch size = {}'.format(self.test_batch_size))
        else:
            raise NotImplementedError('stage {} is not implemented for datamodule'.format(stage))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.train_batch_size,
                          shuffle=True,
                          pin_memory=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=4)