"""
This module load all seld feature into memory.
Reference code:  https://github.com/yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
Note: there are two frame rates:
    1. feature frame rate: 80 frames/s
    2. label frame rate: 10 frames/s
"""
import logging
import os
from typing import List

import h5py
import numpy as np
import pandas as pd


class Database:
    """
    Database class to handle one input streams for SELD
    """
    def __init__(self,
                 feature_root_dir: str = '/data/seld_dcase2021/features/tfmap/foa/'
                                         '24000fs_512nfft_300nhop_5cond_9000fmaxdoa_Falseonset_Falsemask_Truedrr',
                 gt_meta_root_dir: str = '/media/tho_nguyen/disk1/audio_datasets/dcase2021/task3',
                 audio_format: str = 'foa', n_classes: int = 12, fs: int = 24000,
                 n_fft: int = 1024, hop_len: int = 300, label_rate: float = 10, train_chunk_len_s: float = 8.0,
                 train_chunk_hop_len_s: float = 0.5, test_chunk_len_s: float = 4.0, test_chunk_hop_len_s: float = 2.0,
                 output_format: str = 'reg_xyz'):
        """
        :param feature_root_dir: Feature directory. can be SED or DOA feature.
        The data are organized in the following format:
            |__feature_root_dir/
                |__foa_dev/
                |__foa_eval/
                |__mic_dev/
                |__mic_eval/
                |__foa_feature_scaler.h5
                |__mic_feature_scaler.h5
        :param gt_meta_root_dir: Directory that contains groundtruth meta data.
        The data are orgamized in the following format:
            |__gt_meta_dir/
                |__/metadata_dev/
                |__/metadata_eval/
                |__metadata_eval_info.csv
        """
        self.feature_root_dir = feature_root_dir
        self.gt_meta_root_dir = gt_meta_root_dir
        self.audio_format = audio_format
        self.n_classes = n_classes
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.label_rate = label_rate
        self.train_chunk_len = self.second2frame(train_chunk_len_s)
        self.train_chunk_hop_len = self.second2frame(train_chunk_hop_len_s)
        self.test_chunk_len = self.second2frame(test_chunk_len_s)
        self.test_chunk_hop_len = self.second2frame(test_chunk_hop_len_s)
        self.output_format = output_format  # reg_xyz, accdoa
        self.max_nframes_per_file = int(60 * self.label_rate)  # hardcode the filelen of 60 seconds

        assert audio_format in ['foa', 'mic'], 'Incorrect value for audio format {}'.format(audio_format)

        assert os.path.isdir(os.path.join(self.feature_root_dir, self.audio_format + '_dev')), \
            '"dev" folder is not found'

        self.chunk_len = None
        self.chunk_hop_len = None
        self.feature_rate = self.fs / self.hop_len  # Frame rate per second
        self.label_upsample_ratio = int(self.feature_rate / self.label_rate)
        self.feature_mean, self.feature_std = self.load_feature_scaler()

        logger = logging.getLogger('lightning')
        logger.info('Load feature from {}.'.format(self.feature_root_dir))
        logger.info('train_chunk_len = {}, train_chunk_hop_len = {}'.format(
            self.train_chunk_len, self.train_chunk_hop_len))
        logger.info('test_chunk_len = {}, test_chunk_hop_len = {}'.format(
            self.test_chunk_len, self.test_chunk_hop_len))

    def second2frame(self, second: float = None) -> int:
        """
        Convert seconds to frame unit.
        """
        sample = int(second * self.fs)
        frame = int(round(sample/self.hop_len))
        return frame

    def load_feature_scaler(self):
        """
        Load feature scaler for multichannel spectrograms
        """
        scaler_fn = os.path.join(self.feature_root_dir, self.audio_format + '_feature_scaler.h5')
        with h5py.File(scaler_fn, 'r') as hf:
            mean = hf['mean'][:]
            std = hf['std'][:]

        return mean, std

    def get_segment_idxes(self, n_frames: int, downsample_ratio: int, pointer: int):
        """
        This function returns the segment indices when n_crop_frames are divided into segments.
        The segment can be features or ground truth.
        :param n_frames: Number of frame (using feature rate)
        :param downsample_ratio: downsample_ratio = feature_rate/segment_rate.
        :param pointer: the pointer that point to the last segment index. It will be updated after this file is divided
            into segments.
        """
        # Get number of frame using segment rate
        assert n_frames % downsample_ratio == 0, 'n_features_frames is not divisible by downsample ratio'
        n_crop_frames = n_frames // downsample_ratio
        assert self.chunk_len // downsample_ratio <= n_crop_frames, 'Number of cropped frame is less than chunk len'
        idxes = np.arange(pointer,
                          pointer + n_crop_frames - self.chunk_len // downsample_ratio + 1,
                          self.chunk_hop_len // downsample_ratio).tolist()
        # Include the leftover of the cropped data
        if (n_crop_frames - self.chunk_len // downsample_ratio) % self.chunk_hop_len // downsample_ratio != 0:
            idxes.append(pointer + n_crop_frames - self.chunk_len // downsample_ratio)
        pointer += n_crop_frames

        return idxes, pointer

    def get_split(self, split: str, split_meta_dir: str = '/meta/dcase2021/original', stage: str = 'fit'):
        """
        Function to load all data of a split into memory, divide long audio clip/file into smaller chunks, and assign
        labels for clips and chunks.
        :param split: Split of data, choices:
            'train', 'val', 'test', 'eval': load chunk of data
        :param split_meta_dir: Directory where meta of split is stored.
        :param stage:
            'fit' for training and testing,
            'inteference': for inteference. this param decide chunk_len
        :return:
        """
        # Get feature dir, filename list, and gt_meta_dir
        if split == 'eval':
            split_sed_feature_dir = os.path.join(self.feature_root_dir, self.audio_format + '_eval')
            csv_filename = os.path.join(os.path.split(split_meta_dir)[0], 'eval.csv')
            gt_meta_dir = os.path.join(self.gt_meta_root_dir, 'metadata_eval')
        else:
            split_sed_feature_dir = os.path.join(self.feature_root_dir, self.audio_format + '_dev')
            csv_filename = os.path.join(split_meta_dir, split + '.csv')
            gt_meta_dir = os.path.join(self.gt_meta_root_dir, 'metadata_dev')
        meta_df = pd.read_csv(csv_filename)
        split_filenames = meta_df['filename'].tolist()
        logger = logging.getLogger('lightning')
        logger.info('Number of files in split {} is {}'.format(split, len(split_filenames)))
        # Get chunk len and chunk hop len
        if stage == 'fit':
            self.chunk_len = self.train_chunk_len
            self.chunk_hop_len = self.train_chunk_hop_len
        elif stage == 'inference':
            self.chunk_len = self.test_chunk_len
            self.chunk_hop_len = self.test_chunk_hop_len
        else:
            raise NotImplementedError('stage {} is implemented'.format(stage))

        # Load and crop data
        features, sed_targets, doa_targets, feature_chunk_idxes, gt_chunk_idxes, filename_list, test_batch_size = \
            self.load_chunk_data(split_filenames=split_filenames, split_sed_feature_dir=split_sed_feature_dir,
                                 gt_meta_dir=gt_meta_dir)
        # pack data
        db_data = {
            'features': features,
            'sed_targets': sed_targets,
            'doa_targets': doa_targets,
            'feature_chunk_idxes': feature_chunk_idxes,
            'gt_chunk_idxes': gt_chunk_idxes,
            'filename_list': filename_list,
            'test_batch_size': test_batch_size,
            'feature_chunk_len': self.chunk_len,
            'gt_chunk_len': self.chunk_len // self.label_upsample_ratio
        }

        return db_data

    def load_chunk_data(self, split_filenames: List, split_sed_feature_dir: str, gt_meta_dir: str):
        """
        Load feature, crop data and assign labels.
        :param split_filenames: List of filename in the split.
        :param split_sed_feature_dir: Feature directory of the split
        :param gt_meta_dir: Ground truth meta directory of the split.
        :return: features, targets, chunk_idxes, filename_list
        """
        feature_pointer = 0
        gt_pointer = 0
        features_list = []
        filename_list = []
        sed_targets_list = []
        doa_targets_list = []
        feature_idxes_list = []
        gt_idxes_list = []
        for filename in split_filenames:
            # Load features -> n_channels x n_frames x n_features
            feature_fn = os.path.join(split_sed_feature_dir, filename + '.h5')
            with h5py.File(feature_fn, 'r') as hf:
                feature = hf['feature'][:]  # (n_channels, n_frames, n_features)
                # Normalize feature
                n_scaler_chan = self.feature_mean.shape[0]
                # for SALSA feature, only normalize the first 4 channels
                if self.feature_mean.ndim > 1 and n_scaler_chan < feature.shape[0]:
                    feature[:n_scaler_chan] = (feature[:n_scaler_chan] - self.feature_mean) / self.feature_std
                else:
                    feature = (feature - self.feature_mean) / self.feature_std
            n_feature_frames = feature.shape[1]
            # feature frames: n_frames is upsampled back to the original feature frame rate
            n_frames = min(n_feature_frames, self.max_nframes_per_file * self.label_upsample_ratio)
            # trim feature. Make sure we have 4800 channels
            feature = feature[:, :n_frames, :]

            # Load gt info from metadata
            gt_meta_fn = os.path.join(gt_meta_dir, filename + '.csv')
            sed_target, doa_target = self.load_classwise_gt(gt_meta_fn, n_frames)

            # Get sed segment indices
            feature_idxes, feature_pointer = self.get_segment_idxes(
                n_frames=n_frames, downsample_ratio=1, pointer=feature_pointer)

            # Get gt segment indices
            gt_idxes, gt_pointer = self.get_segment_idxes(
                n_frames=n_frames, downsample_ratio=self.label_upsample_ratio, pointer=gt_pointer)

            assert len(feature_idxes) == len(gt_idxes), 'nchunks for sed and gt are different'

            # Append data
            features_list.append(feature)
            filename_list.extend([filename] * len(feature_idxes))
            sed_targets_list.append(sed_target)
            doa_targets_list.append(doa_target)
            feature_idxes_list.append(feature_idxes)
            gt_idxes_list.append(gt_idxes)

        if len(features_list) > 0:
            features = np.concatenate(features_list, axis=1)
            sed_targets = np.concatenate(sed_targets_list, axis=0)
            doa_targets = np.concatenate(doa_targets_list, axis=0)
            sed_chunk_idxes = np.concatenate(feature_idxes_list, axis=0)
            gt_chunk_idxes = np.concatenate(gt_idxes_list, axis=0)
            test_batch_size = len(feature_idxes)  # to load all chunks of the same file
            return features, sed_targets, doa_targets, sed_chunk_idxes, gt_chunk_idxes, filename_list, test_batch_size
        else:
            return None, None, None, None, None, None, None

    @staticmethod
    def sort_tracks(track_number):
        """Sort the track from shortest to longest"""
        n_tracks = np.max(track_number) + 1
        track_durations = np.zeros((n_tracks,), dtype=np.int32)
        for itrack in np.arange(n_tracks):
            track_idx = track_number == itrack
            track_durations[itrack] = np.sum(track_idx)
        sorted_tracks = np.argsort(track_durations)
        return sorted_tracks

    def load_classwise_gt(self, gt_meta_fn, n_frames):
        """
        Load classwise ground truth from csv file
        """
        # Get back to label frame rate: n_frames (follow feature rate) -> n_label_frames (follow label rate)
        assert n_frames % self.label_upsample_ratio == 0, 'mismatch ground truth and feature frame rate'
        n_label_frames = n_frames // self.label_upsample_ratio
        df = pd.read_csv(gt_meta_fn, header=None,
                         names=['frame_number', 'sound_class_idx', 'track_number', 'azimuth', 'elevation'])
        frame_number = df['frame_number'].values
        sound_class_idx = df['sound_class_idx'].values
        track_number = df['track_number'].values
        azimuth = df['azimuth'].values
        elevation = df['elevation'].values
        sorted_tracks = self.sort_tracks(track_number=track_number)
        # Generate target data
        if self.output_format in ['reg_xyz', 'accdoa']:
            sed_target = np.zeros((n_label_frames, self.n_classes), dtype=np.float32)
            azi_target = np.zeros((n_label_frames, self.n_classes), dtype=np.float32)
            ele_target = np.zeros((n_label_frames, self.n_classes), dtype=np.float32)
            for itrack in sorted_tracks:
                track_idx = track_number == itrack
                frame_number_1 = frame_number[track_idx]
                sound_class_idx_1 = sound_class_idx[track_idx]
                azimuth_1 = azimuth[track_idx]
                elevation_1 = elevation[track_idx]
                for idx, iframe in enumerate(frame_number_1):
                    class_idx = int(sound_class_idx_1[idx])
                    sed_target[iframe, class_idx] = 1.0
                    azi_target[iframe, class_idx] = azimuth_1[idx] * np.pi / 180.0  # Radian unit
                    ele_target[iframe, class_idx] = elevation_1[idx] * np.pi / 180.0  # Radian unit
            # Doa target
            x = np.cos(azi_target) * np.cos(ele_target)
            y = np.sin(azi_target) * np.cos(ele_target)
            z = np.sin(ele_target)
            # Those where event is inactive will have x=y=z=0
            x[sed_target < 1] = 0.0
            y[sed_target < 1] = 0.0
            z[sed_target < 1] = 0.0
            doa_target = np.concatenate((x, y, z), axis=-1)
        else:
            raise ValueError('doa output format {} is not valid'.format(self.output_format))

        return sed_target, doa_target


if __name__ == '__main__':
    tp_db = Database(feature_root_dir='/data/seld_dcase2021/features/tfmap/mic/'
                                      '24000fs_512nfft_300nhop_5cond_4000fmaxdoa',
                     output_format='reg_xyz', test_chunk_len_s=8, test_chunk_hop_len_s=0.5, audio_format='mic')
    # tp_db = SeldDatabase()
    db_data = tp_db.get_split(split='val', split_meta_dir='meta/dcase2021/original', stage='inference')
    print(db_data['features'].shape)
    print(db_data['sed_targets'].shape)
    print(db_data['doa_targets'].shape)
    print(len(db_data['feature_chunk_idxes']))
    print(len(db_data['gt_chunk_idxes']))
    print(len(db_data['filename_list']))