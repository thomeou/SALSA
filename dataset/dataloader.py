"""
Module for dataloader for SELD
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from dataset.database import Database


class SeldDataset(Dataset):
    """
    Chunk dataset for SELD task
    """
    def __init__(self, db_data, joint_transform=None, transform=None):
        super().__init__()
        self.features = db_data['features']
        self.sed_targets = db_data['sed_targets']
        self.doa_targets = db_data['doa_targets']
        self.chunk_idxes = db_data['feature_chunk_idxes']
        self.gt_chunk_idxes = db_data['gt_chunk_idxes']
        self.filename_list = db_data['filename_list']
        self.chunk_len = db_data['feature_chunk_len']
        self.gt_chunk_len = db_data['gt_chunk_len']
        self.joint_transform = joint_transform  # transform that change label
        self.transform = transform  # transform that does not change label
        self.n_samples = len(self.chunk_idxes)

    def __len__(self):
        """
        Total of training samples.
        """
        return self.n_samples

    def __getitem__(self, index):
        """
        Generate one sample of data
        """
        # Select sample
        chunk_idx = self.chunk_idxes[index]
        gt_chunk_idx = self.gt_chunk_idxes[index]

        # get filename
        filename = self.filename_list[index]

        # Load data and get label
        # (n_channels, n_timesteps, n_mels)
        X = self.features[:, chunk_idx: chunk_idx + self.chunk_len, :]
        # (n_timesteps, n_classes)
        sed_labels = self.sed_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len]
        # (n_timesteps, x*n_classes) or (n_timesteps, x*n_classes, 2)
        doa_labels = self.doa_targets[gt_chunk_idx:gt_chunk_idx + self.gt_chunk_len]

        if self.joint_transform is not None:
            X, sed_labels, doa_labels = self.joint_transform(X, sed_labels, doa_labels)

        if self.transform is not None:
            X = self.transform(X)

        return X, sed_labels, doa_labels, filename


if __name__ == '__main__':
    # test dataloader
    db = Database(feature_root_dir='/data/seld_dcase2021/features/tfmap/mic/24000fs_512nfft_300nhop_5cond_4000fmaxdoa',
                  audio_format='mic', output_format='reg_xyz')
    data_db = db.get_split(
        split='val', split_meta_dir='meta/dcase2021/original')

    # create train dataset
    dataset = SeldDataset(db_data=data_db)
    print('Number of samples: {}'.format(len(dataset)))

    # load one sample
    index = np.random.randint(len(dataset))
    sample = dataset[index]
    for item in sample[:-1]:
        print(item.shape)
    print(sample[-1])

    # test data generator
    batch_size = 8
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    print('Number of batches: {}'.format(len(dataloader)))  # steps_per_epoch
    for train_iter, (X, sed_labels, doa_labels, filenames) in enumerate(dataloader):
        if train_iter == 0:
            print('X: dtype: {} - shape: {}'.format(X.dtype, X.shape))
            print('sed_labels: dtype: {} - shape: {}'.format(sed_labels.dtype, sed_labels.shape))
            print('doa_labels: dtype: {} - shape: {}'.format(doa_labels.dtype, doa_labels.shape))
            print(type(filenames))
            print(filenames)
            break
