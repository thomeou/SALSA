"""
This module include interface for LightningModule for combined models.
"""
import logging
import os

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from metrics import dcase_utils
from metrics import SELD2020_evaluation_metrics, SELD2021_evaluation_metrics


class BaseModel(pl.LightningModule):
    def __init__(self, sed_threshold: float = 0.3, doa_threshold: int = 20, label_rate: int = 10,
                 feature_rate: int = None, optimizer_name: str = 'Adam', lr: float = 1e-3, output_pred_dir: str = None,
                 submission_dir: str = None, test_chunk_len: int = None, test_chunk_hop_len: int = None,
                 gt_meta_root_dir: str = None, output_format: str = None, sequence_format: str = None,
                 eval_version: str = '2021'):
        super().__init__()
        self.sed_threshold = sed_threshold
        self.doa_threshold = doa_threshold
        self.label_rate = label_rate
        self.feature_rate = feature_rate
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.submission_dir = submission_dir
        self.output_pred_dir = output_pred_dir
        self.test_chunk_len = test_chunk_len
        self.test_chunk_hop_len = test_chunk_hop_len
        self.gt_meta_root_dir = gt_meta_root_dir
        self.output_format = output_format
        self.sequence_format = sequence_format
        self.eval_version = eval_version
        self.lit_logger = logging.getLogger('lightning')
        self.max_nframes_per_file = int(60 * self.label_rate)  # hardcode the max number of frame per file (60 second)

        # Load gt meta for evaluation
        self.gt_labels = self.load_gt_meta()

        # Write submission files
        if self.eval_version == '2021':
            self.df_columns = ['frame_idx', 'event', 'track_number', 'azimuth', 'elevation']
            self.seld_eval_metrics = SELD2021_evaluation_metrics
        elif self.eval_version == '2020':
            self.df_columns = ['frame_idx', 'event', 'azimuth', 'elevation']
            self.seld_eval_metrics = SELD2020_evaluation_metrics
        else:
            raise ValueError('Unknown eval_version {}'.format(self.eval_version))

        # Write output prediction for test step
        if self.output_pred_dir is not None:
            os.makedirs(self.output_pred_dir, exist_ok=True)

        self.lit_logger.info('Initialize lightning model.')

    def forward(self, x):
        pass

    def common_step(self, batch_data):
        pass

    def training_step(self, train_batch, batch_idx):
        pass

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, val_batch, batch_idx):
        pass

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def test_step(self, test_batch, batch_idx):
        pass

    def test_epoch_end(self, test_step_outputs):
        pass

    def configure_optimizers(self):
        """
        Pytorch lightning hook
        """
        if self.optimizer_name in ['Adam', 'adam']:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name in ['AdamW', 'Adamw', 'adamw']:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise NotImplementedError('Optimizer {} is not implemented!'.format(self.optimizer_name))
        return optimizer

    def combine_chunks(self, frame_output_pred, combine_method: str = 'mean'):
        """
        Combine chunks of output prediction into file prediction:
        (n_chunks, chunk_len, n_classes) -> (file_len, n_classes).
        Some params are hardcoded for dcase2020 and dcase2021 dataset: file-len=60s.
        :param combine_method: choices: 'mean', 'gmean'
        """
        file_len = 60  # seconds
        n_frames = file_len * self.label_rate
        # Convert feature test_chunk_len and test_chunk_hop_len to label test_chunk_len and hop_len
        label_test_chunk_len = int(self.test_chunk_len * self.label_rate / self.feature_rate)
        label_test_chunk_hop_len = int(self.test_chunk_hop_len * self.label_rate / self.feature_rate)
        n_chunks = frame_output_pred.shape[0]
        if frame_output_pred.ndim == 3:
            file_pred = np.zeros((n_frames, frame_output_pred.shape[-1]), dtype=np.float32)
        elif frame_output_pred.ndim == 4:
            file_pred = np.zeros((n_frames, frame_output_pred.shape[-2], frame_output_pred.shape[-1]), dtype=np.float32)
        else:
            raise ValueError('unknown dim for frame_output_pred in combine_chunks')
        chunk_idxes = np.arange(0, n_frames - label_test_chunk_len + 1, label_test_chunk_hop_len).tolist()
        # Include the leftover of the cropped data
        if (n_frames - label_test_chunk_len) % label_test_chunk_hop_len != 0:
            chunk_idxes.append(n_frames - label_test_chunk_len)
        chunk_overlap = label_test_chunk_len - label_test_chunk_hop_len
        assert abs(n_chunks - len(chunk_idxes)) < 2
        for chunk_idx in np.arange(len(chunk_idxes)):
            start_idx = chunk_idxes[chunk_idx]
            end_idx = start_idx + label_test_chunk_len
            if chunk_idx == 0:
                file_pred[start_idx: end_idx] = frame_output_pred[chunk_idx]
            else:
                if combine_method == 'mean':
                    file_pred[start_idx: start_idx + chunk_overlap] = (
                        file_pred[start_idx: start_idx + chunk_overlap] +
                        frame_output_pred[chunk_idx, 0: chunk_overlap]) / 2
                elif combine_method == 'gmean':
                    file_pred[start_idx: start_idx + chunk_overlap] = np.sqrt(
                        file_pred[start_idx: start_idx + chunk_overlap] *
                        frame_output_pred[chunk_idx, 0: chunk_overlap])
                else:
                    raise ValueError('combine method {} is unknown'.format(combine_method))
                file_pred[start_idx + chunk_overlap: end_idx] = frame_output_pred[chunk_idx, chunk_overlap:]
        return file_pred

    def load_gt_meta(self):
        """
        Funtion to load gt meta in polar format
        """
        gt_meta_dir = {'dev': os.path.join(self.gt_meta_root_dir, 'metadata_dev'),
                       'eval': os.path.join(self.gt_meta_root_dir, 'metadata_eval')}
        fn_list = {'dev': sorted(os.listdir(gt_meta_dir['dev'])),
                   'eval': []}
        if os.path.isdir(gt_meta_dir['eval']):
            fn_list['eval'] = sorted(os.listdir(gt_meta_dir['eval']))
        for split in ['dev', 'eval']:
            fn_list[split] = [fn for fn in fn_list[split] if fn.startswith('fold') or fn.startswith('mix')
                              and fn.endswith('csv')]
        gt_labels = {}
        for split in ['dev', 'eval']:
            for fn in fn_list[split]:
                full_filename = os.path.join(gt_meta_dir[split], fn)
                gt_dict = dcase_utils.load_output_format_file(full_filename, version=self.eval_version)
                gt_labels[fn[:-4]] = dcase_utils.segment_labels(gt_dict, _max_frames=self.max_nframes_per_file,
                                                                _nb_label_frames_1s=self.label_rate)
        return gt_labels

    def evaluate_output_prediction_csv(self, pred_filenames):
        """
        Evaluate output prediction that have been saved into csv files.
        :param pred_filenames: list of csv filenames.
        """
        seld_eval = self.seld_eval_metrics.SELDMetrics(nb_classes=self.n_classes, doa_threshold=self.doa_threshold)
        for fn in pred_filenames:
            full_filename = os.path.join(self.submission_dir, fn)
            # Load predicted output format file
            pred_dict = dcase_utils.load_output_format_file(full_filename, version=self.eval_version)
            pred_labels = dcase_utils.segment_labels(pred_dict, _max_frames=self.max_nframes_per_file,
                                                     _nb_label_frames_1s=self.label_rate)
            # Calculated scores
            seld_eval.update_seld_scores(pred_labels, self.gt_labels[fn[:-4]])
        # Overall SED and DOA scores
        ER, F1, LE, LR = seld_eval.compute_seld_scores()
        seld_error = (ER + (1.0 - F1) + LE / 180.0 + (1.0 - LR)) / 4
        return ER, F1, LE, LR, seld_error

    def write_output_prediction(self, pred_dict, target_dict, filenames):
        assert len(set(filenames)) == 1, 'Test batch contains different audio files.'
        filename = filenames[0]
        h5_filename = os.path.join(self.output_pred_dir, filename + '.h5')
        event_frame_pred = torch.sigmoid(pred_dict['event_frame_logit']).detach().cpu().numpy()
        event_frame_gt = target_dict['event_frame_gt'].detach().cpu().numpy()
        doa_frame_pred = pred_dict['doa_frame_output'].detach().cpu().numpy()
        doa_frame_gt = target_dict['doa_frame_gt'].detach().cpu().numpy()
        if self.output_format == 'accdoa':
            event_frame_pred = self.get_sed_from_accdoa_output(doa_frame_pred)
        # Combine chunk if necessary
        if event_frame_pred.shape[0] == 1:
            assert event_frame_pred.shape[1] >= self.max_nframes_per_file
        else:
            event_frame_pred = np.expand_dims(self.combine_chunks(event_frame_pred, combine_method='mean'), axis=0)
            doa_frame_pred = np.expand_dims(self.combine_chunks(doa_frame_pred, combine_method='mean'), axis=0)
        with h5py.File(h5_filename, 'w') as hf:
            hf.create_dataset('event_frame_pred', data=event_frame_pred, dtype=np.float32)
            hf.create_dataset('event_frame_gt', data=event_frame_gt, dtype=np.float32)
            hf.create_dataset('doa_frame_pred', data=doa_frame_pred, dtype=np.float32)
            hf.create_dataset('doa_frame_gt', data=doa_frame_gt, dtype=np.float32)

    def write_output_submission(self, pred_dict, filenames):
        assert len(set(filenames)) == 1, 'Test batch contains different audio files.'
        filename = filenames[0]
        submission_filename = os.path.join(self.submission_dir, filename + '.csv')
        self.write_classwise_output_to_file(pred_dict=pred_dict, submission_filename=submission_filename)

    def write_classwise_output_to_file(self, pred_dict, submission_filename: str = None):
        """
        :param pred_dict:
        # pred_dict = {
        #     'event_frame_logit': event_frame_logit,
        #     'doa_frame_output': doa_output,
        # }
        """
        doa_frame_output = pred_dict['doa_frame_output'].detach().cpu().numpy()
        if self.output_format == 'reg_xyz':
            event_frame_output = torch.sigmoid(pred_dict['event_frame_logit']).detach().cpu().numpy()
        elif self.output_format == 'accdoa':
            event_frame_output = self.get_sed_from_accdoa_output(doa_frame_output)
        else:
            raise ValueError('output format {} for classwise sequence is unknown'.format(self.output_format))
        # remove batch dimension
        if event_frame_output.shape[0] == 1:  # full file -> remove batch dimension
            event_frame_output = event_frame_output[0]
            doa_frame_output = doa_frame_output[0]
        else:  # file are divided into chunks - need to combine
            event_frame_output = self.combine_chunks(event_frame_output, combine_method='mean')
            doa_frame_output = self.combine_chunks(doa_frame_output, combine_method='mean')
        # convert sed prediction to binary
        event_frame_output = (event_frame_output >= self.sed_threshold)
        assert event_frame_output.shape[0] >= self.max_nframes_per_file, 'n_output_frames of sed < max_nframes_per_file'
        if self.output_format in ['reg_xyz', 'accdoa']:
            x = doa_frame_output[:, : self.n_classes]
            y = doa_frame_output[:, self.n_classes: 2 * self.n_classes]
            z = doa_frame_output[:, 2 * self.n_classes:]
            # convert to polar rad -> polar degree
            azi_frame_output = np.around(np.arctan2(y, x) * 180.0 / np.pi)
            ele_frame_output = np.around(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180.0 / np.pi)
        else:
            raise ValueError('Unknown doa output format {}'.format(self.output_format))
        # Loop through all the frame
        outputs = []
        for iframe in np.arange(self.max_nframes_per_file):  # trim any excessive length
            event_classes = np.where(event_frame_output[iframe] == 1)[0]
            for idx, class_idx in enumerate(event_classes):
                azi = int(azi_frame_output[iframe, class_idx])
                if azi == 180:
                    azi = -180
                ele = int(ele_frame_output[iframe, class_idx])
                if self.eval_version == '2021':
                    outputs.append([iframe, class_idx, 0, azi, ele])
                else:
                    outputs.append([iframe, class_idx, azi, ele])
        submission_df = pd.DataFrame(outputs, columns=self.df_columns)
        submission_df.to_csv(submission_filename, index=False, header=False)

    def get_sed_from_accdoa_output(self, doa_frame_pred):
        """
        Infer sed from accdoa_output.
        doa_frame_pred: (n_batch, n_timestep, n_classes * 3) -> SED: (n_batch, n_timesteps, n_classes)
        doa_frame_pred: (n_batch, n_timestep, n_classes * 3, 2) -> SED: (n_batch, n_timesteps, n_classes, 2)
        """
        x = doa_frame_pred[:, :, : self.n_classes]
        y = doa_frame_pred[:, :, self.n_classes: 2 * self.n_classes]
        z = doa_frame_pred[:, :, 2 * self.n_classes:]
        sed = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        return sed

    def compute_loss(self, target_dict, pred_dict):
        if self.output_format in ['reg_xyz']:
            loss, sed_loss, doa_loss = self.compute_classwise_clareg_loss(target_dict=target_dict,
                                                                          pred_dict=pred_dict)
        elif self.output_format == 'accdoa':
            sed_loss, doa_loss = self.compute_classwise_accdoa_loss(target_dict=target_dict, pred_dict=pred_dict)
            sed_loss = 0.0
            loss = sed_loss + doa_loss

        return loss, sed_loss, doa_loss

    def compute_classwise_accdoa_loss(self, target_dict, pred_dict):
        """
        target_dict['event_frame_gt']: (batch_size, n_timesteps, n_classes)
        target_dict['doa_frame_gt']: (batch_size, n_timesteps, 3 * n_classes)
        pred_dict['doa_frame_output']: (batch_size, n_timesteps, 3 * n_classes)
        """
        n_active = torch.sum(target_dict['event_frame_gt'])
        n_nonactive = target_dict['event_frame_gt'].shape[0] * target_dict['event_frame_gt'].shape[
            1] * self.n_classes - n_active

        xyz_loss = F.mse_loss(input=pred_dict['doa_frame_output'], target=target_dict['doa_frame_gt'],
                              reduction='none')
        x = xyz_loss[:, :, :self.n_classes]
        y = xyz_loss[:, :, self.n_classes: 2 * self.n_classes]
        z = xyz_loss[:, :, 2 * self.n_classes:]
        doa_loss = torch.sum((x + y + z) * target_dict['event_frame_gt']) / n_active

        sed = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        sed_loss = torch.sum(
            (sed - target_dict['event_frame_gt']) ** 2 * (1 - target_dict['event_frame_gt'])) / n_nonactive

        return sed_loss, doa_loss

    def compute_classwise_clareg_loss(self, target_dict, pred_dict):
        # Event frame loss
        sed_loss = F.binary_cross_entropy_with_logits(input=pred_dict['event_frame_logit'],
                                                      target=target_dict['event_frame_gt'])

        # doa frame loss
        doa_loss = self.compute_doa_reg_loss(target_dict=target_dict, pred_dict=pred_dict)

        loss = self.loss_weight[0] * sed_loss + self.loss_weight[1] * doa_loss

        return loss, sed_loss, doa_loss

    def compute_doa_reg_loss(self, target_dict, pred_dict):
        x_loss = self.compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, : self.n_classes],
                                              target=target_dict['doa_frame_gt'][:, :, : self.n_classes],
                                              mask=target_dict['event_frame_gt'])
        y_loss = self.compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, self.n_classes:2*self.n_classes],
                                              target=target_dict['doa_frame_gt'][:, :, self.n_classes:2*self.n_classes],
                                              mask=target_dict['event_frame_gt'])
        z_loss = self.compute_masked_reg_loss(input=pred_dict['doa_frame_output'][:, :, 2 * self.n_classes:],
                                              target=target_dict['doa_frame_gt'][:, :, 2 * self.n_classes:],
                                              mask=target_dict['event_frame_gt'])
        doa_loss = x_loss + y_loss + z_loss

        return doa_loss

    @staticmethod
    def compute_masked_reg_loss(input, target, mask, loss_type='MAE'):
        """
        Compute masked mean loss.
        :param input: batch_size, n_timesteps, n_classes
        :param target: batch_size, n_timestpes, n_classes
        :param mask: batch_size, n_timestpes, n_classes
        :param loss_type: choice: MSE or MAE. MAE is better for SMN
        """
        # Align the time_steps of output and target
        N = min(input.shape[1], target.shape[1])

        input = input[:, 0: N, :]
        target = target[:, 0: N, :]
        mask = mask[:, 0: N, :]

        normalize_value = torch.sum(mask)

        if loss_type == 'MAE':
            reg_loss = torch.sum(torch.abs(input - target) * mask) / normalize_value
        elif loss_type == 'MSE':
            reg_loss = torch.sum((input - target) ** 2 * mask) / normalize_value
        else:
            raise ValueError('Unknown reg loss type: {}'.format(loss_type))

        return reg_loss


