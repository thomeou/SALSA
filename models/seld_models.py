"""
This module include code to perform SELD task.
output rates:
  - feature: has feature rates
  - gt: has output/label rate
"""
import os
import shutil
from typing import Tuple

import torch.nn as nn

from models.interfaces import BaseModel
from models.model_utils import interpolate_tensor


class SeldModel(BaseModel):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, sed_threshold: float = 0.3, doa_threshold: int = 20,
                 label_rate: int = 10, feature_rate: int = None, optimizer_name: str = 'Adam', lr: float = 1e-3,
                 loss_weight: Tuple = None, output_pred_dir: str = None, submission_dir: str = None,
                 test_chunk_len: int = None, test_chunk_hop_len: int = None, gt_meta_root_dir: str = None,
                 output_format: str = None, eval_version: str = '2021', is_eval: bool = False, **kwargs):
        super().__init__(sed_threshold=sed_threshold, doa_threshold=doa_threshold, label_rate=label_rate,
                         feature_rate=feature_rate, optimizer_name=optimizer_name, lr=lr,
                         output_pred_dir=output_pred_dir, submission_dir=submission_dir, test_chunk_len=test_chunk_len,
                         test_chunk_hop_len=test_chunk_hop_len, gt_meta_root_dir=gt_meta_root_dir,
                         output_format=output_format, eval_version=eval_version)
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_weight = loss_weight
        self.time_downsample_ratio = float(self.encoder.time_downsample_ratio)
        self.n_classes = self.decoder.n_classes
        self.doa_format = self.decoder.doa_format  # doa_format redundance since we have doa_output_format
        self.is_eval = is_eval

        self.seld_val = None

    def forward(self, x):
        """
        x: (batch_size, n_channels, n_timesteps (n_frames), n_features).
        """
        x = self.encoder(x)  # (batch_size, n_channels, n_timesteps, n_features)
        output_dict = self.decoder(x)
        # output_dict = {
        #     'event_frame_logit': event_frame_logit, # (batch_size, n_timesteps, n_classes)
        #     'doa_frame_output': doa_output, # (batch_size, n_timesteps, 3* n_classes)
        # }
        return output_dict

    def common_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)

        return target_dict, pred_dict

    def training_step(self, train_batch, batch_idx):
        target_dict, pred_dict = self.common_step(train_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # logging
        self.log('trl', loss, prog_bar=True, logger=True)
        self.log('trsl', sed_loss, prog_bar=True, logger=True)
        self.log('trdl', doa_loss, prog_bar=True, logger=True)
        training_step_outputs = {'loss': loss}
        return training_step_outputs

    def training_epoch_end(self, training_step_outputs):
        # clear temp folder to write val output
        if self.submission_dir is not None:
            shutil.rmtree(self.submission_dir, ignore_errors=True)
            os.makedirs(self.submission_dir, exist_ok=True)

    def validation_step(self, val_batch, batch_idx):
        target_dict, pred_dict = self.common_step(val_batch)
        loss, sed_loss, doa_loss = self.compute_loss(target_dict=target_dict, pred_dict=pred_dict)
        # write output file
        filenames = val_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # logging
        self.log('vall', loss, prog_bar=True, logger=True)
        self.log('valsl', sed_loss, prog_bar=True, logger=True)
        self.log('valdl', doa_loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, validation_step_outputs):
        # Get list of csv filename
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        # Compute validation metrics
        ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Validation - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def test_step(self, test_batch, batch_idx):
        target_dict, pred_dict = self.common_test_step(test_batch)
        # write output submission
        filenames = test_batch[-1]
        self.write_output_submission(pred_dict=pred_dict, filenames=filenames)
        # write output prediction
        if self.output_pred_dir:
            self.write_output_prediction(pred_dict=pred_dict, target_dict=target_dict, filenames=filenames)

    def test_epoch_end(self, test_step_outputs):
        pred_filenames = os.listdir(self.submission_dir)
        pred_filenames = [fn for fn in pred_filenames if fn.endswith('csv')]
        self.lit_logger.info('Number of test files: {}'.format(len(pred_filenames)))
        # Compute validation metrics
        if self.is_eval:
            ER, F1, LE, LR, seld_error = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            ER, F1, LE, LR, seld_error = self.evaluate_output_prediction_csv(pred_filenames=pred_filenames)
        # log metrics
        self.log('valER', ER)
        self.log('valF1', F1)
        self.log('valLE', LE)
        self.log('valLR', LR)
        self.log('valSeld', seld_error)
        self.lit_logger.info('Epoch {} - Test - SELD: {:.4f} - SED ER: {:.4f} - F1: {:.4f} - DOA LE: {:.4f} - '
                             'LR: {:.4f}'.format(self.current_epoch, seld_error, ER, F1, LE, LR))

    def common_test_step(self, batch_data):
        x, y_sed, y_doa, _ = batch_data
        # target dict has frame_rate = label_rate
        target_dict = {
            'event_frame_gt': y_sed,
            'doa_frame_gt': y_doa,
        }
        # forward
        pred_dict = self.forward(x)
        # interpolate output to match label rate
        pred_dict['event_frame_logit'] = interpolate_tensor(
            pred_dict['event_frame_logit'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)
        pred_dict['doa_frame_output'] = interpolate_tensor(
            pred_dict['doa_frame_output'], ratio=self.time_downsample_ratio * self.label_rate / self.feature_rate)

        return target_dict, pred_dict




