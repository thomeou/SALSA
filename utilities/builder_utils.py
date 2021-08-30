"""
This modules consists code to select different components for
    feature_database
    models
"""
import logging
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn

import models
from dataset.database import Database
from dataset.datamodule import SeldDataModule
from models.seld_models import SeldModel


def build_database(cfg):
    """
    Function to select database according to task
    :param cfg: Experiment config
    """

    feature_db = Database(feature_root_dir=cfg.feature_root_dir, gt_meta_root_dir=cfg.gt_meta_root_dir,
                          audio_format=cfg.data.audio_format, n_classes=cfg.data.n_classes, fs=cfg.data.fs,
                          n_fft=cfg.data.n_fft, hop_len=cfg.data.hop_len, label_rate=cfg.data.label_rate,
                          train_chunk_len_s=cfg.data.train_chunk_len_s,
                          train_chunk_hop_len_s=cfg.data.train_chunk_hop_len_s,
                          test_chunk_len_s=cfg.data.test_chunk_len_s,
                          test_chunk_hop_len_s=cfg.data.test_chunk_hop_len_s,
                          output_format=cfg.data.output_format,
                          )

    return feature_db


def build_datamodule(cfg, feature_db, inference_split: str = None):
    """
    Function to select pytorch lightning datamodule according to different tasks.
    :param cfg: Experiment config.
    :param feature_db: Feature database.
    :param inference_split: Name of inference split.
    """
    datamodule = SeldDataModule(feature_db=feature_db, split_meta_dir=cfg.split_meta_dir, mode=cfg.mode,
                                inference_split=inference_split, train_batch_size=cfg.training.train_batch_size,
                                val_batch_size=cfg.training.val_batch_size, feature_type=cfg.feature_type,
                                audio_format=cfg.data.audio_format)

    return datamodule


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build encoder.
    :param name: Name of the encoder.
    :return: encoder model
    """
    logger = logging.getLogger('lightning')
    # Load model:
    model = models.__dict__[name](**kwargs)
    logger.info('Finish loading model {}.'.format(name))

    return model


def build_task(encoder, decoder, cfg, output_pred_dir: str = None, submission_dir: str = None,
               test_chunk_len: int = None, test_chunk_hop_len: int = None, is_tta: bool = False,
               inference_split: str = None, **kwargs) -> pl.LightningModule:
    """
    Build task
    :param encoder: encoder module.
    :param decoder: decoder module.
    :param cfg: experiment config.
    :param output_pred_dir: Directory to write prediction.
    :param submission_dir: Directory to write output csv file.
    :param test_chunk_len: test chunk_len of sed feature. Required for inference that divide test files into smaller
        chunk
    :param test_chunk_hop_len: test chunk_hop_len of sed feature. Required for inference that divide test files into
        smaller chunk
    :param is_tta: if True, do test time augmentation.
    :return: Lightning module
    """
    feature_rate = cfg.data.fs / cfg.data.hop_len  # Frame rate per second. Duplicate info from feature database
    is_eval = inference_split == 'eval'  # gt for eval is not availabel yet. So no evaluation for eval split
    model = SeldModel(encoder=encoder, decoder=decoder, sed_threshold=cfg.sed_threshold,
                      doa_threshold=cfg.doa_threshold, label_rate=cfg.data.label_rate, feature_rate=feature_rate,
                      optimizer_name=cfg.training.optimizer, loss_weight=cfg.training.loss_weight,
                      output_pred_dir=output_pred_dir, submission_dir=submission_dir, test_chunk_len=test_chunk_len,
                      test_chunk_hop_len=test_chunk_hop_len, gt_meta_root_dir=cfg.gt_meta_root_dir,
                      output_format=cfg.data.output_format, eval_version=cfg.eval_version, is_eval=is_eval)

    return model

