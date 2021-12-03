"""
This module includes code to inference SELD task. Input is a segment/chunk of 60-second audio clips. The segment length
can be 60-s. There are options to write output predictions and prediction files to disk.
Directory to write output predictions:
    exp_root/outputs/predictions/split_group/split_folder_name/fold6_room1_mix001.h5
    exp_root/outputs/submissions/split_group/split_folder_name/fold6_room1_mix001.csv
split_group: 'original', ...
split_name: 'foa_test', 'foa_val', 'mic_test', mic_val', ...
"""
import fire
import logging
import os
import re

import pytorch_lightning as pl
import torch

from experiments.evaluate import evaluate_seld
from utilities.builder_utils import build_database, build_datamodule, build_model, build_task
from utilities.experiments_utils import manage_experiments
from utilities.learning_utils import MyLoggingCallback


def inference(exp_config: str = './configs/seld.yml',
              exp_group_dir: str = '/media/tho_nguyen/disk2/new_seld/dcase2021/outputs',
              exp_suffix: str = '_test',
              epoch: int = None,
              ckpt_type: str = 'best',
              inference_split: str = 'val',
              submission_tag: str = ''
              ):
    """
    Inference script for one split.
    :param exp_config: Config file for experiments
    :param exp_group_dir: Parent directory to store all experiment results.
    :param exp_suffix: Experiment suffix.
    :param epoch: Epoch of the checkpoint. If None, choose the best one in the 'best' or 'checkpoint' folder.
    :param ckpt_type: Type of checkpoint, can be 'best', 'checkpoint'
    :param inference_split: split to do inference if stage = 'inference'
    :param submission_tag: tag name to submission file.
    """
    # Load config, create folders, logging
    cfg = manage_experiments(exp_config=exp_config, exp_group_dir=exp_group_dir, exp_suffix=exp_suffix, is_train=False)
    logger = logging.getLogger('lightning')
    split_group = os.path.split(cfg.split_meta_dir)[-1]
    logger.info('Inference for split : {} in split meta {}'.format(inference_split, split_group))
    # model checkpoint
    if ckpt_type == 'best':
        ckpt_list = [f for f in os.listdir(os.path.join(cfg.dir.model.best)) if
                     f.startswith('epoch') and f.endswith('ckpt')]
        assert len(ckpt_list) >= 1, 'No checkpoint found'
        if epoch is not None:
            ckpt_list = [f for f in ckpt_list if int(f[6:9]) == epoch]
            ckpt_name = ckpt_list[0]
        else:
            min_error = 1000
            for idx, ichpt in enumerate(ckpt_list):
                cur_error = float(re.findall(r"[-+]?\d*\.\d+|\d+", ichpt)[1])
                if cur_error <= min_error:
                    min_error = cur_error
                    min_idx = idx
            ckpt_name = ckpt_list[min_idx]
        ckpt_path = os.path.join(cfg.dir.model.best, ckpt_name)
    elif ckpt_type == 'checkpoint':
        ckpt_list = sorted([f for f in os.listdir(os.path.join(cfg.dir.model.checkpoint)) if
                            f.startswith('epoch') and f.endswith('ckpt')])
        assert len(ckpt_list) >= 1, 'No checkpoint found'
        if epoch is not None:
            ckpt_list = [f for f in ckpt_list if int(f[6:9]) == epoch]
            ckpt_name = ckpt_list[0]
        else:
            ckpt_name = ckpt_list[-1]
        ckpt_path = os.path.join(cfg.dir.model.checkpoint, ckpt_name)
    else:
        raise ValueError('Unknown checkpoint type :{}'.format(ckpt_type))
    logger.info('Model checkpoint: {}'.format(ckpt_path))
    # submission dir
    submission_dir = os.path.join(cfg.dir.output_dir.submission, split_group + submission_tag,
                                      cfg.data.audio_format + '_' + inference_split)
    os.makedirs(submission_dir, exist_ok=True)
    logger.info('Submission dir: {}'.format(submission_dir))
    # output pred directory:
    output_pred_dir = os.path.join(cfg.dir.output_dir.prediction, split_group + submission_tag,
                                       cfg.data.audio_format + '_' + inference_split)
    os.makedirs(output_pred_dir, exist_ok=True)
    logger.info('Inference directory: {}'.format(output_pred_dir))

    # Set random seed for reproducible
    pl.seed_everything(cfg.seed)

    # Load feature database
    feature_db = build_database(cfg=cfg)

    # Load data module
    datamodule = build_datamodule(cfg=cfg, feature_db=feature_db, inference_split=inference_split)
    datamodule.setup(stage='inference')

    # Console logger
    console_logger = MyLoggingCallback()

    # Build encoder and decoder
    encoder_params = cfg.model.encoder.__dict__
    encoder = build_model(**encoder_params)
    decoder_params = cfg.model.decoder.__dict__
    decoder_params = {'n_output_channels': encoder.n_output_channels, 'n_classes': cfg.data.n_classes,
                      'output_format': cfg.data.output_format, **decoder_params}
    decoder = build_model(**decoder_params)

    # Build Lightning model
    model = build_task(encoder=encoder, decoder=decoder, cfg=cfg, output_pred_dir=output_pred_dir,
                       submission_dir=submission_dir, test_chunk_len=feature_db.test_chunk_len,
                       test_chunk_hop_len=feature_db.test_chunk_hop_len, inference_split=inference_split)

    # Manually load model
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # Test
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), limit_test_batches=cfg.data.val_fraction,
                         callbacks=[console_logger])
    trainer.test(model=model, test_dataloaders=datamodule.test_dataloader())

    # Evaluate
    evaluate_seld(output_dir=submission_dir, data_version='2021', metric_version='2021',
                  is_eval_split=inference_split=='eval')


def inference_all_splits(exp_config: str = './configs/seld.yml',
                         exp_group_dir: str = '/media/tho_nguyen/disk2/new_seld/dcase2021/outputs',
                         exp_suffix: str = '_test',
                         epoch: int = None,
                         ckpt_type: str = 'best',  # 'best' | 'checkpoint' (automatic select best model in 'best' or 'checkpoint')
                         submission_tag: str = '',
                         ):
    """
    Inference script for a list of splits.
    :param exp_config: Config file for experiments
    :param exp_group_dir: Parent directory to store all experiment results.
    :param exp_suffix: Experiment suffix.
    :param epoch: Epoch of the checkpoint. If None, choose the best one in the 'best' or 'checkpoint' folder.
    :param ckpt_type: Type of checkpoint, can be 'best', 'checkpoint',
    :param submission_tag: tag name to submission file.
    """
    splits = ['val', 'test', ]
    for split in splits:
        inference(exp_config=exp_config,
                  exp_group_dir=exp_group_dir,
                  exp_suffix=exp_suffix,
                  epoch=epoch,
                  ckpt_type=ckpt_type,
                  inference_split=split,
                  submission_tag=submission_tag)


if __name__ == '__main__':
    fire.Fire(inference_all_splits)
