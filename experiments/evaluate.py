"""
This script evaluates the output submission with ground truth.
Input to evaluate: output submission csv file.
Task: seld.
This file is redundant as seld evaluation is similar in inference.py (we are evaluating output submission in model)
"""
import fire
import logging
import os

import numpy as np

from metrics import dcase_utils, SELD2020_evaluation_metrics, SELD2021_evaluation_metrics


def evaluate_seld(output_dir: str = './outputs/crossval/foa/salsa/seld_test/outputs/submissions/original/foa_test',
                  data_version: str = '2021',
                  metric_version: str = '2021',
                  gt_meta_root_dir: str = 'dataset/data',
                  is_eval_split: bool = False,
                  ):
    """
    Evaluation script for one split.
    :param output_dir: Directory that store csv file
    :param data_version: Version of SELD dataset, can be 2020 or 2021.
    :param metric_version: Version of the evaluation metrics.
    :param is_eval_split: If inference split is 'eval', set is_eval_split to True.
    """
    logger = logging.getLogger('lightning')

    if data_version == '2020':
        n_classes = 14
        if is_eval_split:
            gt_meta_dir = os.path.join(gt_meta_root_dir, 'metadata_eval')
        else:
            gt_meta_dir = os.path.join(gt_meta_root_dir, 'metadata_dev')
    elif data_version == '2021':
        n_classes = 12
        if is_eval_split == 'eval':
            raise ValueError('Eval set for 2021 dataset has not yet had ground truth')
        else:
            gt_meta_dir = os.path.join(gt_meta_root_dir, 'metadata_dev')
    else:
        raise ValueError('data version {} is unknown'.format(data_version))
    if metric_version == '2020':
        seld_eval_metrics = SELD2020_evaluation_metrics
    elif metric_version == '2021':
        seld_eval_metrics = SELD2021_evaluation_metrics
    else:
        raise ValueError('metric version {} is unknown'.format(metric_version))
    label_rate = 10
    n_max_frames_per_file = 600
    doa_threshold = 20
    # initialize SELD metric
    seld_eval = seld_eval_metrics.SELDMetrics(nb_classes=n_classes, doa_threshold=doa_threshold)
    # Load gt
    fn_list = sorted(os.listdir(gt_meta_dir))
    fn_list = [fn for fn in fn_list if (fn.startswith('fold') or fn.startswith('mix')) and fn.endswith('csv')]
    gt_labels = {}
    for fn in fn_list:
        full_filename = os.path.join(gt_meta_dir, fn)
        gt_dict = dcase_utils.load_output_format_file(full_filename, version=metric_version)
        gt_labels[fn[:-4]] = dcase_utils.segment_labels(gt_dict, _max_frames=n_max_frames_per_file,
                                                        _nb_label_frames_1s=label_rate)
    # Load prediction
    pred_filenames = sorted(os.listdir(output_dir))
    pred_filenames = [fn for fn in pred_filenames if (fn.startswith('fold') or fn.startswith('mix'))
                      and fn.endswith('csv')]

    for fn in pred_filenames:
        full_filename = os.path.join(output_dir, fn)
        # Load predicted output format file
        pred_dict = dcase_utils.load_output_format_file(full_filename, version=metric_version)
        pred_labels = dcase_utils.segment_labels(pred_dict, _max_frames=n_max_frames_per_file,
                                                 _nb_label_frames_1s=label_rate)
        # Calculated scores
        seld_eval.update_seld_scores(pred_labels, gt_labels[fn[:-4]])
    # Overall SED and DOA scores
    ER, F1, LE, LR = seld_eval.compute_seld_scores()
    seld_error = (ER + (1.0 - F1) + LE / 180.0 + (1.0 - LR)) / 4
    logger.info('{} SELD metrics: SELD error: {:.3f} - SED ER: {:.3f} - SED F1: {:.3f} - '
                'DOA LE: {:.3f} - DOA LR: {:.3f}'.format(metric_version, seld_error, ER, F1, LE, LR))
    print('{} SELD metrics: SELD error: {:.3f} - SED ER: {:.3f} - SED F1: {:.3f} - '
          'DOA LE: {:.3f} - DOA LR: {:.3f}'.format(metric_version, seld_error, ER, F1, LE, LR))

    return ER, F1, LE, LR, seld_error


if __name__ == '__main__':
    fire.Fire(evaluate_seld)
