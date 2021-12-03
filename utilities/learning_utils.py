"""
This module consists code to facilitate learning such as:
    learning rate schedule
    model checkpoint
    ...
"""
import logging
import time
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn


class LearningRateScheduler(pl.Callback):
    def __init__(self, steps_per_epoch, max_epochs: int = 50, milestones: Tuple = (0, 0.45, 0.9, 1.0),
                 lrs: Tuple = (1e-4, 1e-2, 1e-3, 1e-4), moms: Tuple = (0.9, 0.8, 0.9, 0.9)):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.milestones = milestones
        self.lrs = lrs
        self.moms = moms
        self.n_steps = int(self.max_epochs * self.steps_per_epoch)
        self.step_milestones = [int(i * self.n_steps) for i in self.milestones]

    def on_train_start(self, trainer, pl_module):
        """
        Pytorch lightning hook.
        Set the initial learning rate and momentums (for Adam and AdamW optimizer)
        """
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lrs[0]
                param_group['betas'] = (self.moms[0], 0.999)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """
        Pytorch lightning hook.
        Set new learning rate.
        """
        current_global_step = trainer.current_epoch * self.steps_per_epoch + batch_idx
        lr = np.interp(current_global_step, self.step_milestones, self.lrs)
        mom = np.interp(current_global_step, self.step_milestones, self.moms)
        trainer.logger.log_metrics({'lr': lr}, step=trainer.global_step)  # trainer.global_step same as current_global_step
        trainer.logger.log_metrics({'momentum': mom}, step=trainer.global_step)
        for opt_idx, optimizer in enumerate(trainer.optimizers):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                param_group['betas'] = (mom, 0.999)


def count_model_params(model: nn.Module) -> None:
    """
    Count and log the number of trainable and total params.
    :param model: Pytorch model.
    """
    logger = logging.getLogger('lightning')
    logger.info('Model architecture: \n{}\n'.format(model))
    logger.info('Model parameters and size:')
    for n, (name, param) in enumerate(model.named_parameters()):
        logger.info('{}: {}'.format(name, list(param.size())))
    total_params = sum([param.numel() for param in model.parameters()])
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logging.info('Total number of parameters: {}'.format(total_params))
    logging.info('Total number of trainable parameters: {}'.format(trainable_params))


class MyLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.lit_logger = logging.getLogger('lightning')
        self.train_start_time = None
        self.train_end_time = None
        self.val_start_time = None
        self.val_end_time = None
        self.fit_start_time = None
        self.fit_end_time = None
        self.test_start_time = None
        self.test_end_time = None

    def on_init_start(self, trainer):
        self.lit_logger.info('Start initiating trainer!')

    def on_init_end(self, trainer):
        self.lit_logger.info('Finish initiating trainer.')

    def on_fit_start(self, trainer, pl_module):
        self.lit_logger.info('Start training...')
        self.fit_start_time = time.time()

    def on_fit_end(self, trainer, pl_module):
        self.lit_logger.info('Finish training!')
        self.fit_end_time = time.time()
        duration = self.fit_end_time - self.fit_start_time
        self.lit_logger.info('Total training time: {} s'.format(time.strftime('%H:%M:%S', time.gmtime(duration))))

    def on_test_start(self, trainer, pl_module):
        self.lit_logger.info('Start testing ...')
        self.test_start_time = time.time()

    def on_test_end(self, trainer, pl_module):
        self.lit_logger.info('Finish testing!')
        self.test_end_time = time.time()
        duration = self.test_end_time - self.test_start_time
        self.lit_logger.info('Total testing time: {} s'.format(time.strftime('%H:%M:%S', time.gmtime(duration))))

