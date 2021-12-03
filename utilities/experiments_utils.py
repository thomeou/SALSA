"""
This modules consists utility functions to organize folders, create logger for experiments.
"""
import logging
import os
import shutil
from datetime import datetime

import yaml
from pathlib import Path
from munch import munchify, Munch
from tqdm import tqdm


def manage_experiments(exp_config: str = 'configs/exp1.yml',
                       exp_group_dir: str = '',
                       exp_suffix: str = '_first_exp',
                       is_train: bool = True):
    """
    Function to load config, create folder and logging.
    :param exp_config: Config file for experiments
    :param exp_group_dir: Parent directory to store all experiment results.
    :param exp_suffix: Experiment suffix.
    :param is_train: If True, create folders.
    :return: config
    """
    # Load data config files
    with open(exp_config, 'r') as stream:
        try:
            cfg_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Convert dictionary to object
    cfg = munchify(cfg_dict)

    # Create experiment folder
    exp_name = os.path.splitext(os.path.basename(exp_config))[0] + exp_suffix
    create_exp_folders(cfg=cfg, exp_group_dir=exp_group_dir, exp_name=exp_name, is_train=is_train)

    # Check if exp folder exists
    assert Path(cfg.dir.exp_dir).is_dir(), 'Experiment folder does not exist!'

    # Create logging
    create_logging(log_dir=cfg.dir.logs_dir, filemode='a')
    logger = logging.getLogger('lightning')

    # Write config file to output folder
    if is_train:
        yaml_config_fn = os.path.join(cfg.dir.config_dir,
                                      'exp_config_{}.yml'.format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        write_yaml_config(output_filename=yaml_config_fn, config_dict=cfg_dict)
        logger.info('Write yaml config file to {}'.format(cfg.dir.config_dir))

    logger.info('Finish parsing config file: {}.'.format(exp_config))

    return cfg


def create_empty_folder(folder_name) -> None:
    shutil.rmtree(folder_name, ignore_errors=True)
    os.makedirs(folder_name, exist_ok=True)


def create_exp_folders(cfg, exp_group_dir: str = '', exp_name: str = '', empty: bool = False,
                       is_train: bool = True) -> None:
    """
    Create folders required for experiments.
    :param cfg: Experiment config object.
    :param exp_group_dir: Experiment directory.
    :param exp_name: Experiment name.
    :param empty: If true, delete all previous data in experiment folder.
    :param is_train: If True, create folders.
    """
    # 1. Experiment directory
    cfg.dir = Munch()
    cfg.dir.exp_dir = os.path.join(exp_group_dir, cfg.mode, cfg.data.audio_format, cfg.feature_type, exp_name)
    if is_train:
        if empty:
            create_empty_folder(cfg.dir.exp_dir)
        else:
            os.makedirs(cfg.dir.exp_dir, exist_ok=True)

    # 2. config directory
    cfg.dir.config_dir = os.path.join(cfg.dir.exp_dir, 'configs')
    if is_train:
        os.makedirs(cfg.dir.config_dir, exist_ok=True)

    # 3. log directory
    cfg.dir.logs_dir = os.path.join(cfg.dir.exp_dir, 'logs')
    if is_train:
        os.makedirs(cfg.dir.logs_dir, exist_ok=True)

    # 4. tensorboard directory
    cfg.dir.tb_dir = os.path.join(cfg.dir.exp_dir, 'tensorboard')
    if is_train:
        os.makedirs(cfg.dir.tb_dir, exist_ok=True)

    # 5. model directory
    cfg.dir.model = Munch()
    # 5.1 model checkpoint
    cfg.dir.model.checkpoint = os.path.join(cfg.dir.exp_dir, 'models', 'checkpoint')
    if is_train:
        os.makedirs(cfg.dir.model.checkpoint, exist_ok=True)
    # 5.2 best model
    cfg.dir.model.best = os.path.join(cfg.dir.exp_dir, 'models', 'best')
    if is_train:
        os.makedirs(cfg.dir.model.best, exist_ok=True)

    # 6. output directory
    cfg.dir.output_dir = Munch()
    # 6.1 submission directory
    cfg.dir.output_dir.submission = os.path.join(cfg.dir.exp_dir, 'outputs', 'submissions')
    if is_train:
        os.makedirs(cfg.dir.output_dir.submission, exist_ok=True)
    # 6.2 prediction directory
    cfg.dir.output_dir.prediction = os.path.join(cfg.dir.exp_dir, 'outputs', 'predictions')
    if is_train:
        os.makedirs(cfg.dir.output_dir.prediction, exist_ok=True)


class TqdmLoggingHandler(logging.Handler):
    """Log consistently when using the tqdm progress bar.
    From https://stackoverflow.com/questions/38543506/
    change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def create_logging(log_dir, filemode='a') -> None:
    """
    Initialize logger.
    """
    # log_filename
    log_filename = os.path.join(log_dir, 'log.txt')

    if not logging.getLogger().hasHandlers():
        # basic config for logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%a, %d %b %Y %H:%M:%S',
            filename=log_filename,
            filemode=filemode)

        # Get lightning logger.
        logger = logging.getLogger("lightning")
        logger.setLevel(logging.INFO)
        # Purge old handlers.
        for old_handler in logger.handlers:
            logger.removeHandler(old_handler)

        # create tqdm handler
        handler = TqdmLoggingHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # handler.setFormatter(formatter)
        # add tqdm handler to current logger
        logger.addHandler(handler)

        # For normal code without lightning: logger = logging.getLogger('my_logger')
        # logger = logging.getLogger('my_logger')
        # if not logger.handlers:
        #     console = logging.StreamHandler()
        #     console.setLevel(logging.INFO)
        #     formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        #     console.setFormatter(formatter)
        #     logging.getLogger('').addHandler(TqdmLoggingHandler(logging.INFO))

    logger = logging.getLogger("lightning")
    logger.info('**********************************************************')
    logger.info('****** Start new experiment ******************************')
    logger.info('**********************************************************\n')
    logger.info('Timestamp: {}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    logger.info('Log file is created in {}.'.format(log_dir))


def write_yaml_config(output_filename, config_dict) -> None:
    """
    Write configs to yaml file for reference later.
    """
    with open(output_filename, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=True)
