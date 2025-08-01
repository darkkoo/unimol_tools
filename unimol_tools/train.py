# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import argparse
import copy
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from .data import DataHub
from .models import NNModel
from .tasks import Trainer
from .utils import YamlHandler, logger


class MolTrain(object):
    """A :class:`MolTrain` class is responsible for interface of training process of molecular data."""

    def __init__(
        self,
        task='classification',
        data_type='molecule',
        epochs=10,
        learning_rate=1e-4,
        batch_size=16,
        early_stopping=5,
        metrics="none",
        split='random',  # random, scaffold, group, stratified
        split_group_col='scaffold',  # only active with group split
        kfold=5,
        save_path='./exp',
        remove_hs=False,
        smiles_col='SMILES',
        target_cols=None,
        target_col_prefix='TARGET',
        target_anomaly_check=False,
        smiles_check="filter",
        target_normalize="auto",
        max_norm=5.0,
        use_cuda=True,
        use_amp=True,
        use_ddp=False,
        use_gpu="all",
        freeze_layers=None,
        freeze_layers_reversed=False,
        load_model_dir=None,  # load model for transfer learning
        model_name='unimolv1',
        model_size='84m',
        conf_cache_level=1,
        **params,
    ):
        """
        Initialize a :class:`MolTrain` class.

        :param task: str, default='classification', currently support [`classification`, `regression`, `multiclass`, `multilabel_classification`, `multilabel_regression`].
        :param data_type: str, default='molecule', currently support molecule, oled.
        :param epochs: int, default=10, number of epochs to train.
        :param learning_rate: float, default=1e-4, learning rate of optimizer.
        :param batch_size: int, default=16, batch size of training.
        :param early_stopping: int, default=5, early stopping patience.
        :param metrics: str, default='none', metrics to evaluate model performance.

            currently support: 

            - classification: auc, auprc, log_loss, acc, f1_score, mcc, precision, recall, cohen_kappa. 

            - regression: mse, pearsonr, spearmanr, mse, r2.

            - multiclass: log_loss, acc.

            - multilabel_classification: auc, auprc, log_loss, acc, mcc.

            - multilabel_regression: mae, mse, r2.

        :param split: str, default='random', split method of training dataset. currently support: random, scaffold, group, stratified, select.

            - random: random split.

            - scaffold: split by scaffold.

            - group: split by group. `split_group_col` should be specified.

            - stratified: stratified split. `split_group_col` should be specified.

            - select: use `split_group_col` to manually select the split group. Column values of `split_group_col` should be range from 0 to kfold-1 to indicate the split group.

        :param split_group_col: str, default='scaffold', column name of group split.
        :param kfold: int, default=5, number of folds for k-fold cross validation.

            - 1: no split. all data will be used for training.
        
        :param save_path: str, default='./exp', path to save training results.
        :param remove_hs: bool, default=False, whether to remove hydrogens from molecules.
        :param smiles_col: str, default='SMILES', column name of SMILES.
        :param target_cols: list or str, default=None, column names of target values.
        :param target_col_prefix: str, default='TARGET', prefix of target column name.
        :param target_anomaly_check: str, default=False, how to deal with anomaly target values. currently support: filter, none.
        :param smiles_check: str, default='filter', how to deal with invalid SMILES. currently support: filter, none.
        :param target_normalize: str, default='auto', how to normalize target values. 'auto' means we will choose the normalize strategy by automatic. \
            currently support: auto, minmax, standard, robust, log1p, none.
        :param max_norm: float, default=5.0, max norm of gradient clipping.
        :param use_cuda: bool, default=True, whether to use GPU.
        :param use_amp: bool, default=True, whether to use automatic mixed precision.
        :param use_ddp: bool, default=True, whether to use distributed data parallel.
        :param use_gpu: str, default='all', which GPU to use. 'all' means use all GPUs. '0,1,2' means use GPU 0, 1, 2.
        :param freeze_layers: str or list, frozen layers by startwith name list. ['encoder', 'gbf'] will freeze all the layers whose name start with 'encoder' or 'gbf'.
        :param freeze_layers_reversed: bool, default=False, inverse selection of frozen layers
        :param params: dict, default=None, other parameters.
        :param load_model_dir: str, default=None, path to load model for transfer learning.
        :param model_name: str, default='unimolv1', currently support unimolv1, unimolv2.
        :param model_size: str, default='84m', model size. work when model_name is unimolv2. Avaliable: 84m, 164m, 310m, 570m, 1.1B.
        :param conf_cache_level: int, optional [0, 1, 2], default=1, configuration cache level to save the conformers to sdf file.
            - 0: no caching.
            - 1: cache if not exists.
            - 2: always cache.

        """
        if load_model_dir is not None:
            config_path = os.path.join(load_model_dir, 'config.yaml')
            logger.info('Load config file from {}'.format(config_path))
        else:
            config_path = os.path.join(os.path.dirname(__file__), 'config/default.yaml')
        self.yamlhandler = YamlHandler(config_path)
        config = self.yamlhandler.read_yaml()
        config.task = task
        config.data_type = data_type
        config.epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.patience = early_stopping
        config.metrics = metrics
        config.split = split
        config.split_group_col = split_group_col
        config.kfold = kfold
        config.remove_hs = remove_hs
        config.smiles_col = smiles_col
        config.target_cols = target_cols
        config.target_col_prefix = target_col_prefix
        config.anomaly_clean = target_anomaly_check or target_anomaly_check in [
            'filter'
        ]
        config.smi_strict = smiles_check in ['filter']
        config.target_normalize = target_normalize
        config.max_norm = max_norm
        config.use_cuda = use_cuda
        config.use_amp = use_amp
        config.use_ddp = use_ddp
        config.use_gpu = use_gpu
        config.freeze_layers = freeze_layers
        config.freeze_layers_reversed = freeze_layers_reversed
        config.load_model_dir = load_model_dir
        config.model_name = model_name
        config.model_size = model_size
        config.conf_cache_level = conf_cache_level
        self.save_path = save_path
        self.config = config

    def fit(self, data):
        """
        Fit the model according to the given training data with multi datasource support, including SMILES csv file and custom coordinate data.

        For example: custom coordinate data.

        .. code-block:: python

            from unimol_tools import MolTrain
            import numpy as np
            custom_data ={'target':np.random.randint(2, size=100),
                        'atoms':[['C','C','H','H','H','H'] for _ in range(100)],
                        'coordinates':[np.random.randn(6,3) for _ in range(100)],
                        }

            clf = MolTrain()
            clf.fit(custom_data)
        """
        self.datahub = DataHub(
            data=data, is_train=True, save_path=self.save_path, **self.config
        )
        self.data = self.datahub.data
        self.update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = NNModel(self.data, self.trainer, **self.config)
        self.model.run()
        scalar = self.data['target_scaler']
        y_pred = self.model.cv['pred']
        y_true = np.array(self.data['target'])
        metrics = self.trainer.metrics
        if scalar is not None:
            y_pred = scalar.inverse_transform(y_pred)
            y_true = scalar.inverse_transform(y_true)

        if self.config["task"] in ['classification', 'multilabel_classification']:
            threshold = metrics.calculate_classification_threshold(y_true, y_pred)
            joblib.dump(threshold, os.path.join(self.save_path, 'threshold.dat'))

        self.cv_pred = y_pred
        return

    def update_and_save_config(self):
        """
        Update and save config file.
        """
        self.config['num_classes'] = self.data['num_classes']
        self.config['target_cols'] = ','.join(self.data['target_cols'])
        if self.config['task'] == 'multiclass':
            self.config['multiclass_cnt'] = self.data['multiclass_cnt']

        self.config['split_method'] = (
            f"{self.config['kfold']}fold_{self.config['split']}"
        )
        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                logger.info('Create output directory: {}'.format(self.save_path))
                os.makedirs(self.save_path)
            else:
                logger.info(
                    'Output directory already exists: {}'.format(self.save_path)
                )
                logger.info(
                    'Warning: Overwrite output directory: {}'.format(self.save_path)
                )
            out_path = os.path.join(self.save_path, 'config.yaml')
            self.yamlhandler.write_yaml(data=self.config, out_file_path=out_path)
        return
