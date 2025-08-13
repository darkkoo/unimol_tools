import os
import random
import logging

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from unimol_tools.pretrain import (
    LMDBDataset,
    UniMolDataset,
    UniMolLoss,
    UniMolModel,
    UniMolPretrainTrainer,
    UniMolV2Dataset,
    UniMolV2Model,
    UniMolV2Loss,
    build_dictionary,
    preprocess_dataset,
)
from unimol_tools.pretrain.pretrain_config import apply_unimolv2_model_defaults

logger = logging.getLogger(__name__)


class MolPretrain:
    def __init__(self, cfg: DictConfig):
        self.config = cfg
        apply_unimolv2_model_defaults(self.config.model)
        self.local_rank = getattr(self.config.training, "local_rank", 0)
        seed = getattr(self.config.training, "seed", 42)
        self.set_seed(seed)
        
        ds_cfg = self.config.dataset
        train_lmdb = ds_cfg.train_path
        val_lmdb = ds_cfg.valid_path
        if ds_cfg.data_type != "lmdb" and not ds_cfg.train_path.endswith(".lmdb"):
            lmdb_path = os.path.splitext(ds_cfg.train_path)[0] + ".lmdb"
            logger.info(
                f"Preprocessing training data from {ds_cfg.train_path} to {lmdb_path}"
            )
            preprocess_dataset(
                ds_cfg.train_path,
                lmdb_path,
                data_type=ds_cfg.data_type,
                smiles_col=ds_cfg.smiles_column,
                num_conf=ds_cfg.num_conformers if ds_cfg.add_2d else 1,
                add_2d=ds_cfg.add_2d,
                remove_hs=ds_cfg.remove_hydrogen,
            )
            train_lmdb = lmdb_path
            logger.info(
                f"Dataset preprocessing finished, LMDB saved at {lmdb_path}"
            )

            if ds_cfg.valid_path:
                val_lmdb = os.path.splitext(ds_cfg.valid_path)[0] + ".lmdb"
                logger.info(
                    f"Preprocessing validation data from {ds_cfg.valid_path} to {val_lmdb}"
                )
                preprocess_dataset(
                    ds_cfg.valid_path,
                    val_lmdb,
                    data_type=ds_cfg.data_type,
                    smiles_col=ds_cfg.smiles_column,
                    num_conf=ds_cfg.num_conformers if ds_cfg.add_2d else 1,
                    add_2d=ds_cfg.add_2d,
                    remove_hs=ds_cfg.remove_hydrogen,
                )
                logger.info(
                    f"Validation dataset preprocessing finished, LMDB saved at {val_lmdb}"
                )

        model_name = self.config.model.model_name.lower()
        if model_name == "unimolv2":
            logger.info(f"Loading LMDB dataset from {train_lmdb}")
            lmdb_dataset = LMDBDataset(train_lmdb)
            self.dataset = UniMolV2Dataset(
                lmdb_dataset,
                remove_hs=ds_cfg.remove_hydrogen,
                max_atoms=ds_cfg.max_atoms,
                seed=seed,
                noise_type=ds_cfg.noise_type,
                noise=ds_cfg.noise,
                mask_token_prob=ds_cfg.mask_token_prob,
                drop_feat_prob=ds_cfg.drop_feat_prob,
                use_2d_pos_prob=ds_cfg.use_2d_pos_prob,
                sample_conformer=ds_cfg.add_2d,
            )
            if val_lmdb:
                logger.info(f"Loading validation LMDB dataset from {val_lmdb}")
                val_lmdb_dataset = LMDBDataset(val_lmdb)
                self.valid_dataset = UniMolV2Dataset(
                    val_lmdb_dataset,
                    remove_hs=ds_cfg.remove_hydrogen,
                    max_atoms=ds_cfg.max_atoms,
                    seed=seed,
                    noise_type=ds_cfg.noise_type,
                    noise=ds_cfg.noise,
                    mask_token_prob=ds_cfg.mask_token_prob,
                    drop_feat_prob=ds_cfg.drop_feat_prob,
                    use_2d_pos_prob=ds_cfg.use_2d_pos_prob,
                    sample_conformer=ds_cfg.add_2d,
                )
            else:
                self.valid_dataset = None
            self.dictionary = None
        else:
            dict_path = ds_cfg.get("dict_path", None)
            if dict_path:
                from unimol_tools.data.dictionary import Dictionary

                self.dictionary = Dictionary.load(dict_path)
                logger.info(f"Loaded dictionary from {dict_path}")
            else:
                self.dictionary = build_dictionary(train_lmdb)
                logger.info("Built dictionary from training LMDB")

            logger.info(f"Loading LMDB dataset from {train_lmdb}")
            lmdb_dataset = LMDBDataset(train_lmdb)
            self.dataset = UniMolDataset(
                lmdb_dataset,
                self.dictionary,
                remove_hs=ds_cfg.remove_hydrogen,
                max_atoms=ds_cfg.max_atoms,
                seed=seed,
                noise_type=ds_cfg.noise_type,
                noise=ds_cfg.noise,
                mask_prob=ds_cfg.mask_prob,
                leave_unmasked_prob=ds_cfg.leave_unmasked_prob,
                random_token_prob=ds_cfg.random_token_prob,
                sample_conformer=ds_cfg.add_2d,
            )
            if val_lmdb:
                logger.info(f"Loading validation LMDB dataset from {val_lmdb}")
                val_lmdb_dataset = LMDBDataset(val_lmdb)
                self.valid_dataset = UniMolDataset(
                    val_lmdb_dataset,
                    self.dictionary,
                    remove_hs=ds_cfg.remove_hydrogen,
                    max_atoms=ds_cfg.max_atoms,
                    seed=seed,
                    noise_type=ds_cfg.noise_type,
                    noise=ds_cfg.noise,
                    mask_prob=ds_cfg.mask_prob,
                    leave_unmasked_prob=ds_cfg.leave_unmasked_prob,
                    random_token_prob=ds_cfg.random_token_prob,
                    sample_conformer=ds_cfg.add_2d,
                )
            else:
                self.valid_dataset = None

    def pretrain(self):
        if self.config.model.model_name.lower() == "unimolv2":
            model = UniMolV2Model(self.config.model)
            loss_fn = UniMolV2Loss(
                padding_idx=0,
                masked_token_loss=self.config.model.masked_token_loss,
                masked_coord_loss=self.config.model.masked_coord_loss,
                masked_dist_loss=self.config.model.masked_dist_loss,
            )
        else:
            model = UniMolModel(self.config.model, dictionary=self.dictionary)
            loss_fn = UniMolLoss(
                padding_idx=self.dictionary.pad(),
                masked_token_loss=self.config.model.masked_token_loss,
                masked_coord_loss=self.config.model.masked_coord_loss,
                masked_dist_loss=self.config.model.masked_dist_loss,
                x_norm_loss=self.config.model.x_norm_loss,
                delta_pair_repr_norm_loss=self.config.model.delta_pair_repr_norm_loss,
            )
        trainer = UniMolPretrainTrainer(
            model,
            self.dataset,
            loss_fn,
            self.config.training,
            local_rank=self.local_rank,
            resume=self.config.training.get("resume", None),
            valid_dataset=self.valid_dataset,
        )
        logger.info("Starting pretraining")
        trainer.train(max_steps=self.config.training.total_steps)
        logger.info("Training finished. Checkpoints saved under the run directory.")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path=None, config_name="pretrain_config")
def main(cfg: DictConfig):
    task = MolPretrain(cfg)
    task.pretrain()

if __name__ == "__main__":
    main()