from datasets.TimeSeriesDataset import TimeSeriesDataset
import pytorch_lightning as L
from torch.utils.data import DataLoader
from typing import Optional
import torch
import numpy as np
import math

class TimeSeriesDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_dir: str = "/mnt/argo/Studies",
        window_timeserie: int = 100,
        samples_per_file: int = 20,
        subcortical: bool = True,
        batch_size: int = 32,
        num_workers: int = 20,
        ratio_train_val: float = .8,
        seed: int = 123,
        k_fold = None,
        rate_output: bool = False,
        identifiers_output: bool = False,
        scores_output: bool = False,
        notrandom = None,
        set = "TASK_FINA",
        pin_memory: bool = False
    ):
        super().__init__()
        self.base_dir = base_dir
        self.window_timeserie = window_timeserie
        self.samples_per_file = samples_per_file
        self.subcortical = subcortical
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio_train_val = ratio_train_val
        self.seed = seed
        self.rate_output = rate_output
        self.identifiers_output = identifiers_output
        self.scores_output = scores_output
        self.notrandom = notrandom
        self.pin_memory = pin_memory
        self.k = k_fold
        self.set = set
        
    def setup(self, stage = ""):
        if stage == "" or "fit" in stage:
            train_val_dataset = TimeSeriesDataset(
                base_dir=self.base_dir,
                window_timeserie=self.window_timeserie,
                samples_per_file=self.samples_per_file,
                subcortical=self.subcortical,
                rate_output=self.rate_output,
                seed=self.seed,
                scores_output=self.scores_output,
                identifiers_output=self.identifiers_output,
                set=self.set
            )
            if self.notrandom is None:
                # ID / n subjects in each set
                identifiers_dataset = np.array([train_val_dataset._extract_identifiers(i) for i in range(len(train_val_dataset))])
                identifiers = np.unique(identifiers_dataset, axis=0)
                if identifiers.shape[0] < 100:
                    raise ValueError(f"Expected at least 100 subjects, got {identifiers.shape[0]}")
                nsubject = identifiers.shape[0]

                # Val set
                rng = np.random.default_rng(seed=self.seed)
                if self.k is None:
                    shuffled_identifiers = identifiers
                    keep = rng.choice(nsubject, nsubject-(math.floor(nsubject*self.ratio_train_val)), replace=False)
                else:
                    shuffled_identifiers = rng.permutation(identifiers, axis=0)
                    keep = range(round(self.k*0.25*nsubject), round((self.k+1)*0.25*nsubject))

                val_identifiers = shuffled_identifiers[keep, :]
                val_identifiers = [i for i, v in enumerate(identifiers_dataset) if v[0] in val_identifiers[:, 0] and v[1] in val_identifiers[:, 1]]
                if len(val_identifiers) != round(nsubject*0.25)*self.samples_per_file and self.k is not None:
                    raise ValueError(f"Expected {nsubject}*0.25 subjects so *{self.samples_per_file} samples, got {len(val_identifiers)}")
                self.val_dataset = torch.utils.data.Subset(train_val_dataset, val_identifiers)

                # Train set (not in valid)
                notkeep = [i for i in range(nsubject) if i not in keep]
                train_identifiers = shuffled_identifiers[notkeep, :]
                train_identifiers = [i for i, v in enumerate(identifiers_dataset) if v[0] in train_identifiers[:, 0] and v[1] in train_identifiers[:, 1]]
                if len(train_identifiers) != round(nsubject*0.75)*self.samples_per_file and self.k is not None:
                    raise ValueError(f"Expected {nsubject}*0.75 subjects so *{self.samples_per_file} samples, got {len(train_identifiers)}")
                self.train_dataset = torch.utils.data.Subset(train_val_dataset, train_identifiers)

                # Verification
                print(f"{identifiers.shape[0]} = {len(train_identifiers)}+{len(val_identifiers)} / val {len(keep)} - train {len(notkeep)} / val {len(self.val_dataset)} - train {len(self.train_dataset)}")
            else:
                if len(self.notrandom) == 4:
                    id1, id2, train_id1, train_id2 = self.notrandom
                    notkeep = train_val_dataset._keep_in_list(train_id1, train_id2)
                    keep = train_val_dataset._keep_in_list(id1, id2)
                else:
                    id1, id2 = self.notrandom
                    keep = train_val_dataset._keep_in_list(id1, id2)
                    notkeep = [i for i in range(len(train_val_dataset)) if i not in keep]
                self.val_dataset = torch.utils.data.Subset(train_val_dataset, keep)
                self.train_dataset = torch.utils.data.Subset(train_val_dataset, notkeep)

        if stage == "" or "test" in stage:
            self.test_dataset = TimeSeriesDataset(
                base_dir=self.base_dir,
                window_timeserie=self.window_timeserie,
                samples_per_file=self.samples_per_file,
                subcortical=self.subcortical,
                seed=self.seed,
                rate_output=True,
                identifiers_output=True,
                scores_output=self.scores_output,
                set="TASK_RAW"
            )

    def _get_dataloader_kwargs(self):
        """Get common DataLoader kwargs"""
        return {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'persistent_workers': True if self.num_workers > 0 else False,
            'prefetch_factor': 2 if self.num_workers > 0 else None  # Prefetch 2 batches per worker
        }

    def train_dataloader(self):
        kwargs = self._get_dataloader_kwargs()
        kwargs['shuffle'] = True  # Enable shuffling for training
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self): # As a reminder, validation is testing in lightning convention
        kwargs = self._get_dataloader_kwargs()
        kwargs['shuffle'] = False  # No shuffling for validation
        return DataLoader(self.val_dataset, **kwargs)

    def test_dataloader(self): # As a reminder, testing is validation in lightning convention
        kwargs = self._get_dataloader_kwargs()
        kwargs['shuffle'] = False  # No shuffling for testing
        return DataLoader(self.test_dataset, **kwargs)

    def predict_dataloader(self):
        raise ValueError("Predict dataloader is not implemented in this module. Use test_dataloader instead.")
