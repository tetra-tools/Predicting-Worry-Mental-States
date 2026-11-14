"""Compute SHAP (DeepExplainer) attributions of a trained model on task blocks.

Inputs are the 34 task blocks of each participant (windows of 25 TR starting at each
block onset). The explained output is the prediction at the **last time point of the
window** (3 conditions); the attributions cover every input (25 TR x 481 ROIs).

The background distribution is the whole training set (cohort 1, training subjects,
same block windows). The validation subjects are first recovered from the training
configuration (windows of 20 TR, 100 samples per subject) so that the split is the
one of the trained model.

One run processes one batch of 102 blocks (3 participants x 34 blocks) and writes
``shap-lstmv4vf_pop-<population>_batch-<batch_idx>.npz`` (``shap``: (102, 25, 481, 3),
``id1``/``id2``: identifiers) in the current directory. The visualization code expects
these files in ``results/<name>/predictions/shap/``:
    --population val  : batches 0-7  (24 validation participants)
    --population test : batches 0-29 (88 participants of cohort 2)

Memory: about 15 GB per batch with the full background (run on 64 GB nodes).

Example::

    python main_shap.py --checkpoint checkpoints/<run>/model_best_score=val_loss=0.1113.ckpt --batch_idx 0
"""
import os
import random
from argparse import ArgumentParser

import numpy as np
import shap
import torch
from tqdm import tqdm

from config import DATA_ROOT
from datasets.TimeSeriesDataModule import TimeSeriesDataModule
from models.lstm_v4 import LSTMClassifier as LSTMClassifierv4


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class ModelWrapper(torch.nn.Module):
    """Expose only the prediction at the last time point: (batch, 3)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        result = self.model(x)
        return result[:, -1, :].squeeze()


def main(args):
    # Number of CPU threads used by PyTorch
    torch.set_num_threads(20)

    if args.seed is not None:
        set_seed(args.seed)

    # 1) Recover the validation subjects of the training configuration (20 TR x 100 windows)
    datamodule = TimeSeriesDataModule(
        base_dir=DATA_ROOT,
        window_timeserie=20,  # Window size used for training
        samples_per_file=100,  # Samples per subject used for training
        subcortical=True,  # 481 ROIs
        num_workers=20,
        batch_size=32,
        identifiers_output=True,
        seed=args.seed if args.seed is not None else 123  # Use provided seed or default
    )
    datamodule.setup("fit")
    datamodule = datamodule.val_dataloader()
    idx1, idx2 = [], []
    for batch in datamodule:
        _, _, id1, id2 = batch
        idx1.extend(id1)
        idx2.extend(id2)

    # 2) Same split, one window of 25 TR per task block (34 blocks per subject)
    datamodule = TimeSeriesDataModule(
        base_dir=DATA_ROOT,
        window_timeserie=25,  # One task block
        samples_per_file=34,  # 34 blocks per subject
        subcortical=True,  # 481 ROIs
        num_workers=40,
        batch_size=34*3,  # 3 subjects per batch
        identifiers_output=True,
        seed=args.seed if args.seed is not None else 123,  # Use provided seed or default
        notrandom=[idx1, idx2]  # Validation subjects found above
    )
    datamodule.setup("fit-test")
    print(f"Blocks: train {len(datamodule.train_dataset)} / val {len(datamodule.val_dataset)} / test {len(datamodule.test_dataset)}")

    # Background distribution = all training blocks
    all_batches = []
    for batch in datamodule.train_dataloader():
        x, _, _, _ = batch
        all_batches.append(x)
    background_data = torch.cat(all_batches)

    model = LSTMClassifierv4.load_from_checkpoint(args.checkpoint)
    model.eval()
    wmodel = ModelWrapper(model)

    # SHAP
    explainer = shap.DeepExplainer(wmodel, background_data)
    loader = datamodule.test_dataloader() if args.population == "test" else datamodule.val_dataloader()
    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing SHAP")):
        if batch_idx == args.batch_idx:
            tqdm.write("Step 1: Get batch...")
            # test batches: x, y, id1, id2, rate / val batches: x, y, id1, id2
            x, id1, id2 = batch[0], batch[2], batch[3]
            x.requires_grad = True
            tqdm.write("Step 2: Computing SHAP values...")
            shap_vals = explainer.shap_values(x, ranked_outputs=None, check_additivity=False)
            tqdm.write("Step 3: Compressing results...")
            np.savez_compressed(f"shap-lstmv4vf_pop-{args.population}_batch-{batch_idx}.npz", shap=shap_vals, id1=id1, id2=id2)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--checkpoint', default="./checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/model_best_score=val_loss=0.1113.ckpt", type=str,
                        help='Checkpoint of the trained model (default: published model)')
    parser.add_argument('--batch_idx', type=int, default=1, help='Index of the batch of 102 blocks to explain')
    parser.add_argument('--population', type=str, default='test', choices=['test', 'val'],
                        help='test = cohort 2 (default), val = validation subjects of cohort 1')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    # Unused: kept so that existing job scripts keep working
    for name, kind, default in [('learning_rate', float, 0.0005), ('dropout_rate', float, 0.1), ('weight_decay', float, 0.2),
                                ('gradient_clip_val', float, 0.8), ('hidden_size', int, 256), ('noise_min', float, 0),
                                ('noise_max', float, .5), ('noise_duration', int, 200), ('cosine_annealing_T0', int, 5),
                                ('cosine_annealing_T_mult', int, 1)]:
        parser.add_argument(f'--{name}', type=kind, default=default, help='Unused')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine'], help='Unused')
    parser.add_argument('--noise_scheduler', type=str, default='linear', choices=['linear', 'quadratic'], help='Unused')

    args = parser.parse_args()
    main(args)
