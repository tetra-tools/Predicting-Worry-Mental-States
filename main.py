"""Train the LSTM classifier of worry-related mental states.

Cohort 1 (FINA, N=116) is split at the subject level into a training set (80 %)
and a validation set (20 %); the checkpoint with the lowest validation L1 loss
is kept. Cohort 2 (RAW, N=88) is never used here: it is the independent test
set evaluated in ``visualization/``.

Example (settings of the published model; they are the defaults)::

    WANDB_MODE=offline python main.py --model lstm_v4

Training was run on CPU (20 cores, 64 GB RAM) on the University of Pittsburgh
CRC cluster, logging to Weights & Biases.
"""
import os
import random
import socket
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import DATA_ROOT
from datasets.TimeSeriesDataModule import TimeSeriesDataModule
from models.lstm_v4 import LSTMClassifier as LSTMClassifierv4


def set_seed(seed):
    """Set seed for reproducibility (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_next_version():
    """Return the next free version number in ``checkpoints/`` (1 if empty)."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return 1
    existing_versions = [int(d.split('_version')[-1]) for d in os.listdir(checkpoint_dir)
                         if os.path.isdir(os.path.join(checkpoint_dir, d)) and 'version' in d]
    return max(existing_versions + [0]) + 1


def main(args):
    # Number of CPU threads used by PyTorch
    if not args.use_gpu:
        torch.set_num_threads(20)

    # Seed Python / NumPy / PyTorch. NB: the random windows drawn by the dataset use
    # NumPy's global generator, so this seed also fixes the training samples.
    if args.seed is not None:
        set_seed(args.seed)

    # Data: cohort 1 split into train/validation by subject (see TimeSeriesDataModule)
    datamodule = TimeSeriesDataModule(
        base_dir=DATA_ROOT,
        window_timeserie=args.window_size,  # Window length (TR)
        samples_per_file=args.samples_per_subject,  # Random windows drawn per subject
        subcortical=True,  # Always use the 400 Schaefer + 81 AAL3 subcortical ROIs (481 inputs)
        num_workers=20 if not args.use_gpu else 4,
        batch_size=args.batch_size,
        k_fold=args.k_fold,  # None = single 80/20 split (published model)
        scores_output=("lstm_v4b" in args.model),  # Always False for lstm_v4
        seed=args.seed if args.seed is not None else 42  # Use provided seed or default
    )
    datamodule.setup()

    # Logger (run name: <model>_host-<hostname>_version<n>)
    version = get_next_version()
    hostname = socket.gethostname()
    model_name = f'{args.model}_host-{hostname}_version{version}'
    logger = WandbLogger(
        project='DeepNetfMRIWorry',
        name=model_name,
        log_model=True,
        config={
            'learning_rate': args.learning_rate,
            'dropout_rate': args.dropout_rate,
            'weight_decay': args.weight_decay,
            'gradient_clip_val': args.gradient_clip_val,
            'batch_size': args.batch_size,
            'noise_min': args.noise_min,
            'noise_max': args.noise_max,
            'noise_duration': args.noise_duration,
            'noise_scheduler': args.noise_scheduler,
            'cosine_annealing_T0': args.cosine_annealing_T0,
            'scheduler': args.scheduler,
            'cosine_annealing_T_mult': args.cosine_annealing_T_mult,
            'seed': args.seed,
            'window_size': args.window_size,
            'samples_per_subject': args.samples_per_subject,
            'cpu_cores': 20,
            'device': 'cpu',
            'precision': 32
        }
    )
    # Initialize wandb for graph logging (set WANDB_MODE=offline to run without an account)
    wandb.init(project='DeepNetfMRIWorry', name=model_name, dir="./wandb", tags=["lstm_v4"])

    # Keep only the checkpoint with the lowest validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{model_name}",
        filename="model_best_score={val_loss:.4f}",
        monitor='val_loss',
        mode='min',
        enable_version_counter=False,
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=50,
        mode='min',
        min_delta=0.0001,
        check_finite=True,
    )

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]

    # Model
    if args.model == 'lstm_v4':
        model = LSTMClassifierv4(
            input_channels=481,  # 400 Schaefer cortical + 81 AAL3 subcortical ROIs
            hidden_size=256,
            learning_rate=args.learning_rate,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            gradient_clip_val=args.gradient_clip_val,
            noise_min=args.noise_min,
            noise_max=args.noise_max,
            noise_duration=args.noise_duration,
            scheduler=args.scheduler,
            cosine_annealing_T0=args.cosine_annealing_T0,
            cosine_annealing_T_mult=args.cosine_annealing_T_mult,
            noise_scheduler=args.noise_scheduler,
            logger_type=args.logger,
        )
    else:
        raise ValueError("Invalid model type. Choose 'lstm_v4'.")

    # Trainer (CPU, fp32, as in the paper)
    trainer = pl.Trainer(
        max_epochs=250,
        accelerator='cpu',
        devices=1,
        precision=32,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=callbacks,
        deterministic=True if args.seed is not None else False,
        gradient_clip_val=args.gradient_clip_val,
        strategy="auto"
    )

    trainer.fit(model, datamodule)

    wandb.finish()


def build_parser():
    parser = ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--model', type=str, default='lstm_v4', choices=['lstm_v4'])
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--gradient_clip_val', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--noise_min', type=float, default=0.2, help='Final (minimum) input-mixing noise level')
    parser.add_argument('--noise_max', type=float, default=0.3, help='Initial (maximum) input-mixing noise level')
    parser.add_argument('--noise_duration', type=int, default=200, help='Epochs to decay noise from max to min')
    parser.add_argument('--cosine_annealing_T0', type=int, default=5, help='T0 (only with --scheduler cosine)')
    parser.add_argument('--cosine_annealing_T_mult', type=int, default=1, help='T_mult (only with --scheduler cosine)')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'])
    parser.add_argument('--noise_scheduler', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--window_size', type=int, default=20, help='Window length in TR')
    parser.add_argument('--samples_per_subject', type=int, default=100, help='Random windows per subject')
    parser.add_argument('--k_fold', type=int, default=None, choices=[0, 1, 2, 3],
                        help='Fold of a 4-fold subject split; default None = single 80/20 split (published model)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Only lowers CPU threads/workers; the Trainer always runs on CPU as in the paper')
    parser.add_argument('--logger', type=str, default='wandb', choices=['wandb', 'tensorboard'],
                        help='Stored in the model hyper-parameters; logging always uses Weights & Biases')
    parser.add_argument('--pretrained_path', type=str, help='Unused, kept for backward compatibility')
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
