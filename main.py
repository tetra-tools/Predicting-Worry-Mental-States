import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import  WandbLogger
import wandb
from argparse import ArgumentParser
from datasets.TimeSeriesDataModule import TimeSeriesDataModule
from models.lstm_v4 import LSTMClassifier as LSTMClassifierv4
import random
import numpy as np

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_next_version():
    """Get next version number for model checkpoints"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return 1
    existing_versions = [int(d.split('_version')[-1]) for d in os.listdir(checkpoint_dir) 
                        if os.path.isdir(os.path.join(checkpoint_dir, d)) and 'version' in d]
    return max(existing_versions + [0]) + 1

def main(args):
    # Set number of CPU cores
    if not args.use_gpu:
        torch.set_num_threads(20)
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    # Initialize data module with fixed settings
    datamodule = TimeSeriesDataModule(
        base_dir="/mnt/argo/Studies",
        window_timeserie=args.window_size,  # Fixed window size
        samples_per_file=args.samples_per_subject,  # Fixed samples per subject
        subcortical=True,  # Always use subcortical
        num_workers=20 if not args.use_gpu else 4,  # Reduce workers for GPU
        batch_size=args.batch_size,
        k_fold=args.k_fold,
        scores_output=("lstm_v4b" in args.model),
        seed=args.seed if args.seed is not None else 42  # Use provided seed or default
    )
    datamodule.setup()
    
    # Initialize logger based on argument
    version = get_next_version()
    import socket
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
    # Initialize wandb for graph logging
    wandb.init(project='DeepNetfMRIWorry', name=model_name, dir="./wandb", tags=["lstm_v4"])
    
    # Initialize callbacks
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

    # Combine all callbacks
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Initialize model
    if(args.model == 'lstm_v4'):
        model = LSTMClassifierv4(
            input_channels=481,  # Fixed for subcortical
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

    # Initialize trainer with GPU support if requested
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
    
    # Train model
    trainer.fit(model, datamodule)
    
    # Close wandb run if using wandb
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm_v4b', choices=['lstm_v4'])
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--gradient_clip_val', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--noise_min', type=float, default=0.2, help='Noise min')
    parser.add_argument('--noise_max', type=float, default=0.3, help='Noise max')
    parser.add_argument('--noise_duration', type=int, default=200, help='Noise duration')
    parser.add_argument('--cosine_annealing_T0', type=int, default=5, help='T0')
    parser.add_argument('--cosine_annealing_T_mult', type=int, default=1, help='Multiplier')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'])
    parser.add_argument('--noise_scheduler', type=str, default='linear', choices=['linear', 'quadratic'])
    parser.add_argument('--window_size', type=int, default=20)
    parser.add_argument('--samples_per_subject', type=int, default=100)
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained lstm_v4 checkpoint')
    
    args = parser.parse_args()
    main(args)
