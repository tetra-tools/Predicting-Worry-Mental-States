"""LSTM classifier of worry-related mental states (model ``lstm_v4`` of the paper).

Input : regional fMRI time series, shape ``(batch, time, 481)``
        (400 Schaefer-2018 cortical ROIs + 81 AAL3 subcortical/cerebellar ROIs).
Output: one value in [0, 1] per time point and per condition, shape ``(batch, time, 3)``,
        in the order (Worry, Neutral, Reappraisal). Targets are the task regressors
        rescaled to [0, 1] (see ``TimeSeriesDataset``) and the loss is the L1 distance.

Architecture
    input_processing : Linear 481->512, ReLU, Dropout(p/2), LayerNorm, Linear 512->256, ReLU
    lstm_layers      : LSTMBlock(256->256) then LSTMBlock(256->128), each a bidirectional
                       LSTM + Linear/ReLU/Dropout/LayerNorm with a residual projection
    output           : Linear 128->64, ReLU, Dropout(p/4), LayerNorm, Linear 64->3,
                       then (tanh(x) + 1) / 2 clamped to [1e-7, 1 - 1e-7]
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
import numpy as np

class LSTMBlock(nn.Module):
    """Bidirectional LSTM followed by a projection to ``hidden_size`` with a residual connection."""

    def __init__(self, input_size, hidden_size, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.post_lstm = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size)
        )
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
    
    def forward(self, x):
        identity = self.residual(x)
        pred, _ = self.lstm(x)
        return self.post_lstm(pred) + identity

class LSTMClassifier(pl.LightningModule):
    """Sequence-to-sequence classifier of Worry / Neutral / Reappraisal states.

    Args:
        input_channels: Number of ROIs (481 with subcortical regions).
        hidden_size: Width of the first LSTM block (the second one uses ``hidden_size // 2``).
        learning_rate, weight_decay: AdamW parameters.
        dropout_rate: Dropout of the LSTM blocks (``/2`` at the input, ``/4`` at the output).
        gradient_clip_val: Stored for logging; clipping is applied by the Trainer.
        noise_min, noise_max, noise_duration, noise_scheduler: Input-mixing augmentation,
            see :meth:`get_noise_level` and :meth:`training_step`.
        scheduler: ``'plateau'`` (ReduceLROnPlateau, published model) or ``'cosine'``.
        cosine_annealing_T0, cosine_annealing_T_mult: Only used with ``scheduler='cosine'``.
        logger_type: Stored in the hyper-parameters only.
    """

    def __init__(
        self,
        input_channels: int = 481, 
        hidden_size: int = 256,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.1,
        weight_decay: float = 0.01,
        gradient_clip_val: float = 1.0,
        noise_min: float = 0.05,
        noise_max: float = 0.3,
        noise_duration: int = 200,
        scheduler: str = 'cosine',
        noise_scheduler: str = 'linear',
        cosine_annealing_T0: int = 5,
        cosine_annealing_T_mult: int = 1,
        logger_type: str = 'tensorboard',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_processing = nn.Sequential(
            nn.Linear(input_channels, hidden_size*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.LayerNorm(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
        )
        
        self.lstm_layers = nn.Sequential(
            LSTMBlock(hidden_size, hidden_size, dropout_rate),
            LSTMBlock(hidden_size, hidden_size//2, dropout_rate)
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Dropout(dropout_rate/4),
            nn.LayerNorm(hidden_size//4),
            nn.Linear(hidden_size//4, 3)
        )
        
        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.logger_type = logger_type
        self.min_noise = noise_min
        self.max_noise = noise_max
        self.noise_duration = noise_duration
        self.warmup_epochs = 2
        self.scheduler = scheduler
        self.cosine_annealing_T0 = cosine_annealing_T0
        self.cosine_annealing_T_mult = cosine_annealing_T_mult
        self.noise_scheduler = noise_scheduler
        self.best_val_loss = 10
        self.history_val_loss = []
        self.all_val_loss = []

        self._init_weights()

    def _init_weights(self):
        """Xavier (input-hidden) and orthogonal (hidden-hidden) init of the LSTM weights.

        NB: the ``'linear'`` branch never matches because parameter names are e.g.
        ``input_processing.0.weight`` (no ``'linear'`` in them), so the Linear layers keep
        PyTorch's default initialisation. Left unchanged so that results are reproducible.
        """
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data, gain=0.8)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data, gain=0.8)
            elif 'linear' in name and 'weight' in name:
                torch.nn.init.xavier_normal_(param.data, gain=0.7)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Features projection
        pred = self.input_processing(x)

        # Sequence learning
        pred = self.lstm_layers(pred)
        
        # Output prediction
        pred = self.output(pred)
        pred = (F.tanh(pred) + 1) / 2
        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        
        return pred
    
    def get_noise_level(self):
        """Augmentation strength for the current epoch.

        ``noise_max`` during the first ``warmup_epochs`` (2) epochs, then decreases
        (linearly or quadratically) to ``noise_min`` over ``noise_duration`` epochs.
        """
        # Keep maximum noise during warmup
        if self.current_epoch < self.warmup_epochs:
            return self.max_noise

        # Progressive decrease after warmup
        progress = (self.current_epoch - self.warmup_epochs) / self.noise_duration
        progress = progress**2 if self.noise_scheduler == 'quadratic' else progress
        noise = self.max_noise - (self.max_noise - self.min_noise) * min(progress, 1.0)
        return noise

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        # Data augmentation: mix each window with another window of the batch (random
        # permutation) weighted by the noise level, then renormalise:
        #   x <- (x + level * x[perm]) / (1 + level)
        # Labels are not mixed.
        if self.training:
            noise = x[torch.randperm(x.size(0), device=x.device)] * self.get_noise_level()
            x = (x + noise) / (1 + self.get_noise_level())

        y_hat = self(x)
        batch_size = x.size(0)
        
        l1_loss = F.l1_loss(
            y_hat.view(batch_size, -1, 3),
            y.view(batch_size, -1, 3),
            reduction='mean'
        )
        loss = l1_loss
        
        # Logging        
        self.log('noise_level', self.get_noise_level(), on_epoch=True)
        self.log('train_accuracy', self.accuracy(y_hat.view(batch_size, -1, 3), y.view(batch_size, -1, 3)), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        batch_size = x.size(0)
        
        l1_loss = F.l1_loss(
            y_hat.view(batch_size, -1, 3),
            y.view(batch_size, -1, 3),
            reduction='mean'
        )
        val_loss = l1_loss
        self.history_val_loss.append(val_loss)
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', self.accuracy(y_hat.view(batch_size, -1, 3), y.view(batch_size, -1, 3)), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return y_hat

    def on_validation_epoch_end(self):
        # NB: also runs after Lightning's sanity check, whose loss is included in
        # ``best_val_loss`` / ``mean_val_loss`` (logging only; checkpointing uses ``val_loss``).
        avg_loss = torch.stack(self.history_val_loss).mean()
        self.best_val_loss = min(self.best_val_loss, avg_loss)
        self.history_val_loss = []
        self.all_val_loss = self.all_val_loss + [avg_loss]
        self.all_val_loss = self.all_val_loss[-20:]
        self.log('mean_val_loss', sum(self.all_val_loss)/len(self.all_val_loss), prog_bar=True, sync_dist=True)
        self.log('best_val_loss', self.best_val_loss, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        batch_size = x.size(0)
        
        l1_loss = F.l1_loss(
            y_hat.view(batch_size, -1, 3),
            y.view(batch_size, -1, 3),
            reduction='mean'
        )
        test_loss = l1_loss
        
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_accuracy', self.accuracy(y_hat.view(batch_size, -1, 3), y.view(batch_size, -1, 3)), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return y_hat

    def accuracy(self, y_hat, y):
        """Per-time-point accuracy: recall of each condition weighted by its share of task blocks.

        The class of a time point is the argmax over the 3 conditions (16 Worry, 10 Neutral
        and 8 Reappraisal blocks out of 34). Logged for monitoring only.
        """
        y_pred_proba = torch.nn.functional.softmax(y_hat, dim=-1).detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        # Get predicted classes
        y_pred_classes = np.argmax(y_pred_proba, axis=-1)
        y_classes = np.argmax(y, axis=-1)
        
        # Calculate weighted global accuracy
        class_weights = np.array([16/34, 10/34, 8/34])
        class_accuracies = np.array([
            ((y_classes == y_pred_classes) & (y_pred_classes == i)).sum()/(y_classes == i).sum() for i in range(len(class_weights))
        ])
        return np.sum(class_accuracies * class_weights)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        if self.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.4,
                patience=5,
                min_lr=5e-5,
                threshold=0.0001,
            )
        elif self.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cosine_annealing_T0,
                T_mult=self.cosine_annealing_T_mult,
                eta_min=1e-4
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        }
