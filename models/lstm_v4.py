import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
import numpy as np

class LSTMBlock(nn.Module):
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
        # Keep minimum noise during warmup
        if self.current_epoch < self.warmup_epochs:
            return self.max_noise
            
        # Progressive increase after warmup
        progress = (self.current_epoch - self.warmup_epochs) / self.noise_duration
        progress = progress**2 if self.noise_scheduler == 'quadratic' else progress
        noise = self.max_noise - (self.max_noise - self.min_noise) * min(progress, 1.0)
        return noise

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch

        # Add fixed Gaussian noise during training
        if self.training:
            # Generate noise on same device as input
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
        y_pred_proba = torch.nn.functional.softmax(y_hat, dim=-1).detach().numpy()
        y = y.detach().numpy()
        
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
