"""Analyses of the paper used by ``results_article.ipynb`` and ``create_R_data.ipynb``
(classification metrics, rating heatmaps, SHAP network heatmaps).

``Visualization(checkpoint_path, name)`` loads cohorts 1 and 2 with three window
configurations (``type``):
    "training" : 100 random windows of 20 TR per subject (as during training)
    "block"    : the 34 task blocks (25 TR from each block onset) per subject
    "full"     : the whole run (1190 TR) per subject
and ``step``/``pop``: "train" / "val" (cohort 1) and "test" (cohort 2).

Predictions are cached in ``config.PREDICTIONS_DIR`` (``*.pt``). SHAP values computed by
``main_shap.py`` are read from ``config.SHAP_DIR``
(``shap-lstmv4vf_pop-val_batch-{0..7}.npz`` and ``shap-lstmv4vf_pop-test_batch-{0..29}.npz``).
Figures are written to ``results/<name>/predictions/{validation,test}/heatmap/``.
"""
from dataclasses import dataclass
import os
import warnings
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics import auc, confusion_matrix, roc_curve
import torch
import numpy as np
from config import PREDICTIONS_DIR, SHAP_DIR
from datasets.TimeSeriesDataModule import TimeSeriesDataModule
from models.lstm_v4 import LSTMClassifier
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd

@dataclass
class MetricConfig:
    names: List[str] = ('sensitivity', 'specificity', 'ppv', 'npv', 'lr_positive', 'lr_negative', 'auc')
    labels: List[str] = ('Sensitivity', 'Specificity', 'PPV', 'NPV', 'LR+', 'LR-', 'AUC')
    colors: List[str] = ('blue', 'green', 'red')

class Visualization:
    def __init__(self, checkpoint_path, name: str = "lstm", plot: bool = True):
        self.plot = plot
        self.checkpoint_path = checkpoint_path
        self.name = name
        self.subcortical = True
        self.conditions = ['Worry', 'Neutral', 'Reappraisal']
        self.contrasts = [
            (0, 1, 'worry_vs_neutral'),
            (0, 2, 'worry_vs_reapp'),
            (1, 2, 'neutral_vs_reapp')
        ]
        self.metric_config = MetricConfig()
        self._comparisons = [
            'Worry_vs_rest', 'Neutral_vs_rest', 'Reappraisal_vs_rest',
            'Worry_vs_Neutral', 'Worry_vs_Reappraisal', 'Neutral_vs_Reappraisal'
        ]
        self._labels = ['Worry', 'Neutral', 'Reappraisal']
        self.mae_loss = torch.nn.L1Loss()
        
        # Create output directories
        self._create_output_dirs()

        # Initialize data and model
        self._init_data_and_model()

        # Compute or load SHAP values
        self._init_shap_values()

        # Figure style of the paper (SciencePlots >= 2 registers its styles on import)
        try:
            import scienceplots  # noqa: F401
            plt.style.use(['science','nature','no-latex'])
        except (ImportError, OSError):
            warnings.warn("SciencePlots is not installed (pip install SciencePlots): "
                          "figures will use the default matplotlib style.")
        plt.rcParams.update({"font.size":12}) 

    def _get_predictions_filename(self, n_subjects: Optional[int] = None, step: str = "val", type: str = "full") -> str:
        """Generate a filename for predictions that reflects the configuration"""
        base_dir = PREDICTIONS_DIR
        os.makedirs(base_dir, exist_ok=True)
        
        # Build filename components
        model_info = Path(self.checkpoint_path).stem
        components = [
            model_info,
            f"sub-{str(self.subcortical).lower()}",
            f"type-{type}",
            f"step-{step}"
        ]
        components.append(f"n-{n_subjects}") if n_subjects is not None else components.append(f"n-all")
            
        filename = "_".join(components) + ".pt"
        return os.path.join(base_dir, filename)

    def _create_output_dirs(self):
        """Create necessary output directories"""
        # Create main results directory structure
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name)
        os.makedirs(f"{base_dir}/predictions/validation/heatmap", exist_ok=True)
        os.makedirs(f"{base_dir}/predictions/test/heatmap", exist_ok=True)

    def _init_data_and_model(self):
        """Initialize the data modules (full / training / block windows) and load the model"""
        # Load data
        self.full_dataset_trainvaltest = TimeSeriesDataModule(subcortical=self.subcortical, window_timeserie=1190, 
                                          samples_per_file=1, rate_output=True, identifiers_output=True, scores_output=True, batch_size=32)
        self.full_dataset_trainvaltest.setup("fit-test")

        self.std_dataset_trainvalpredtest = TimeSeriesDataModule(subcortical=self.subcortical, window_timeserie=20, 
                                          samples_per_file=100, rate_output=True, identifiers_output=True, scores_output=True, batch_size=32)
        self.std_dataset_trainvalpredtest.setup("")

        self.block_dataset_trainvaltest = TimeSeriesDataModule(subcortical=self.subcortical, window_timeserie=25, 
                                          samples_per_file=34, rate_output=True, identifiers_output=True, scores_output=True, batch_size=32)
        self.block_dataset_trainvaltest.setup("fit-test")

        # Load model and set to eval mode
        self.trained_model = LSTMClassifier.load_from_checkpoint(self.checkpoint_path)
        self.trained_model.eval()

    def _init_shap_values(self):
        """Load the SHAP values computed by main_shap.py (shape: blocks x 25 TR x 481 ROIs x 3 conditions)"""
        shap_vals, shap_vals_id1, shap_vals_id2 = [], [], []
        base_dir = SHAP_DIR
        for i in range(8):
            tmp = np.load(f"{base_dir}/shap-lstmv4vf_pop-val_batch-{i}.npz")
            shap_vals.append(tmp["shap"])
            shap_vals_id1.extend(tmp["id1"])
            shap_vals_id2.extend(tmp["id2"])
        self.shap_vals = np.vstack(shap_vals).reshape((-1, 25, 481, 3))
        self.shap_vals_id1 = np.array(shap_vals_id1).flatten()
        self.shap_vals_id2 = np.array(shap_vals_id2).flatten()

        shap_test, shap_test_id1, shap_test_id2 = [], [], []
        for i in range(30):
            tmp = np.load(f"{base_dir}/shap-lstmv4vf_pop-test_batch-{i}.npz")
            shap_test.append(tmp["shap"])
            shap_test_id1.extend(tmp["id1"])
            shap_test_id2.extend(tmp["id2"])
        self.shap_test = np.vstack(shap_test).reshape((-1, 25, 481, 3))
        self.shap_test_id1 = np.array(shap_test_id1).flatten()
        self.shap_test_id2 = np.array(shap_test_id2).flatten()

    def _get_predictions(self, step: str = "val", type: str = "full"):
        """Model predictions for a population (cached on disk after the first call).

        Args:
            step: "train", "val" (cohort 1) or "test" (cohort 2)
            type: "full", "block" or "training" window configuration

        Returns:
            [data, predictions, labels, id1, id2, rates, scores]
        """
        if type == "full":
            if step == "train":
                dataset = self.full_dataset_trainvaltest.train_dataloader()
            elif step == "val":
                dataset = self.full_dataset_trainvaltest.val_dataloader()
            elif step == "test":
                dataset = self.full_dataset_trainvaltest.test_dataloader()
        elif type == "block":
            if step == "train":
                dataset = self.block_dataset_trainvaltest.train_dataloader()
            elif step == "val":
                dataset = self.block_dataset_trainvaltest.val_dataloader()
            elif step == "test":
                dataset = self.block_dataset_trainvaltest.test_dataloader()
        elif type == "training":
            if step == "train":
                dataset = self.std_dataset_trainvalpredtest.train_dataloader()
            elif step == "val":
                dataset = self.std_dataset_trainvalpredtest.val_dataloader()
            elif step == "test":
                dataset = self.std_dataset_trainvalpredtest.test_dataloader()

        # Check if predictions are already saved to file
        predictions_file = self._get_predictions_filename(step=step, type=type)
        if os.path.exists(predictions_file):
            print(f"Loading predictions from {predictions_file}")
            return torch.load(predictions_file, weights_only=False)  # local file written below

        # Generate new predictions
        data, predictions, labels, idx1, idx2, rates, scores = [], [], [], [], [], [], []
        for batch in dataset:
            # batch = [features, labels, id1, id2, rate, scores]
            input = batch[0]
            labels.append(batch[1])
            rates.extend(batch[4])
            scores.extend(batch[-1])
            idx1.extend(batch[2])
            idx2.extend(batch[3])

            with torch.no_grad():
                input = input.clone().detach()
                data.append(input)
                y_hat = self.trained_model(input)
                predictions.append(y_hat)

        # Merge DATA YHAT Y ID1 ID2 RATE
        result = []
        result.append(torch.cat(data))
        result.append(torch.cat(predictions))
        result.append(torch.cat(labels))
        result.append(idx1)
        result.append(idx2)
        result.append(rates)
        result.append(scores)
            
        # Save predictions to file
        torch.save(result, predictions_file)
        return result

    def generate_heatmap(self, pop: str = "val", contours: int = 0, smoothing: bool = True, delta_rating: bool = False, only_concerned_block: bool = False):
        """
        Generate heatmaps for each condition.
        
        Args:
            contours: Number of contour lines (0 for no contours)
            smoothing: Whether to apply Gaussian smoothing
            delta_rating: If True, plot delta of rating between previous to next iteration
        """
        self.data, self.prediction, self.labels, self.id1, self.id2, self.rate, _ = self._get_predictions(pop, "block")
        self.id1, self.id2, self.rate = np.array(self.id1), np.array(self.id2), np.array(self.rate)
        shap_vals = np.mean(np.abs(self.shap_vals), axis=2) if pop == "val" else np.mean(np.abs(self.shap_test), axis=2)
        
        for label in range(3):
            matrix_pred_rate_time = np.zeros((6, 25))
            matrix_labels_rate_time = np.zeros((6, 25))
            matrix_aucs_rate_time = np.zeros((6, 25))
            matrix_shap_rate_time = np.zeros((6, 25))
            
            if delta_rating:
                delta_rates = np.diff(self.rate.reshape(-1, 34, 25)[:, ::-1], axis=1)[:, ::-1]
                delta_rates = np.pad(delta_rates, ((0, 0), (1, 0), (0, 0)), mode='constant').reshape(-1, 25)
                if delta_rates.shape[0] != self.rate.shape[0]:
                    raise ValueError(f"Bad conversion to diff {delta_rates.shape[0]} {self.rate.shape[0]}")
                
            for i in range(6):
                for j in range(25):
                    mask = np.abs(delta_rates[:, j]) == i if delta_rating else self.rate[:, j] == i
                    mask_new = np.argmax(softmax(self.labels[:, j, :].numpy(), axis=-1), axis=-1) == label
                    mask = np.logical_and(mask, mask_new) if only_concerned_block else mask
                    if np.count_nonzero(mask) > 0 or (i < 5 and delta_rating): # 5 do not exists as max=5 and min=1 so diff max is 5-1 so 4 and 0 as to stay NaN like value excluded of heatmap, 5 is possible if 5 before 0 of the cross fixation so not interesting
                        matrix_pred_rate_time[i, j] = np.mean(self.prediction[mask, j, label].numpy())
                        matrix_labels_rate_time[i, j] = np.mean(self.labels[mask, j, label].numpy())

                        y_true = (np.argmax(torch.nn.functional.softmax(self.labels[mask, j, :], dim=-1).numpy(), axis=-1) == label).astype(int).flatten()
                        y_pred_proba = torch.nn.functional.softmax(self.prediction[mask, j, :], dim=-1).numpy().reshape(-1, 3)[:, label].flatten()
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                        matrix_aucs_rate_time[i, j] = auc(fpr, tpr)
                        
                        matrix_shap_rate_time[i, j] = np.mean(np.abs(shap_vals[mask, j, label]))
            
            steps = ["Error", "Raw", "AUC", "Shap"]
            import matplotlib as mpl
            for step in steps:
                # Create smooth heatmap# Create figure and gridspec layout
                fig = plt.figure(figsize=(12, 10))

                # Create main axes for heatmap and marginal plots
                ax = fig.add_subplot()

                if step == "Shap":
                    data = matrix_shap_rate_time
                elif step == "AUC":
                    data = matrix_aucs_rate_time
                    cmap = plt.get_cmap('jet')
                    cmaplist = [cmap(i) for i in range(cmap.N)]
                    cmaplist[0] = (.5, .5, .5, 1.0)
                    bounds = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    cmap = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
                else:
                    data = matrix_pred_rate_time if step == "Raw" else (matrix_pred_rate_time - matrix_labels_rate_time)
                    data = data / (matrix_labels_rate_time + 1e-7)
                
                # Create main heatmap
                im = ax.imshow(data, origin='lower', aspect='auto',
                             norm=(norm if step == "AUC" else None),
                             interpolation=('gaussian' if smoothing else 'none'),
                             vmin=(None if step == "AUC" else 0),
                             vmax=((.2 if step == "Error" else 1) if step not in ["Shap", "AUC"] else None))
                
                # Add contour lines
                if contours > 0 and step != "AUC":
                    levels = np.linspace(np.min(data), np.max(data), contours)
                    _contours = ax.contour(np.arange(data.shape[1]), np.arange(data.shape[0]), 
                                         data, levels=levels, colors='black', alpha=0.5, 
                                         linewidths=1)
                    
                plt.title(f'{step} Heatmap - {self.conditions[label]}')
                if step != "AUC":
                    cbar = plt.colorbar(im, label=f'{step} Value', ax=ax)
                    if contours > 0:
                        cbar.add_lines(_contours)
                else:
                    cbar = plt.colorbar(im, label=f'{step} Value', ax=ax)
                
                plt.xlim(-.5, 23.5)
                plt.xticks(np.linspace(-.5, 23.5, 25), np.linspace(-10, 14, 25))
                plt.ylim(-.5, 4.5)
                plt.yticks(np.linspace(0, 4, 5), (np.linspace(0, 4, 5) if delta_rating else np.linspace(1, 5, 5)))
                plt.xlabel('Time (TR)')
                plt.ylabel("Diff Rating" if delta_rating else 'Rating')
                plt.tight_layout()
                
                suffix = "delta" if delta_rating else "value"
                pop_name = "validation" if pop=="val" else "test"
                base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", pop_name, "heatmap")
                plt.savefig(f'{base_dir}/heatmap_{suffix}-{step}_condition-{self.conditions[label]}.png')
                plt.close()
        return matrix_pred_rate_time,  matrix_labels_rate_time, matrix_aucs_rate_time, matrix_shap_rate_time

    def generate_network_heatmap(self, pop: str = "val", label: int = 0, interpolation: str = "none", only_concerned_block: bool = False, normalize_time: bool = False):
        """Heatmap (time x network) of the mean |SHAP| projected on the 18 network templates.

        Each block's SHAP map (481 ROIs) is L2-normalised, then projected on the L2-normalised
        templates of ``external/schaeffer_subcortical_smith_templates.csv``.
        """
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "external")
        network = pd.read_csv(f"{base_dir}/schaeffer_subcortical_smith_templates.csv")
        data, prediction, labels, id1, id2, rate, scores = self._get_predictions(pop, "block")
        id1, id2, rate = np.array(id1), np.array(id2), np.array(rate)

        weights = network.iloc[:, 1:].to_numpy()
        weights = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
        shap_vals = self.shap_vals if pop == "val" else self.shap_test
        mask = np.argmax(softmax(labels.numpy().sum(1), axis=-1), axis=-1) == label
        shap_vals = shap_vals[mask] if only_concerned_block else shap_vals
        nblocks = (16*int(label==0)+10*int(label==1)+8*int(label==2)) if only_concerned_block else 34
        shap_vals = shap_vals.reshape(-1, 25, 481, 3)

        x_norm = shap_vals / np.sqrt(np.sum(shap_vals ** 2, axis=2, keepdims=True))
        # Project each map on the templates along the ROI axis (x_norm.reshape(-1, 481) on an
        # array of shape (..., 481, 3) mixed ROIs and conditions)
        x_network = np.einsum("...rc,kr->...kc", x_norm, weights).reshape(-1, 25, 18, 3)
        x_network = ((x_network - x_network.min(axis=1, keepdims=True)) / (x_network.max(axis=1, keepdims=True) - x_network.min(axis=1, keepdims=True))) if normalize_time else x_network
        x_network = np.abs(x_network[:, :, :, label]).mean(axis=0)
        max_x = x_network.max()

        sort = np.argsort(x_network.mean(axis=0))
        #sort = np.flip(sort)

        x_network = x_network[:, sort]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot()

        # Create main heatmap
        im = ax.imshow(x_network.T, origin='lower', cmap="Reds", aspect='auto', vmax=max_x, vmin=0, interpolation=interpolation)
            
        plt.title(f'Shap Network Heatmap - {self.conditions[label]}')
        cbar = plt.colorbar(im, label=f'Shap Network Value', ax=ax)
        
        plt.xlim(-.5, 23.5)
        plt.xticks(np.linspace(-.5, 23.5, 25), np.linspace(-10, 14, 25))
        plt.yticks(np.linspace(0, 17, 18), network.iloc[sort, 0].tolist())
        plt.xlabel('Time (TR)')
        plt.ylabel("Shap Network")
        plt.tight_layout()
        pop_name = "validation" if pop=="val" else "test"
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", pop_name, "heatmap")
        plt.savefig(f'{base_dir}/heatmap_ShapNetwork_condition-{self.conditions[label]}.png')
        plt.close()

    def generate_diffnetwork_heatmap(self, pop: str = "val", slabels: list = [0,1], interpolation: str = "none", only_concerned_block: bool = False):
        """Difference of the network |SHAP| heatmaps of two conditions (``slabels``).

        SHAP values are reshaped to 5-D (subject, block, time, ROI, condition), so ``axis=2``
        below normalises each ROI time course over the 25 TR of the block (whereas
        ``generate_network_heatmap`` normalises each map over the ROIs).
        """
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "external")
        network = pd.read_csv(f"{base_dir}/schaeffer_subcortical_smith_templates.csv")
        data, prediction, labels, id1, id2, rate, scores = self._get_predictions(pop, "block")
        id1, id2, rate = np.array(id1), np.array(id2), np.array(rate)

        weights = network.iloc[:, 1:].to_numpy()
        weights = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
        shap_vals = self.shap_vals if pop == "val" else self.shap_test
        mask0 = np.argmax(softmax(labels.numpy().sum(1), axis=-1), axis=-1) == slabels[0]
        mask1 = np.argmax(softmax(labels.numpy().sum(1), axis=-1), axis=-1) == slabels[1]
        shap_vals = shap_vals[np.logical_or(mask0, mask1)] if only_concerned_block else shap_vals
        nblocks = (16*int(0 in slabels)+10*int(1 in slabels)+8*int(2 in slabels)) if only_concerned_block else 34
        shap_vals = shap_vals.reshape(-1, nblocks, 25, 481, 3)

        x_norm = shap_vals / np.sqrt(np.sum(shap_vals ** 2, axis=2, keepdims=True))
        # Project each map on the templates along the ROI axis (x_norm.reshape(-1, 481) on an
        # array of shape (..., 481, 3) mixed ROIs and conditions)
        x_network = np.einsum("...rc,kr->...kc", x_norm, weights).reshape(-1, 25, 18, 3)
        x_network_1 = np.abs(x_network[:, :, :, slabels[0]])
        x_network_2 = np.abs(x_network[:, :, :, slabels[1]])
        x_network = (x_network_1-x_network_2).mean(axis=0)
        max_x = max(x_network.max(), abs(x_network.min()))

        sort = np.argsort(np.abs(x_network).mean(axis=0))
        #sort = np.flip(sort)

        x_network = x_network[:, sort]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot()

        # Create main heatmap
        im = ax.imshow(x_network.T, origin='lower', aspect='auto', cmap="seismic", vmax=max_x, vmin=-max_x, interpolation=interpolation)
            
        plt.title(f'Shap Network Heatmap - Absolute {self.conditions[slabels[0]]}-{self.conditions[slabels[1]]}')
        cbar = plt.colorbar(im, label=f'Shap Network Value', ax=ax)
        
        plt.xlim(-.5, 23.5)
        plt.xticks(np.linspace(-.5, 23.5, 25), np.linspace(-10, 14, 25))
        plt.yticks(np.linspace(0, 17, 18), network.iloc[sort, 0].tolist())
        plt.xlabel('Time (TR)')
        plt.ylabel("Shap Network")
        plt.tight_layout()
        pop_name = "validation" if pop=="val" else "test"
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", pop_name, "heatmap")
        plt.savefig(f'{base_dir}/heatmap_DiffShapNetwork_condition-{self.conditions[slabels[0]]}-{self.conditions[slabels[1]]}.png')
        plt.close()

    # METRICS

    def loss_compute(self, population: str = "val", presentation: str = "block"):
        """Mean absolute error between predictions and rescaled task regressors."""
        _, prediction, labels, _, _, _, _ = self._get_predictions(population, presentation)
        return self.mae_loss(prediction, labels)

    def _calculate_binary_metrics(self, y_true: np.array, y_pred: np.array, 
                                y_pred_proba: np.array) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        sensitivity, specificity, ppv, npv, lr_positive, lr_negative, auc_score = 0, 0, 0, 0, 0, 0, 0
        if not (y_true == y_pred).all():
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            lr_positive = sensitivity / (1 - specificity) if (1 - specificity) > 0 else float('inf')
            lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
        
        result = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'lr_positive': lr_positive,
            'lr_negative': lr_negative,
            'auc': auc_score
        }
        return result

    def _calculate_one_vs_rest_metrics(self, y_true: np.array, y_pred: np.array, 
                                     label_idx: int) -> Dict[str, float]:
        """Calculate metrics for one-vs-rest comparison."""
        y_true_binary = (softmax(y_true, axis=-1).argmax(axis=-1) == label_idx).astype(np.int8)
        y_pred_proba = softmax(y_pred, axis=-1)[:, label_idx]
        y_pred_binary = (softmax(y_pred, axis=-1).argmax(axis=-1) == label_idx).astype(np.int8)
        
        return self._calculate_binary_metrics(y_true_binary, y_pred_binary, y_pred_proba)

    def _calculate_one_vs_one_metrics(self, y_true: np.array, y_pred: np.array, 
                                    idx1: int, idx2: int) -> Optional[Dict[str, float]]:
        """Calculate metrics for one-vs-one comparison."""
        mask = (softmax(y_true, axis=-1).argmax(axis=-1) == idx1) | (softmax(y_true, axis=-1).argmax(axis=-1) == idx2)
        if mask.sum() == 0:
            return None
        
        y_true_binary = (softmax(y_true, axis=-1)[mask].argmax(axis=-1) == idx1).astype(np.int8)
        y_pred_binary = (softmax(y_pred, axis=-1)[mask].argmax(axis=-1) == idx1).astype(np.int8)
        prob_ratio = softmax(y_pred[mask], axis=-1)[:, idx1]
        
        return self._calculate_binary_metrics(y_true_binary, y_pred_binary, prob_ratio)

    def _calculate_metrics_for_block(self, y_true: np.array, y_pred: np.array) -> Dict[str, Dict[str, float]]:
        """Calculate all metrics for a given block."""
        metrics = {}
        
        # One-vs-Rest metrics
        for i, label in enumerate(self._labels):
            metrics[f'{label}_vs_rest'] = self._calculate_one_vs_rest_metrics(y_true, y_pred, i)
        
        # One-vs-One metrics
        label_to_idx = {label: i for i, label in enumerate(self._labels)}
        pairs = [('Worry', 'Neutral'), ('Worry', 'Reappraisal'), ('Neutral', 'Reappraisal')]
        
        for label1, label2 in pairs:
            idx1, idx2 = label_to_idx[label1], label_to_idx[label2]
            result = self._calculate_one_vs_one_metrics(y_true, y_pred, idx1, idx2)
            metrics[f'{label1}_vs_{label2}'] = result if result is not None else {k: np.nan for k in self.metric_config.names}
        
        return metrics
        
    def _initialize_metric_arrays(self, population_subs: Dict[str, List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Initialize arrays for storing metric values."""
        return {
            pop_name: {k: np.zeros(len(list(set(population_sub)))).astype(np.float64) for k in self.metric_config.names}
            for pop_name, population_sub in population_subs.items() if len(list(set(population_sub))) > 0
        }

    def _calculate_population_metrics(self, y: np.array, y_hat: np.array, 
                                    population_ids: Dict[str, List[int]], population_subs: Dict[str, List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate metrics for each population."""
        metric_values = [self._initialize_metric_arrays(population_subs) for i in range(len(self._comparisons))]
        for pop_name, pop_ids in population_ids.items():
            metrics = self._calculate_metrics_for_block(y[pop_ids, :, :].reshape(-1, 3), y_hat[pop_ids, :, :].reshape(-1, 3))
            for i, comparison in enumerate(self._comparisons):
                for metric_name in self.metric_config.names:
                    metric_values[i][pop_name][metric_name] = round(metrics[comparison][metric_name], 2)
                metric_values[i][pop_name]["mae"] = round(self.mae_loss(y_hat[pop_ids, :, :].reshape(-1, 3), y[pop_ids, :, :].reshape(-1, 3)).item(), 2)
        return metric_values

    def visualize_roc_metrics_rates(self, type: str = "block", threshold: int = 0, asc = 0, delta: bool = True) -> list:
        """Classification metrics (AUC, sensitivity, specificity, PPV, NPV, LR+, LR-, MAE) per population.

        Metrics pool every time point of the selected windows. Populations: "validation"
        (cohort 1), "test" (cohort 2), and cohort 2 split by participation in cohort 1
        ("raw_in_training", "raw_in_validation", "raw_not_in_training", see common_file.csv).

        Args:
            type: window configuration ("block", "full" or "training")
            threshold, asc, delta: with type="block", keep the blocks whose median rating
                (delta=False) or median absolute rating change from the previous block
                (delta=True) is <= threshold (asc=0), >= threshold (asc=1) or == threshold (asc=2)

        Returns:
            One dict per comparison of ``self._comparisons``: {population: {metric: value}}
        """
        # Get predictions and process data
        batchTr = self._get_predictions("train", type)
        _, y_hatTr, yTr, id1Tr, _, ratesTr, mTr = batchTr
        batchV = self._get_predictions("val", type)
        _, y_hatV, yV, id1V, _, ratesV, mV = batchV
        batchT = self._get_predictions("test", type)
        _, y_hatT, yT, id1T, _, ratesT, mT = batchT
        y_hat = torch.cat((y_hatTr, y_hatV, y_hatT), dim=0)
        y = torch.cat((yTr, yV, yT), dim=0)
        rates = np.concatenate((np.array(ratesTr), np.array(ratesV), np.array(ratesT)), axis=0)
        id1 = np.concatenate((np.array(id1Tr), np.array(id1V), np.array(id1T)), axis=0) if type == "block" else np.concatenate((np.array(id1Tr), np.array(id1V), np.array(id1T)), axis=0)
        
        # Separate populations
        dataset = self.block_dataset_trainvaltest.test_dataset
        
        raw_in_training_ids = np.concatenate([np.where(np.array(id1T) == dataset.is_fina_in_raw(id))[0] for id in np.unique(id1Tr)])
        raw_in_training_ids = raw_in_training_ids.flatten() if raw_in_training_ids.size > 0 else np.array([], dtype=np.int64)  # int: NumPy refuses empty float index arrays

        raw_in_validation_ids = np.concatenate([np.where(np.array(id1T) == dataset.is_fina_in_raw(id))[0] for id in np.unique(id1V)])
        raw_in_validation_ids = raw_in_validation_ids.flatten() if raw_in_validation_ids.size > 0 else np.array([], dtype=np.int64)
        
        mask = np.ones(len(id1T), dtype=bool)
        mask[raw_in_training_ids] = False
        raw_not_in_training_ids = np.where(mask)[0]
        raw_not_in_training_ids = raw_not_in_training_ids.flatten() if raw_not_in_training_ids.size > 0 else np.array([], dtype=np.int64)

        population_ids = {
            'raw_not_in_training': np.array(len(id1Tr)+len(id1V)+raw_not_in_training_ids).astype(np.int16),
            'raw_in_training': np.array(len(id1Tr)+len(id1V)+raw_in_training_ids).astype(np.int16),
            'raw_in_validation': np.array(len(id1Tr)+len(id1V)+raw_in_validation_ids).astype(np.int16),
            'validation': np.arange(len(id1Tr), len(id1Tr)+len(id1V)).astype(np.int16),
            'test': np.arange(len(id1V)+len(id1Tr), len(id1)).astype(np.int16),
        }

        print([value.shape[0] for pop, value in population_ids.items()])

        # Ref delta or ratings
        population_subs = {}
        if type == "block":
            if delta:
                ref = np.abs(np.diff(rates.reshape(-1, 34, 25)[:, ::-1], axis=1)[:, ::-1])
                ref = np.pad(ref, ((0, 0), (1, 0), (0, 0)), mode='constant').reshape(-1, 25)
            else:
                ref = rates.reshape(-1, 25)

            # Apply threshold
            ref = np.median(ref, axis=-1)
            if asc == 1 or asc is True:
                mask = ref >= threshold
            elif asc == 0 or asc is False:
                mask = ref <= threshold
            else:
                mask = ref == threshold

            for pop, values in population_ids.items():
                mask_alt = mask[values]
                population_ids[pop] = values[mask_alt].flatten()
                population_subs[pop] = id1[values].flatten()
        else:
            for pop, values in population_ids.items():
                population_subs[pop] = id1[values].flatten()

        # Calculate metrics
        metric_values = self._calculate_population_metrics(y, y_hat, population_ids, population_subs)

        return metric_values

