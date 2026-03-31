from dataclasses import dataclass
import os
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.isotonic import spearmanr
from sklearn.metrics import auc, confusion_matrix, roc_curve
import torch
import numpy as np
from datasets.TimeSeriesDataModule import TimeSeriesDataModule
from models.lstm_v4 import LSTMClassifier
from nilearn import datasets, image
from utils.get_aal import get_aal
import nibabel as nib
from scipy.stats import zscore
from typing import List, Optional, Tuple
from pathlib import Path
import pandas as pd

from typing import Dict, Any

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
        self.step = ""
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
        self._get_atlas()

        # Compute or load SHAP values
        self._init_shap_values()

        plt.style.use(['science','nature','no-latex'])
        plt.rcParams.update({"font.size":12}) 

    def _get_predictions_filename(self, n_subjects: Optional[int] = None, step: str = "pred", type: str = "full") -> str:
        """Generate a filename for predictions that reflects the configuration"""
        base_dir = f"results/{self.name}/predictions"
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", base_dir)
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
        os.makedirs(f"{base_dir}/predictions/compare", exist_ok=True)
        os.makedirs(f"{base_dir}/predictions/validation", exist_ok=True)
        os.makedirs(f"{base_dir}/predictions/restingstate", exist_ok=True)
        os.makedirs(f"{base_dir}/predictions/test", exist_ok=True)
        os.makedirs(f"{base_dir}/predictions/shap", exist_ok=True)
        os.makedirs(f"{base_dir}/maps/nifti", exist_ok=True)
        os.makedirs(f"{base_dir}/maps/surfice", exist_ok=True)
        os.makedirs(f"{base_dir}/maps/plot", exist_ok=True)

    def _init_data_and_model(self):
        """Initialize dataset, model, and prepare data"""
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

        self.full_dataset_pred = TimeSeriesDataModule(subcortical=self.subcortical, window_timeserie=480, 
                                          samples_per_file=1, identifiers_output=True, batch_size=32, scores_output=True)
        self.full_dataset_pred.setup("pred")

        # Load model and set to eval mode
        self.trained_model = LSTMClassifier.load_from_checkpoint(self.checkpoint_path)
        self.trained_model.eval()

    def _init_shap_values(self):
        """Initialize or load SHAP values"""
        shap_vals, shap_vals_id1, shap_vals_id2 = [], [], []
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", "shap")
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
            base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", "shap")
            tmp = np.load(f"{base_dir}/shap-lstmv4vf_pop-test_batch-{i}.npz")
            shap_test.append(tmp["shap"])
            shap_test_id1.extend(tmp["id1"])
            shap_test_id2.extend(tmp["id2"])
        self.shap_test = np.vstack(shap_test).reshape((-1, 25, 481, 3))
        self.shap_test_id1 = np.array(shap_test_id1).flatten()
        self.shap_test_id2 = np.array(shap_test_id2).flatten()

    def _get_predictions(self, step: str = "pred", type: str = "full"):
        """Get model predictions"""
        if type == "full":
            if step == "pred":
                dataset = self.full_dataset_pred.predict_dataloader()
            elif step == "train":
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
            return torch.load(predictions_file)

        # Generate new predictions
        data, predictions, labels, idx1, idx2, rates, scores = [], [], [], [], [], [], []
        for batch in dataset:
            input = batch[0]
            i = 0

            if step != "pred":
                labels.append(batch[1])
                i = 1
                rates.extend(batch[(i+3)])
            scores.extend(batch[-1])
            idx1.extend(batch[(i+1)])
            idx2.extend(batch[(i+2)])

            with torch.no_grad():
                input = input.clone().detach()
                data.append(input)
                y_hat = self.trained_model(input)
                predictions.append(y_hat)

        # Merge DATA YHAT Y ID1 ID2 RATE
        result = []
        result.append(torch.cat(data))
        result.append(torch.cat(predictions))
        if step != "pred":
            result.append(torch.cat(labels))
        result.append(idx1)
        result.append(idx2)
        if step != "pred":
            result.append(rates)
        result.append(scores)
            
        # Save predictions to file
        torch.save(result, predictions_file)
        return result

    def _get_atlas(self):
        """Get atlas data"""
        self.atlas_schaeffer = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
        atlas_img_schaeffer = image.load_img(self.atlas_schaeffer.maps)
        self.atlas_aal = get_aal()
        atlas_img_aal = image.load_img(self.atlas_aal.maps)
        
        # Get ROI names
        rois = ["rAmygdala_L", "rAmygdala_R", "rCaudate_L", "rCaudate_R", "rCerebellum_10_L", "rCerebellum_10_R", 
                "rCerebellum_3_L", "rCerebellum_3_R", "rCerebellum_4_5_L", "rCerebellum_4_5_R", "rCerebellum_6_L", 
                "rCerebellum_6_R", "rCerebellum_7b_L", "rCerebellum_7b_R", "rCerebellum_8_L", "rCerebellum_8_R", 
                "rCerebellum_9_L", "rCerebellum_9_R", "rCerebellum_Crus1_L", "rCerebellum_Crus1_R", "rCerebellum_Crus2_L", 
                "rCerebellum_Crus2_R", "rHippocampus_L", "rHippocampus_R", "rLC_L", "rLC_R", "rN_Acc_L", "rN_Acc_R", 
                "rPallidum_L", "rPallidum_R", "rParaHippocampal_L", "rParaHippocampal_R", "rPutamen_L", "rPutamen_R", 
                "rRaphe_D", "rRaphe_M", "rRed_N_L", "rRed_N_R", "rSN_pc_L", "rSN_pc_R", "rSN_pr_L", "rSN_pr_R", 
                "rThal_AV_L", "rThal_AV_R", "rThal_IL_L", "rThal_IL_R", "rThal_LGN_L", "rThal_LGN_R", "rThal_LP_L", 
                "rThal_LP_R", "rThal_MDl_L", "rThal_MDl_R", "rThal_MDm_L", "rThal_MDm_R", "rThal_MGN_L", "rThal_MGN_R", 
                "rThal_PuA_L", "rThal_PuA_R", "rThal_PuI_L", "rThal_PuI_R", "rThal_PuL_L", "rThal_PuL_R", "rThal_PuM_L", 
                "rThal_PuM_R", "rThal_Re_L", "rThal_Re_R", "rThal_VA_L", "rThal_VA_R", "rThal_VL_L", "rThal_VL_R", 
                "rThal_VPL_L", "rThal_VPL_R", "rVTA_L", "rVTA_R", "rVermis_10", "rVermis_3", "rVermis_4_5", "rVermis_6", 
                "rVermis_7", "rVermis_8", "rVermis_9"]
        
        idx = [np.where(np.array(self.atlas_aal.labels) == a.replace("r", "", 1))[0] for a in rois]
        idx = [id[0] for id in idx if id.shape[0]>0]

        schaeffer_labels = np.array(self.atlas_schaeffer.labels, dtype=str)
        aal_labels = np.array(self.atlas_aal.labels, dtype=str)[idx]
        self.roi_names = np.concatenate([schaeffer_labels, aal_labels])

        return atlas_img_schaeffer.get_fdata(), atlas_img_aal.get_fdata(), atlas_img_schaeffer.affine, idx

    def _create_nifti_map_from_shap(self, shap_values, output_file: str, zscore: bool = False, save: bool = True):
        """Create NIfTI map from SHAP values"""
        atlas_data_schaeffer, atlas_data_aal, affine, idx = self._get_atlas()
        volume = np.zeros_like(atlas_data_schaeffer)
        volume = np.repeat(volume[..., np.newaxis], (25 if shap_values.shape[0] == 25 else 1), axis=-1)
        
        # Normalize SHAP values
        norm_values = zscore(shap_values) if zscore else shap_values
        print(f"SHAP values shape: {norm_values.shape}")
        norm_values = norm_values.reshape((1, 481)) if norm_values.ndim == 1 else norm_values.reshape((25, 481))
        
        # Fill volume
        for k in range(len(norm_values)):
            for i, value in enumerate(norm_values[k, :]):
                if (i+1) > 400 and self.subcortical:
                    volume[atlas_data_aal == idx[(i - 400)], k] = value
                else:
                    volume[atlas_data_schaeffer == (i + 1), k] = value
        
        # Create and save NIfTI
        if len(norm_values) == 1:
            volume = volume[..., 0]
        nifti_img = nib.Nifti1Image(volume, affine)
        if not save:
            return nifti_img
        nib.save(nifti_img, output_file)
        return output_file
    
    # HEATMAP

    def generate_heatmap(self, pop: str = "val", contours: int = 0, smoothing: bool = True, delta_rating: bool = False, only_concerned_block: bool = False):
        """
        Generate heatmaps for each condition.
        
        Args:
            contours: Number of contour lines (0 for no contours)
            smoothing: Whether to apply Gaussian smoothing
            delta_rating: If True, plot delta of rating between previous to next iteration
        """
        self.data, self.labels, self.prediction, self.id1, self.id2, self.rate = self._get_predictions(pop, "block")
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
                    cmap = plt.cm.get_cmap('jet')
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

    def generate_corr_network_heatmap(self, pop: str = "val", label: int = 0, interpolation: str = "none", only_concerned_block: bool = False):
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "external")
        network = pd.read_csv(f"{base_dir}/schaeffer_subcortical_smith_templates.csv")
        data, prediction, labels, id1, id2, rate, scores = self._get_predictions(pop, "block")
        id1, id2, rate = np.array(id1), np.array(id2), np.array(rate)

        # Shap value
        shap_vals = self.shap_vals if pop == "val" else self.shap_test
        mask = np.argmax(softmax(labels.numpy().sum(1), axis=-1), axis=-1) == label
        shap_vals = shap_vals[mask] if only_concerned_block else shap_vals
        nblocks = (16*int(label==0)+10*int(label==1)+8*int(label==2)) if only_concerned_block else 34
        shap_vals = shap_vals.reshape(-1, nblocks, 25, 481, 3)

        # Correlation
        corr = np.zeros(481)
        for i in range(481):
            corr[i] = spearmanr(np.abs(shap_vals)[:, :, :, i, label].mean(axis=(1,2))).statistic

        # Dot product
        x_norm = corr / np.sqrt(np.sum(corr ** 2, axis=2, keepdims=True))
        weights = network.iloc[:, 1:].to_numpy()
        weights = weights / np.sqrt(np.sum(weights ** 2, axis=1, keepdims=True))
        x_network = np.dot(x_norm, weights)
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

    def generate_network_heatmap(self, pop: str = "val", label: int = 0, interpolation: str = "none", only_concerned_block: bool = False, use_network: bool = True, normalize_time: bool = False):
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
        x_network = np.matmul(x_norm.reshape(-1, 481), weights.T).reshape(-1, 25, 18, 3)
        x_network = ((x_network - x_network.min(axis=1, keepdims=True)) / (x_network.max(axis=1, keepdims=True) - x_network.min(axis=1, keepdims=True))) if normalize_time else x_network
        x_network = x_network if use_network else shap_vals
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
        plt.yticks(np.linspace(0, 17, 18), network.iloc[sort, 0].tolist()) if use_network else plt.yticks(np.linspace(0, 480, 481), self.roi_names)
        plt.xlabel('Time (TR)')
        plt.ylabel("Shap Network")
        plt.tight_layout()
        pop_name = "validation" if pop=="val" else "test"
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", pop_name, "heatmap")
        plt.savefig(f'{base_dir}/heatmap_ShapNetwork_condition-{self.conditions[label]}.png')
        plt.close()

    def generate_diffnetwork_heatmap(self, pop: str = "val", slabels: list = [0,1], interpolation: str = "none", only_concerned_block: bool = False):
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
        x_network = np.matmul(x_norm.reshape(-1, 481), weights.T).reshape(-1, 25, 18, 3)
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

    # PREDICTIONS

    def visualize_subject(self, idx: int = 0, step: str = "val"):
        """
        Visualize predictions for a specific subject.
        
        Args:
            idx: Subject index
            start_time: Starting time point
            validation: Whether to use validation or resting state data
        """
        self.step = step
        self.idx = idx

        # Get predictions and prepare data
        if step == "val" or step == "test":
            data = self._get_validation_subject_data()
        elif step == "pred":
            data = self._get_resting_state_subject_data()
        else:
            raise ValueError(f"{step} not implemented")

        # Plot if enabled
        if self.plot:
            if step == "val" or step == "test":
                self._plot_validation_subject(data)
            else:
                self._plot_resting_state_subject(data)

        return data

    def _get_validation_subject_data(self):
        """Get validation subject data with ground truth and ratings"""
        results = self._get_predictions(self.step)
        x, y, yh, id1, id2, r, m = results

        return {
            'input': x[self.idx].numpy(),
            'predictions': yh[self.idx].numpy(),
            'ground_truth': y[self.idx].numpy(),
            'ratings': r[self.idx],
            'subject_id1': id1[self.idx],
            'subject_id2': id2[self.idx],
            'time_points': y.size(1)
        }

    def _get_resting_state_subject_data(self):
        """Get resting state subject data"""
        result = self._get_predictions("pred")
        x, yh, id1, id2, m = result
        
        return {
            'input': x[self.idx].numpy(),
            'predictions': yh[self.idx].numpy(),
            'subject_id1': id1[self.idx],
            'subject_id2': id2[self.idx],
            'time_points': x.size(1)
        }

    def _plot_validation_subject(self, data):
        """Plot validation subject data with ground truth and ratings"""
        # Create subplot grid            
        fig = plt.figure(figsize=(24, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 1, 1])

        time = range(data['time_points'])

        # Predictions plot
        ax1 = fig.add_subplot(gs[0])
        self._plot_predictions(ax1, data['predictions'], time, data['time_points'])

        # Ground truth plot
        ax2 = fig.add_subplot(gs[1])
        self._plot_ground_truth(ax2, data['ground_truth'], time, data['time_points'])

        # Rate plot
        ax3 = fig.add_subplot(gs[2])
        self._plot_rate(ax3, data['ratings'], time, data['time_points'])

        # Add MAE loss title
        mae_loss = self.mae_loss(torch.tensor(data['predictions']), torch.tensor(data['ground_truth']))
        fig.suptitle(f"MAE Loss: {mae_loss.item():.4f}", fontsize=16)

        self._finalize_plot(data['time_points'], 
                          f"idx-{self.idx}_sub-{data['subject_id1']}_ses-{data['subject_id2']}", 
                          self.step)

    def _plot_resting_state_subject(self, data):
        """Plot resting state subject data"""            
        time = range(data['time_points'])
        fig, ax = plt.subplots(figsize=(15, 5))
        self._plot_predictions(ax, data['predictions'], time, data['time_points'])
        self._finalize_plot(data['time_points'], 
                          f"idx-{self.idx}_sub-{data['subject_id1']}_ses-{data['subject_id2']}", 
                          "restingstate")

    def _plot_predictions(self, ax, predictions, time, end_time):
        """Plot model predictions"""
        ax.plot(time, predictions[:end_time, 0], 'b--', label='W pred')
        ax.plot(time, predictions[:end_time, 1], 'g--', label='N pred')
        ax.plot(time, predictions[:end_time, 2], 'r--', label='R pred')
        ax.set_title('Model Predictions')
        ax.legend()

    def _plot_ground_truth(self, ax, y, time, end_time):
        """Plot ground truth values"""
        ax.plot(time, y[:end_time, 0], 'b-', label='W')
        ax.plot(time, y[:end_time, 1], 'g-', label='N')
        ax.plot(time, y[:end_time, 2], 'r-', label='R')
        ax.set_title('Ground Truth')
        ax.legend()

    def _plot_rate(self, ax, rate, time, end_time):
        """Plot rating values"""
        ax.plot(time, rate[:end_time], 'b--', label='Rate')
        ax.set_title('Rate')
        ax.legend()

    def _finalize_plot(self, end_time, idx, step):
        """Finalize and save the plot"""
        plt.xlabel('Time (s)')
        plt.tight_layout()
        if self.step == "val":
            step = "validation"
        elif self.step == "test":
            step = "test"
        elif self.step == "pred":
            step = "restingstate"
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "predictions", step)
        plt.savefig(f"{base_dir}/prediction-{step}_{idx}.png")
        plt.show()
        plt.close()

    # SCORES

    def analyze_clinical_patterns(self):
        """
        Analyze relationships between model predictions and clinical measures.
        
        Args:
            n_subjects: Number of subjects to analyze
        """
        # Get resting state predictions
        data, predictions, id1, id2 = self._get_predictions(step="pred")
        predictions = predictions.numpy()
        n_subjects, end_time, _ = data.size()

        # Create base dataframe with predictions
        df = pd.DataFrame(predictions.reshape(-1, 3), columns=["Worry", "Neutral", "Reappraisal"])
        df["timepoint"] = np.tile(np.arange(end_time), n_subjects)
        df["subject_id"] = np.repeat(np.arange(n_subjects), end_time)
        df["subject_id1"] = np.repeat(id1, end_time)
        df["subject_id2"] = np.repeat(id2, end_time)
        df["WorryOrNeutral"] = (df["Worry"] > df["Neutral"]).astype(int)

        # Calculate metrics per subject
        df_subjects = self._calculate_subject_metrics(df)
        
        # Load and merge clinical data
        df_subjects = self._merge_clinical_data(df_subjects)
        
        # Create clinical correlation plots
        self._plot_clinical_correlations(df_subjects)
        
        # Analyze state transitions
        self._analyze_state_transitions(df, df_subjects)

    def _calculate_subject_metrics(self, df):
        """Calculate various metrics for each subject"""
        return (df.groupby(['subject_id1', 'subject_id2'])
         .agg(
             # Basic probability metrics
             probability=('WorryOrNeutral', lambda x: x.sum()/x.count()),
             surprise=('WorryOrNeutral', lambda x: np.log2(1/(x.sum()/x.count()))), 
             entropy=('WorryOrNeutral', lambda x: -np.sum((x.sum()/x.count())*np.log2(x.sum()/x.count()))),
             max_duration=('WorryOrNeutral', lambda x: (np.max(np.diff(np.where(np.r_[1, x, 1] == 0))[::2]) - 1)),
             
             # Average activation values
             mean_worry=('Worry', 'mean'),
             mean_neutral=('Neutral', 'mean'),
             mean_reapp=('Reappraisal', 'mean'),
             
             # Variability metrics
             std_worry=('Worry', 'std'),
             std_neutral=('Neutral', 'std'),
             std_reapp=('Reappraisal', 'std')
         )
         .reset_index())

    def _merge_clinical_data(self, df_subjects):
        """Merge clinical data with subject metrics"""
        fina = pd.read_csv(os.path.join(self.full_dataset_trainvaltest.base_dir, "FINA", "Public", "Analysis", "misc", "FINA_2023_01_06.csv"))
        return pd.merge(fina, df_subjects, 
                       left_on=["Vault_UID", "Vault_ScanID"], 
                       right_on=['subject_id1', 'subject_id2'], 
                       how='inner')

    def _plot_clinical_correlations(self, df_subjects):
        """Create correlation plots between metrics and clinical measures"""
        clinical_measures = ["pswq_total", "rsq_total", "madrs_total", "hars_score"]
        metric_groups = {
            'probability': ["probability", "surprise", "entropy", "max_duration"],
            'activation': ["mean_worry", "mean_neutral", "mean_reapp"],
            'variability': ["std_worry", "std_neutral", "std_reapp"]
        }
        
        import seaborn as sns
        
        for measure in clinical_measures:
            for group_name, metrics in metric_groups.items():
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.flatten()
                
                for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics per plot
                    sns.regplot(data=df_subjects, x=metric, y=measure, 
                              robust=True, ax=axes[i])
                    axes[i].set_xlabel(metric.replace('_', ' ').title())
                    
                    # Calculate and display correlation
                    corr = df_subjects[metric].corr(df_subjects[measure])
                    axes[i].set_title(f'r = {corr:.3f}')
                
                plt.suptitle(f'{measure.upper()} vs {group_name.title()} Metrics')
                plt.tight_layout()
                base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "clinical", "scores")
                plt.savefig(f"{base_dir}/{measure}_{group_name}_correlations.png")
                plt.close()

    def _analyze_state_transitions(self, df, df_subjects):
        """Analyze state transitions and their clinical correlations"""
        # Calculate transition probabilities
        df_transitions = (df.groupby(['subject_id1', 'subject_id2'])
         .agg(
             pzto=('WorryOrNeutral', lambda x: len(np.where((x.to_numpy()[:-1] == 0) & (x.to_numpy()[1:] == 1))[0])/(x.count()-x.sum())),
             pztz=('WorryOrNeutral', lambda x: len(np.where((x.to_numpy()[:-1] == 0) & (x.to_numpy()[1:] == 0))[0])/(x.count()-x.sum())),
             potz=('WorryOrNeutral', lambda x: len(np.where((x.to_numpy()[:-1] == 1) & (x.to_numpy()[1:] == 0))[0])/x.sum()),
             poto=('WorryOrNeutral', lambda x: len(np.where((x.to_numpy()[:-1] == 1) & (x.to_numpy()[1:] == 1))[0])/x.sum()),
         )
         .reset_index())
        
        # Merge with clinical data
        df_transitions = self._merge_clinical_data(df_transitions)
        
        # Plot transition probabilities vs clinical measures
        clinical_measures = ["pswq_total", "rsq_total", "madrs_total", "hars_score"]
        transitions = {
            'pzto': 'Probability to get Worry',
            'pztz': 'Probability to stay Neutral',
            'potz': 'Probability to get Neutral',
            'poto': 'Probability to stay Worry'
        }

        import seaborn as sns
        
        for measure in clinical_measures:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, (trans_key, trans_label) in enumerate(transitions.items()):
                sns.regplot(data=df_transitions, x=trans_key, y=measure, 
                          robust=True, ax=axes[i])
                axes[i].set_xlabel(trans_label)
                
                # Calculate and display correlation
                corr = df_transitions[trans_key].corr(df_transitions[measure])
                axes[i].set_title(f'r = {corr:.3f}')
            
            plt.suptitle(f'{measure.upper()} vs State Transitions')
            plt.tight_layout()
            base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "clinical", "scores")
            plt.savefig(f"{base_dir}/{measure}_transitions.png")
            plt.close()
    
    # METRICS ACCURACY BLOCK        

    def loss_compute(self, population: str = "val", presentation: str = "block"):
        _, labels, prediction, _, _, _, _ = self._get_predictions(population, presentation)
        return self.mae_loss(prediction, labels)

    def _separate_populations(self, id1: np.array, cutoff: int) -> Tuple[List[int], List[int], List[int]]:
        """Separate subjects into RAW, FINA, and common populations."""
        dataset = self.block_dataset_trainvaltest.test_dataset
        raw_only_ids, fina_only_ids, common_ids = [], [], []
        
        for i, subject_id in enumerate(id1):
            raw_id = dataset.is_fina_in_raw(subject_id)
            fina_id = dataset.is_raw_in_fina(subject_id)
            if raw_id is None and fina_id is None:
                if i >= cutoff:
                    common_ids.append(i)
            else:
                if i < cutoff:
                    fina_only_ids.append(i)
                else:
                    raw_only_ids.append(i)
        
        return raw_only_ids, fina_only_ids, common_ids

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

    def _calculate_individual_metrics(self, y: np.array, y_hat: np.array, 
                                    population_ids: Dict[str, List[int]], population_subs: Dict[str, List[str]]) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate metrics for each population."""
        metric_values = [self._initialize_metric_arrays(population_subs) for i in range(len(self._comparisons))]
        
        for pop_name, pop_ids in population_ids.items():
            for j, subject_id in enumerate(list(set(population_subs[pop_name]))):
                subj_idx_sub = pop_ids[np.array(population_subs[pop_name]) == subject_id]
                metrics = self._calculate_metrics_for_block(
                    y[subj_idx_sub, :25, :].reshape(-1, 3),
                    y_hat[subj_idx_sub, :25, :].reshape(-1, 3)
                )
            
                for i, comparison in enumerate(self._comparisons):
                    for metric_name in self.metric_config.names:
                        metric_values[i][pop_name][metric_name][j] = metrics[comparison][metric_name]
        
        return metric_values

    def visualize_roc_metrics_rates(self, type: str = "block", threshold: int = 0, asc = 0, delta: bool = True) -> None:
        """Visualize ROC metrics with violin plots for different populations."""
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
        raw_in_training_ids = raw_in_training_ids.flatten() if raw_in_training_ids.size > 0 else np.array([])

        raw_in_validation_ids = np.concatenate([np.where(np.array(id1T) == dataset.is_fina_in_raw(id))[0] for id in np.unique(id1V)])
        raw_in_validation_ids = raw_in_validation_ids.flatten() if raw_in_validation_ids.size > 0 else np.array([])
        
        mask = np.ones(len(id1T), dtype=bool)
        mask[raw_in_training_ids] = False
        raw_not_in_training_ids = np.where(mask)[0]
        raw_not_in_training_ids = raw_not_in_training_ids.flatten() if raw_not_in_training_ids.size > 0 else np.array([])

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

    # BRAIN

    def analyze_brain_attributions(self, pop: str = "val", threshold: float = 0.3):
        """Analyze and visualize brain attributions for each condition and contrast"""
        shap_values = self.shap_vals if pop == "val" else self.shap_test
        from nilearn import plotting
        
        # Process each condition
        for i, condition in enumerate(self.conditions):
            print(f"Processing {condition} condition...")
            # Same seq time point SHAP values over time dimension
            mean_shap = np.abs(shap_values[..., i]).mean(axis=0).mean(axis=0)
            
            # Create and save brain map
            base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "results", self.name, "maps")
            output_file = f"{base_dir}/nifti/{condition.lower()}_map.nii.gz"
            self._create_nifti_map_from_shap(mean_shap, output_file)
            
            # Create interactive HTML visualization
            html = plotting.view_img(output_file, cmap=plt.cm.get_cmap("Reds"), vmax=mean_shap.max(), vmin=0)
            html.save_as_html(f"{base_dir}/plot/{condition.lower()}_map.html")

            mean_shap = np.abs(shap_values[:, :, :, i]).mean(axis=0)
            output_file = f"{base_dir}/nifti/{condition.lower()}_map_4d.nii.gz"
            self._create_nifti_map_from_shap(mean_shap, output_file)
            if threshold is not None:
                for t in range(25):
                    img = self._create_nifti_map_from_shap(mean_shap[t, :], output_file, save=False)
                    plotting.plot_glass_brain(img, threshold=np.quantile(mean_shap, threshold))
                    plt.title(f"{condition} - Time {t}")
                    plt.savefig(f"{base_dir}/plot/{condition.lower()}_map_time-{t}.png")
                    plt.close()

        # Process contrasts
        print("Processing contrasts...")
        for idx1, idx2, name in self.contrasts:
            contrast_vals = self._calculate_contrast(shap_values, idx1, idx2)
            max_contrast_vals = np.abs(contrast_vals).max()
            
            output_file = f"{base_dir}/nifti/{name}_map.nii.gz"
            self._create_nifti_map_from_shap(contrast_vals, output_file)
            
            html = plotting.view_img(output_file, cmap=plt.cm.get_cmap("seismic"), vmax=max_contrast_vals, vmin=-max_contrast_vals)
            html.save_as_html(f"r{base_dir}/plot/{name}_map.html")

    def _calculate_contrast(self, shap_values, idx1, idx2):
        """Calculate contrast between two conditions"""
        return (
            (np.abs(shap_values)[:, :, :, idx1]-np.abs(shap_values)[:, :, :, idx2]).mean(axis=0).mean(axis=0)
        )