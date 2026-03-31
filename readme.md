# Predicting Worry-Related Mental States using Regional Brain Activity with LSTM Neural Networks

**An advanced research project using deep learning to predict worry-related mental states from fMRI data and LSTM neural networks.**

> A deep learning approach using LSTM neural networks to predict worry-related mental states from regional brain activity measured by fMRI.

## Table of Contents

- [Abstract](#abstract)
- [Resources](#resources)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Visualization and Analysis Tools](#visualization-and-analysis-tools)
- [Training Configuration](#training-configuration)
- [Performance Metrics](#performance-metrics)
- [Results and Findings](#results-and-findings)
- [Authors and Affiliations](#authors-and-affiliations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Abstract

**Background:** Mild worry is a relatively common, spontaneous and evolutionary adaptive phenomenon. By contrast, severe, pathologic worry is defined as a complex cognitive and cognitive process, negative-affect laden and relatively uncontrollable. Severe worry is frequently encountered transdiagnostic symptom in late-life mood and anxiety disorders. We have previously mapped neural correlates of severe, pathological worry by using functional magnetic resonance imaging (fMRI). However, much of this work has focused on functional statistical approaches that can miss the inherent fluctuations of worry-related mental states. In this analysis, we used a long short-term memory (LSTM) approach, a type of deep recurrent neural network, to predict worry-related mental states using regional timeseries brain activity.

## Resources

- [Article](https://1drv.ms/w/c/5caa542f08cff116/EXqrijPtGqFAj5z8SjT4S_EB6a3MFbiajNqjwEhOjIjvLg)
- [Poster](https://1drv.ms/p/c/5caa542f08cff116/ERbxzwgvVKoggFxccRQAAAABcQCJQ3pDkrJ9goaKiydB9w) 

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training with default parameters:**
   ```bash
   python main.py
   ```

3. **Run training with custom parameters:**
   ```bash
   python main.py --model lstm_v4 --learning_rate 0.001 --batch_size 64
   ```

4. **Run SHAP analysis:**
   ```bash
   python main_shap.py
   ```

```bash
# Core dependencies
torch>=1.9.0
pytorch-lightning>=1.5.0
wandb>=0.12.0
numpy>=1.19.2
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.24.1

# Data processing
nibabel>=3.2.0  # For brain imaging data
nilearn>=0.8.0  # For brain visualization
scikit-learn>=0.24.0

# Utilities
tqdm>=4.62.0
pyyaml>=5.4.0

# Optional but recommended
tensorboard>=2.6.0  # For tensorboard logging
scipy>=1.7.0  # For statistical computations
shap>=0.46.0  # For attribution analysis
```

To install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── datasets/           # Data loading and processing modules
│   ├── TimeSeriesDataModule.py
│   └── TimeSeriesDataset.py
├── external/           # External resources (AAL atlas, templates)
│   ├── aal_labels.tsv
│   ├── aal_metadata.json
│   ├── aal.nii.gz
│   └── images/         # Brain network visualization images
├── models/            # Neural network architectures
│   └── lstm_v4.py     # Main LSTM model implementation
├── sbatch/            # SLURM job submission scripts
├── utils/             # Utility functions
├── visualization/     # Visualization tools and analysis results
│   ├── basic/         # Basic visualization notebooks
│   └── networks/      # Network-based analysis tools
├── main.py           # Main training script
├── main_shap.py      # SHAP analysis script
├── main_perforated.py # Perforated analysis script
└── requirements.txt  # Project dependencies
```

## Usage

The main training script (`main.py`) supports various configurations through command-line arguments:

```bash
python main.py [arguments]
```

### Key Arguments

- `--logger`: Logger type ('tensorboard' or 'wandb', default: 'wandb')
- `--model`: Model version ('lstm_v3' or 'lstm_v4', default: 'lstm_v4')
- `--learning_rate`: Learning rate (default: 0.0005)
- `--batch_size`: Batch size (default: 32)
- `--window_size`: Time window size (default: 20)
- `--samples_per_subject`: Samples per subject (default: 100)
- `--use_gpu`: Enable GPU training
- `--num_gpus`: Number of GPUs to use
- `--precision`: Training precision ('32', '16', or 'bf16')

### Example Command

```bash
python main.py --model lstm_v4 --learning_rate 0.0005 --batch_size 32 --window_size 20 --use_gpu
```

## Model Architecture

The project implements several versions of LSTM-based models for time series classification, with LSTM v4 being the primary model used for the results. The LSTM v4 architecture includes:

### Network Structure
- **Input Processing**:
  - Initial projection from 481 channels to 512 (hidden_size*2)
  - Dimensionality reduction to 256 features with ReLU activation
  - Dropout and Layer Normalization for regularization

- **LSTM Blocks**:
  - Two stacked bidirectional LSTM layers with residual connections
  - First block: 256 → 256 features
  - Second block: 256 → 128 features
  - Each block includes:
    - Bidirectional LSTM
    - Post-processing with Linear, ReLU, Dropout, and LayerNorm
    - Residual connection

- **Output Layer**:
  - Progressive dimensionality reduction: 128 → 64 → 3
  - Final activation: Tanh + normalization to [0,1] range

### Training Features
- **Noise Injection**:
  - Adaptive Gaussian noise during training
  - Noise level decreases over time (linear or quadratic scheduling)
  - Helps prevent overfitting and improves generalization

- **Optimization**:
  - AdamW optimizer with weight decay
  - Learning rate scheduling options:
    - Cosine annealing with warm restarts
    - Reduce on plateau
  - Gradient clipping for stability
  - Xavier and Orthogonal weight initialization

- **Loss Function**:
  - L1 loss for regression
  - Class-weighted accuracy metric
  - Support for multi-class prediction (3 classes)

### Graph Neural Network Extension (LSTM-GNN)
The project also implements a sophisticated Graph Neural Network architecture that combines LSTM with graph-based learning. This model uses TransformerConv layers to capture spatial relationships between brain regions, with a hierarchical structure of graph blocks (481→128→32→16 regions) interleaved with bidirectional LSTM layers. The network incorporates edge attributes based on brain network templates, TopK pooling for region selection, and maintains both temporal and spatial information through the entire processing pipeline. This architecture enables the model to learn both temporal dynamics and spatial relationships between brain regions simultaneously.

## Dataset

The dataset consists of fMRI time series data from older adults across two studies:

### Data Structure
- **Subjects**: 116 participants total (FINA) and 88 subjects (RAW)
- **Data Split**:
  - Training: 80% of subjects (93 subjects)
  - Validation: 20% of subjects (23 subjects)
  - Test: Separate holdout set (RAW)

### Processing
- **Atlas**: AAL3 parcellation for region definition
  - 481 channels including subcortical regions
  - Comprehensive brain coverage

### Time Series Processing
- **Windowing**:
  - Configurable window size (default: 20 timepoints)
  - Sliding window approach for temporal analysis
  - 100 samples per subject for balanced representation

### Features
- **Data Types**:
  - TASK_FINA: Task-based fMRI data for training/validation
  - RS_FINA: Resting-state data for predictions
  - TASK_RAW: Raw task data for testing
- **Cross-validation**:
  - Support for k-fold cross-validation
  - Subject-wise splitting to prevent data leakage
- **Augmentation**:
  - Random sampling of time windows
  - Noise injection during training

## Visualization and Analysis Tools

### Brain Network Visualization
- **Network Analysis**:
  - Visualization of brain networks using nilearn
  - Support for multiple atlases (AAL3, Schaefer)
  - Network-based analysis of brain regions
  - Interactive HTML visualizations of brain maps

### SHAP Value Analysis
- **Attribution Visualization**:
  - Generation of brain attribution maps
  - Condition-specific SHAP value analysis
  - Contrast analysis between conditions
  - Network-level attribution analysis

### Statistical Visualization
- **Tools and Libraries**:
  - matplotlib and seaborn for basic plotting
  - nilearn for brain visualization
  - R-based ridge plots using ggridges
  - Interactive plots using plotly
  - Science-style plots using scienceplots

### Analysis Features
- **Heatmap Generation**:
  - Time-series heatmaps of network activity
  - Rating-based analysis
  - Network-specific temporal patterns
  - Contrast visualization between conditions

- **Clinical Pattern Analysis**:
  - Correlation analysis with clinical measures
  - State transition analysis
  - ROC metrics visualization
  - Population-specific analyses

## Training Configuration

- **Maximum epochs**: 250
- **Early stopping**: Patience of 50 epochs
- **Model checkpointing**: Based on validation loss
- **Logging**: Wandb and TensorBoard integration
- **Reproducibility**: Random seed management for consistent results

## Performance Metrics

The model is evaluated using multiple metrics:
- **Accuracy**: Class-weighted accuracy for balanced evaluation
- **Loss**: L1 loss for regression tasks
- **Validation**: Subject-wise cross-validation to prevent overfitting
- **Generalization**: Performance on held-out test set (RAW dataset)

## Results and Findings

The LSTM-based approach demonstrates superior performance compared to traditional statistical methods in capturing temporal dynamics of worry-related brain states. Key findings include:

- **Temporal Sensitivity**: The model successfully captures fluctuations in worry states over time
- **Network Specificity**: Different brain networks show varying importance for worry prediction
- **Generalization**: Good performance on independent test datasets
- **Clinical Relevance**: Predictions correlate with clinical measures of worry and anxiety

## Authors and Affiliations

**Campion J.Y.¹·²**, Butters M.A.¹, Tudorascu D.L.⁴·⁵, Karim H.T.¹·⁵, Andreescu C.¹

**Affiliations:**
1. Department of Psychiatry, University of Pittsburgh, Pittsburgh, PA, USA
2. UMR 1253, iBrain, Université de Tours, Inserm, Tours, France
3. Centre Hospitalier Régional Universitaire (CHRU) de Tours, Tours, France
4. Department of Biostatistics, University of Pittsburgh, Pittsburgh, PA, USA
5. Department of Bioengineering, University of Pittsburgh, Pittsburgh, PA, USA

## Citation

If you use this code in your research, please cite:

```bibtex

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

