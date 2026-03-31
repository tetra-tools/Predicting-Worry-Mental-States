import os
import torch
import shap
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

def main(args):
    # Set number of CPU cores
    torch.set_num_threads(20)
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)

    # Initialize data module with fixed settings
    datamodule = TimeSeriesDataModule(
        base_dir="/mnt/argo/Studies",
        window_timeserie=20,  # Fixed window size
        samples_per_file=100,  # Fixed samples per subject
        subcortical=True,  # Always use subcortical
        num_workers=20,  # Use 20 cores
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
    
    # Initialize data module with fixed settings
    datamodule = TimeSeriesDataModule(
        base_dir="/mnt/argo/Studies",
        window_timeserie=25,  # Fixed window size
        samples_per_file=34,  # Fixed samples per subject
        subcortical=True,  # Always use subcortical
        num_workers=40,  # Use 20 cores
        batch_size = 34*3,
        identifiers_output=True,
        seed=args.seed if args.seed is not None else 123,  # Use provided seed or default
        notrandom=[idx1, idx2]
    )
    datamodule.setup("fit-test")
    print(len(datamodule.train_dataset))
    print(len(datamodule.val_dataset))
    print(len(datamodule.test_dataset))

    all_batches = []
    for batch in datamodule.train_dataloader():
        x, _, _, _ = batch
        all_batches.append(x)
    background_data = torch.cat(all_batches)
    
    # Initialize model
    model = LSTMClassifierv4.load_from_checkpoint(args.checkpoint)
    model.eval()

    class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.model.eval()
            
            def forward(self, x):
                result = self.model(x)
                return result[:, -1, :].squeeze()

    wmodel = ModelWrapper(model)

    # SHAP
    from tqdm import tqdm
    explainer = shap.DeepExplainer(wmodel, background_data)
    for batch_idx, batch in enumerate(tqdm(datamodule.test_dataloader(), desc="Processing SHAP")):
        if batch_idx == args.batch_idx:
            tqdm.write("Step 1: Get batch...")
            x, y, id1, id2, rate = batch
            x.requires_grad = True
            tqdm.write("Step 1: Preprocessing SHAP...")
            shap_vals = explainer.shap_values(x, ranked_outputs=None, check_additivity=False)
            tqdm.write("Step 3: Compressing results...")
            np.savez_compressed(f"shap-lstmv4vf_pop-test_batch-{batch_idx}.npz", shap=shap_vals, id1=id1, id2=id2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', default="./checkpoints/lstm_v4_host-smp-n217.crc.pitt.edu_version3/model_best_score=val_loss=0.1113.ckpt", type=str)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_idx', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--gradient_clip_val', type=float, default=0.8)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123, help='Random seed for reproducibility')
    parser.add_argument('--noise_min', type=float, default=0, help='Noise min')
    parser.add_argument('--noise_max', type=float, default=.5, help='Noise max')
    parser.add_argument('--noise_duration', type=int, default=200, help='Noise duration')
    parser.add_argument('--cosine_annealing_T0', type=int, default=5, help='T0')
    parser.add_argument('--cosine_annealing_T_mult', type=int, default=1, help='Multiplier')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['plateau', 'cosine'])
    parser.add_argument('--noise_scheduler', type=str, default='linear', choices=['linear', 'quadratic'])
    
    args = parser.parse_args()
    main(args)
