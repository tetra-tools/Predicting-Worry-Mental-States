import re
import scipy
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Tuple, Dict
import numpy as np
import glob
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 base_dir: str = "/mnt/argo/Studies",
                 window_timeserie: int = 100,
                 samples_per_file: int = 20,
                 subcortical: bool = False,
                 set: str = "TASK_FINA",
                 seed: int = 123,
                 rate_output: bool = False,
                 identifiers_output: bool = False,
                 scores_output: bool = False,
                 cache_dir: str = "cached_data"):
        """
        Args:
            base_dir: Base directory containing the data structure
            window_timeserie: Size of the sliding window
            samples_per_file: Number of random segments to generate per file
            subcortical: Whether to use subcortical data
            set: Dataset type (TASK_FINA, RS_FINA, etc.)
            cache_dir: Directory to save preprocessed data
        """
        cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", cache_dir)
        
        self.window_timeserie = window_timeserie
        self.samples_per_file = samples_per_file
        self.set = set
        self.seed = seed
        self.subcortical = subcortical
        self.base_dir = base_dir
        self.cache_dir = cache_dir
        self.rate_output = rate_output
        self.identifiers_output = identifiers_output
        self.scores_output = scores_output
        self.fina = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get data paths based on set type
        database = ["FINA", "9*", "8*"] if "FINA" in set else ["RAW", "1*", "1*"]
        task = ("step02_WorryInduction" if "FINA" in set else "step05_worry") if "TASK" in set else "step03_Rest"
        subcortical_file = "schaeffer_timeseries.csv" if not self.subcortical else "schaeffer_subcortical_timeseries.csv"

        # Remove duplicates if needed
        if "RAW" in set and "NEW" in set:
            self.filter_raw_duplicate()

        # Load cached data or create new samples
        cache_file = self._get_cache_filename()
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}")
            cached_data = torch.load(cache_file, weights_only=False, map_location='cpu')  # Always load to CPU first
            self.samples = cached_data['samples']
            self.nan_report = cached_data.get('nan_report', {})
        else:
            # Get all file paths
            self.get_data(database, task, subcortical_file)
            print("Processing files and creating cache...")
            self.samples = []
            self.nan_report = {}
            self._preprocess_and_cache()

        # Get common file
        common_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "common_file.csv")
        self.common_file = pd.read_csv(common_file)
        self.common_file = self.common_file[self.common_file["raw_id"].notna() & self.common_file["mr_studyid"].notna()]
        self.common_file["raw_id"] = self.common_file["raw_id"].apply(lambda x: str(x).replace("RAW_", ""))
        self.common_file = self.common_file[["raw_id", "mr_studyid"]]

    def _get_cache_filename(self) -> str:
        """Generate unique cache filename based on parameters"""
        params = f"data-{self.set}_size-{self.window_timeserie}_sample-{self.samples_per_file}"
        params += f"_roi-{'wSubcortical' if self.subcortical else 'woSubcortical'}_seed-{self.seed}"
        return os.path.join(self.cache_dir, f"cache_{params}.pt")

    def get_data(self, database, task, subcortical):
        """Get task data file paths"""
        subject_dirs = glob.glob(os.path.join(self.base_dir, f"{database[0]}/Public/Analysis/data", database[1]))
        self.file_paths = []
        for subject_dir in subject_dirs:
            session_dirs = sorted(glob.glob(os.path.join(subject_dir, database[2])))
            if session_dirs:
                target_dir = os.path.join(session_dirs[0], task)
                if os.path.exists(target_dir):
                    file_path = os.path.join(target_dir, subcortical)
                    if os.path.exists(file_path):
                        self.file_paths.append(file_path)

    def _check_nan(self, df: pd.DataFrame, file_path: str) -> Dict:
        """Check for NaN values in dataframe"""
        nan_sum = df.isnull().sum().sum()
        if nan_sum > 0:
            nan_cols = df.columns[df.isnull().any()].tolist()
            return {
                'file': file_path,
                'total_nan': nan_sum,
                'nan_columns': nan_cols,
                'total_rows': len(df)
            }
        return None
    
    def _preprocess_matlab_data(self, dir, n_timepoints):
        # Load and process mat file
        file = glob.glob(dir)
        if file:
            df = scipy.io.loadmat(file[0])
            df = pd.DataFrame(df['responses'])
            df[0] = df[0].apply(lambda s: int(s[0]) if s[0] else 0)
            df[1] = df[1].apply(lambda s: float(s[0][0]) if s[0] else 0)
            df[2] = df[2].apply(lambda s: str(s[0]) if s[0] else "BL")
        else:
            return None, None
        
        # Create timeseries        
        rates = np.zeros(n_timepoints)
        base = np.arange(34)
        start_idx = base*35+10

        # Iterate for each block
        for idx in start_idx:
            # Relative time = 0-9 RT / 10-24 RP / 24-34 FC
            values = df[(df[1] < idx+25) & (df[1] > idx+9)][0]
            rates[idx:idx+25] = np.median(np.array(values))
        
        return rates

    def _create_tensor(self, data, dtype=torch.float32):
        """Create a contiguous tensor with proper memory layout"""
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype).contiguous()
        return torch.tensor(data, dtype=dtype).contiguous()

    def _preprocess_and_cache(self):
        """Preprocess files and cache the results"""
        for file_path in tqdm(self.file_paths, desc="Processing files"):
            # Load and check CSV
            df = pd.read_csv(file_path)
            nan_info = self._check_nan(df, file_path)

            if "RAW" in self.set:
                time_index = pd.date_range(start="2024-01-01", periods=len(df), freq=f"700ms")
                df.index = time_index
                df = df.resample("1s").mean().interpolate(method="cubicspline")
            
            if nan_info:
                self.nan_report[file_path] = nan_info
                print(f"Warning: NaN found in {file_path}")
                continue

            task = "step02_WorryInduction" if "TASK" in self.set else "step03_Rest"
            task = task if "FINA" in self.set else "step05_worry"
            subcortical_file = "schaeffer_timeseries.csv" if not self.subcortical else "schaeffer_subcortical_timeseries.csv"
    
            # Process valid file
            n_timepoints = len(df)
            max_start_idx = n_timepoints - self.window_timeserie
            if self.rate_output:
                folder = "WorryInduction/behavioral_data" if "FINA" in self.set else "WorryTask"
                rates = self._preprocess_matlab_data(str(file_path).replace(f"{task}/{subcortical_file}", f"converted/{folder}/responses*.mat"), n_timepoints)
            else:
                rates = np.zeros(n_timepoints)

            blocks_start = np.arange(34)
            blocks_start = blocks_start*35+10

            seg_start = np.arange(70)
            seg_start = seg_start*17+10

            id1, id2 = self._extract_identifiers_pre(file_path)
            score = self._get_scores_clinic_fina(id2) if "FINA" in self.set else self._get_scores_clinic_raw(id2)

            if max_start_idx >= 0:
                for i in range(self.samples_per_file):
                    rate = None

                    if self.samples_per_file == 1:
                        start_idx = 10 if "TASK" in self.set else 0
                        self.window_timeserie = n_timepoints-start_idx
                    elif self.samples_per_file == 34:
                        start_idx = blocks_start[i]
                        rate = rates[start_idx:start_idx + self.window_timeserie]
                    elif self.samples_per_file == 70:
                        start_idx = seg_start[i]
                    else:
                        start_idx = np.random.randint(10, max_start_idx)
                    
                    if rate is None:
                        rate = rates[start_idx:start_idx + self.window_timeserie]
                    # Extract and preprocess window
                    df_window = df.iloc[start_idx:start_idx + self.window_timeserie]
                    if "RS" in self.set:
                        features = self._create_tensor(df_window.values)
                        labels = self._create_tensor(np.zeros((self.window_timeserie, 3)))
                    else:
                        features = self._create_tensor(df_window.iloc[:, :-3].values)
                        labels = self._create_tensor(df_window.iloc[:, -3:].values)
                        labels = (labels - (-0.2)) / (1.4)  # Normalize
                    
                    self.samples.append({
                        'features': features,
                        'labels': labels,
                        'file_path': file_path,
                        'start_idx': start_idx,
                        'rate': torch.FloatTensor(rate),
                        'id1': id1,
                        'id2': id2,
                        'scores': torch.FloatTensor(score),
                    })
        
        # Save processed data and NaN report
        cache_data = {
            'samples': self.samples,
            'nan_report': self.nan_report
        }
        print(f"Saving cache to {self._get_cache_filename()}")
        torch.save(cache_data, self._get_cache_filename())
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total files: {len(self.file_paths)}")
        print(f"Files with NaN: {len(self.nan_report)}")
        print(f"Valid samples created: {len(self.samples)}")
        if self.nan_report:
            print("\nFiles with NaN values:")
            for file_path, info in self.nan_report.items():
                print(f"\nFile: {os.path.basename(file_path)}")
                print(f"Total NaN: {info['total_nan']}")
                print(f"Affected columns: {info['nan_columns']}")

    def filter_raw_duplicate(self):
        """Filter out duplicates for RAW data"""
        pass

    def _extract_identifiers(self, idx: int):
        sample = self.samples[idx]
        match = re.search(r"^.+(9[0-9]{5}).+(8[0-9]{5}).+$", sample['file_path']) if "FINA" in self.set else re.search(r"^.+(1[0-9]{3}).+(1[0-9]{3}).+$", sample['file_path'])
        return match.groups()

    def _extract_identifiers_pre(self, file_path):
        match = re.search(r"^.+(9[0-9]{5}).+(8[0-9]{5}).+$", file_path) if "FINA" in self.set else re.search(r"^.+(1[0-9]{3}).+(1[0-9]{3}).+$", file_path)
        return match.groups()
    
    def _get_scores_clinic_fina(self, id):
        fina = pd.read_csv(os.path.join(self.base_dir, "FINA", "Public", "Analysis", "misc", "FINA_2023_01_06.csv"))
        cols = ["pswq_total", "rsq_total", "hars_score"]
        fina = fina.loc[fina["Vault_ScanID"] == str(id), cols].to_numpy().flatten()
        if fina.shape[0] == 0:
            return ValueError(f"Subject {id} has no data")
        fina = (fina - np.array([0, 22, 0])) / np.array([80, 88, 56])
        return fina
    
    def _get_scores_clinic_raw(self, id):
        raw = pd.read_csv(os.path.join("/mnt/argo/Workspaces", "Staff", "Jacques-Yves_Campion", "Public", "WorryInduction", "RAW_JY_Data_Request_2.10.25.csv"))
        cols = ["pswq_total", "rsq_total", "hars_score"]
        raw["Vault_UID"] = raw.raw_id.str.replace("RAW_", "").astype(np.int64)
        raw = raw.loc[raw["Vault_UID"] == int(id), cols].to_numpy().flatten()
        if raw.shape[0] == 0:
            return ValueError(f"Subject {id} has no data")
        raw = (raw - np.array([0, 22, 0])) / np.array([80, 88, 56])
        return raw
    
    def _keep_in_list(self, id1, id2):
        keep = []
        for id in range(len(self)):
            val1, val2 = self._extract_identifiers(id)
            if val1 in id1 and val2 in id2:
                keep.append(id)
        return keep

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.samples[idx]
        returned = [sample['features']]
        if "TASK" in self.set:
            returned += [sample['labels']]
        if self.identifiers_output:
            id1, id2 = self._extract_identifiers(idx)
            returned += [id1]
            returned += [id2]
        if self.rate_output:
            returned += [sample['rate']]
        if self.scores_output:
            returned += [sample['scores']]
        
        return returned

    def get_feature_dim(self) -> int:
        if self.samples:
            return self.samples[0]['features'].shape[1]
        return None
    
    def is_raw_in_fina(self, id1: int) -> bool:
        if self.common_file[self.common_file["raw_id"] == id1].shape[0] > 0:
            return self.common_file[self.common_file["raw_id"] == id1]["mr_studyid"].values[0]
        return None
    
    def is_fina_in_raw(self, id1: int) -> bool:
        if self.common_file[self.common_file["mr_studyid"] == id1].shape[0] > 0:
            return self.common_file[self.common_file["mr_studyid"] == id1]["raw_id"].values[0]
        return None
