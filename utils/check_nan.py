import pandas as pd
import os
import glob
from typing import List, Dict
import numpy as np

def check_nan_in_files(base_dir: str = "/mnt/argo/Studies/FINA", subcortical: bool = False) -> Dict[str, Dict]:
    """
    Check for NaN values in all files in the dataset.
    
    Args:
        base_dir: Base directory containing the data structure
        subcortical: Whether to check subcortical files
        
    Returns:
        Dictionary with filenames as keys and NaN information as values
    """
    nan_report = {}
    
    # Find all files
    subject_dirs = glob.glob(os.path.join(base_dir, "9*"))
    
    for subject_dir in subject_dirs:
        session_dirs = sorted(glob.glob(os.path.join(subject_dir, "8*")))
        if session_dirs:
            # Check both task types
            for task in ["step02_WorryInduction", "step03_Rest"]:
                target_dir = os.path.join(session_dirs[0], task)
                if os.path.exists(target_dir):
                    filename = "schaeffer_subcortical_timeseries.csv" if subcortical else "schaeffer_timeseries.csv"
                    file_path = os.path.join(target_dir, filename)
                    
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            
                            # Check for NaN values
                            nan_count = df.isnull().sum().sum()
                            if nan_count > 0:
                                # Get columns with NaN
                                nan_cols = df.columns[df.isnull().any()].tolist()
                                # Get rows with NaN
                                nan_rows = df.index[df.isnull().any(axis=1)].tolist()
                                
                                nan_report[file_path] = {
                                    'total_nan': nan_count,
                                    'nan_columns': nan_cols,
                                    'nan_rows': nan_rows,
                                    'total_rows': len(df),
                                    'total_cols': len(df.columns)
                                }
                        except Exception as e:
                            nan_report[file_path] = {
                                'error': str(e)
                            }
    
    return nan_report

def print_nan_report(report: Dict):
    """Print a formatted report of NaN findings"""
    if not report:
        print("No files with NaN values found!")
        return
        
    print("\n=== NaN Values Report ===\n")
    
    for file_path, info in report.items():
        print(f"\nFile: {os.path.basename(file_path)}")
        print(f"Directory: {os.path.dirname(file_path)}")
        
        if 'error' in info:
            print(f"Error reading file: {info['error']}")
            continue
            
        print(f"Total NaN values: {info['total_nan']}")
        print(f"Dataset shape: {info['total_rows']} rows × {info['total_cols']} columns")
        
        if info['nan_columns']:
            print("\nColumns with NaN values:")
            for col in info['nan_columns']:
                print(f"  - {col}")
                
        if info['nan_rows']:
            print(f"\nRows containing NaN values: {len(info['nan_rows'])} rows")
            if len(info['nan_rows']) > 5:
                print(f"First 5 row indices: {info['nan_rows'][:5]}")
            else:
                print(f"Row indices: {info['nan_rows']}")
        
        print("-" * 50)

if __name__ == "__main__":
    # Check regular files
    print("\nChecking regular timeseries files...")
    regular_report = check_nan_in_files(subcortical=False)
    print_nan_report(regular_report)
    
    # Check subcortical files
    print("\nChecking subcortical timeseries files...")
    subcortical_report = check_nan_in_files(subcortical=True)
    print_nan_report(subcortical_report)