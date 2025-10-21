#!/usr/bin/env python3
"""Find the sample with the highest action volatility."""

import pickle
import sys
from pathlib import Path
import numpy as np

def compute_action_volatility(pickle_path: str):
    """Compute volatility (standard deviation) of actions in a pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract actions
        actions = np.array([step['action'] for step in data['steps']])
        
        # Compute volatility as mean of standard deviations across all dimensions
        volatility = np.mean(np.std(actions, axis=0))
        
        # Also compute total variance
        total_variance = np.sum(np.var(actions, axis=0))
        
        return volatility, total_variance, len(actions), actions.shape[1]
    except Exception as e:
        print(f"Error processing {pickle_path}: {e}")
        return None, None, None, None

def find_highest_volatility_sample(data_dir: str):
    """Find the sample with highest action volatility in a directory."""
    data_path = Path(data_dir)
    pickle_files = sorted(data_path.glob("*.data.pickle"))
    
    if not pickle_files:
        print(f"No pickle files found in {data_dir}")
        return None
    
    print(f"Analyzing {len(pickle_files)} samples...\n")
    
    results = []
    for pickle_file in pickle_files:
        volatility, variance, n_steps, n_dims = compute_action_volatility(pickle_file)
        if volatility is not None:
            results.append({
                'file': pickle_file,
                'volatility': volatility,
                'variance': variance,
                'n_steps': n_steps,
                'n_dims': n_dims
            })
            print(f"{pickle_file.name:40s} | Volatility: {volatility:.6f} | Variance: {variance:.6f} | Steps: {n_steps:3d}")
    
    if not results:
        print("No valid samples found")
        return None
    
    # Sort by volatility
    results.sort(key=lambda x: x['volatility'], reverse=True)
    
    print("\n" + "="*80)
    print("HIGHEST VOLATILITY SAMPLE:")
    print("="*80)
    top = results[0]
    print(f"File: {top['file']}")
    print(f"Volatility: {top['volatility']:.6f}")
    print(f"Total Variance: {top['variance']:.6f}")
    print(f"Steps: {top['n_steps']}")
    print(f"Action Dimensions: {top['n_dims']}")
    
    return top['file']

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_highest_volatility.py <data_directory>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    highest_volatility_file = find_highest_volatility_sample(data_dir)
    
    if highest_volatility_file:
        print(f"\n{highest_volatility_file}")
