#!/usr/bin/env python3
"""Plot time series of actions and observation states from a pickle file."""

import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_timeseries_from_pickle(pickle_path: str, output_dir: str = None):
    """Extract and plot time series data from pickle file."""
    
    # Load pickle file
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract actions and states
    actions = []
    states = []
    for step in data['steps']:
        actions.append(step['action'])
        states.append(step['observation']['state'])
    
    actions = np.array(actions)
    states = np.array(states)
    
    print(f"Loaded {len(actions)} timesteps from {pickle_path}")
    print(f"  - Action shape: {actions.shape}")
    print(f"  - State shape: {states.shape}")
    
    # Set output directory
    if output_dir is None:
        pickle_file = Path(pickle_path)
        output_dir = pickle_file.parent
    else:
        output_dir = Path(output_dir)
    
    pickle_stem = Path(pickle_path).stem
    
    # Plot actions
    fig, axes = plt.subplots(actions.shape[1], 1, figsize=(12, 2 * actions.shape[1]), sharex=True)
    if actions.shape[1] == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(actions[:, i], linewidth=1.5)
        ax.set_ylabel(f'Action {i}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep', fontsize=10)
    fig.suptitle(f'Actions Time Series - {pickle_stem}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    actions_output = output_dir / f"{pickle_stem}_actions.png"
    plt.savefig(actions_output, dpi=150, bbox_inches='tight')
    print(f"Saved actions plot: {actions_output}")
    plt.close()
    
    # Plot states
    fig, axes = plt.subplots(states.shape[1], 1, figsize=(12, 2 * states.shape[1]), sharex=True)
    if states.shape[1] == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(states[:, i], linewidth=1.5)
        ax.set_ylabel(f'State {i}', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Timestep', fontsize=10)
    fig.suptitle(f'Observation States Time Series - {pickle_stem}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    states_output = output_dir / f"{pickle_stem}_states.png"
    plt.savefig(states_output, dpi=150, bbox_inches='tight')
    print(f"Saved states plot: {states_output}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_episode_timeseries.py <pickle_file> [output_dir]")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_timeseries_from_pickle(pickle_path, output_dir)
