# Open-X Embodiment (LeRobot Mirror)

This note captures the practical details for working with the Open-X Embodiment (OXE) dataset inside the nanoaction project. The OXE release aggregates demonstrations from 20+ research labs, spanning a broad range of robot morphologies, sensors, and task families (table-top manipulation, mobile manipulation, mobile navigation, bimanual teleoperation, language-conditioned tasks, etc.). The public release that powers RT-X and Pi0.5 covers roughly a million trajectories, several thousand distinct tasks/skills, and closes to 4 TB of compressed data.

The LeRobot community maintains a curated mirror of the aggregate data on Hugging Face. Those copies follow the **LeRobot dataset specification**, which normalises metadata, step tables, and media assets so models trained with `lerobot` or the HF `datasets` API can stream the trajectories efficiently. The same raw data is also available from Google’s original OXE bucket via the `open_x_embodiment` Python package.

> 📚 References
> • Open-X Embodiment paper: <https://arxiv.org/abs/2307.00622>
> • Google Research release notes & data portal: <https://github.com/google-research/open_x_embodiment>
> • LeRobot dataset spec & tooling: <https://huggingface.co/docs/lerobot>
> • Hugging Face dataset card for the curated mirror: <https://huggingface.co/datasets/jxu124/OpenX-Embodiment>
> • Robotics Transformer X: <https://robotics-transformer-x.github.io/>

---

## Dataset anatomy in LeRobot format

Each contributing source (e.g. `aloha`, `bridge`, `dexterity`, `rtx`) is packaged as one LeRobot dataset instance. The repositories follow a consistent layout so downstream training code can be shared:

```
openx/
  README.md                      # upstream metadata mirrors OXE docs
  dataset_infos.json             # Hugging Face dataset metadata
  aloha/                         # one source dataset (top-level folder)
    data/
      train/
        step_data.parquet        # flattened step-wise table (obs/actions/rewards/done flags)
        episode_index.parquet    # mapping episodes → slice into step_data
        media/                   # encoded RGB frames or videos referenced from step_data
      val/…                      # optional split
    metadata/
      dataset_config.json        # schema (modality names, control type, normalisation)
      statistics.json            # mins/maxes, normalisation factors, etc.
  bridge/
    …
```

Common column groups inside `step_data.parquet`:

- `observation.*`: camera images (front/left/right RGB or depth), proprioceptive signals, force/torque, language tokens.
- `action.*`: typically target joint positions, delta joint velocities, or end-effector pose deltas depending on the source.
- `episode_index`/`mask`: allow efficient slicing into episodes and batching variable-length rollouts.
- Optional extra modalities (`language_instruction`, `segmentation_masks`, `gripper_state`, …) appear when provided by the originating lab.

The LeRobot tooling can read these tables directly (`lerobot.common.dataset.replay_buffer.LeRobotDataset`) and emits PyTorch-ready batches with automatic normalisation. If you download “raw” assets from Google’s bucket you must run LeRobot’s converters yourself to obtain this schema; the Hugging Face mirror already ships the converted layout.

---

## Download options

### 1. Hugging Face + LeRobot mirror (recommended)

The easiest way to grab specific slices of the OXE aggregate is via the curated `lerobot/openx-embodiment` dataset on Hugging Face. Benefits: resumable downloads, fine-grained filtering, HF caching, and a schema that plugs straight into `lerobot`.

1. Authenticate with Hugging Face (required for large downloads):
   ```bash
   pip install --upgrade huggingface_hub
   huggingface-cli login
   ```
2. Inspect what is available:
   ```bash
   python scripts/download_openx.py --list
   ```
3. Download the subsets you need (into `data/openx` by default):
   ```bash
   # Example: pull the ALOHA and BRIDGE subsets
   python scripts/download_openx.py --datasets aloha bridge

   # Copy files instead of lightweight symlinks and place them elsewhere
   python scripts/download_openx.py --datasets aloha \
       --local-dir ~/datasets/openx \
       --copy
   ```
   The script supports additional filters (`--include-patterns`, `--ignore-patterns`), custom revisions, and re-downloading (`--force-download`). Run `python scripts/download_openx.py --help` for the full CLI surface.

LeRobot itself also ships a helper (identical end result, just bundled with the library). After installing the package, run its downloader CLI to see the available options:
```bash
pip install 'lerobot[download]'
python -m lerobot.common.download_dataset --help
```
Check the help text for the exact dataset argument names (e.g. `--dataset openx-embodiment`) and output controls. Use one approach or the other depending on your tooling preferences.

### 2. Google’s original Open-X Embodiment release

If you prefer to work directly from Google’s bucket (e.g., you plan to run custom conversions or need modalities not mirrored on HF yet):

```bash
pip install open_x_embodiment
python -m open_x_embodiment.scripts.download_dataset --help
```

That CLI mirrors the Google Cloud Storage buckets to a local folder and exposes flags such as `--dataset aloha --output_dir ~/datasets/openx_raw`. It fetches the raw NumPy / TFRecord payloads, so you must subsequently convert them into LeRobot format (`python -m lerobot.common.data_converter --config <config.yaml>`) or adapt your own preprocessing pipeline. Expect significantly larger downloads before compression.

---

## Working with the data in nanoaction

1. **Place the mirror** under `data/openx/` (default from `scripts/download_openx.py`).
2. **Point training configs** to the relevant source datasets. The LeRobot loader expects the `step_data.parquet` and `episode_index.parquet` files and will automatically load the associated media through relative paths.
3. **Normalisation / statistics** are available under each dataset’s `metadata/` folder. Use them when constructing observation/action pipelines to match the canonical LeRobot training recipes.
4. **Disk footprint**: each source dataset ranges from a few GB (e.g., language-conditioned tabletop) to hundreds of GB (multi-view teleop). Check `python scripts/download_openx.py --list` to estimate before pulling a subset.

Suggested next steps once the data is synced:

1. Use LeRobot’s inspection utilities (see `python -m lerobot --help`) to sanity-check a small subset.
2. Spin up a minimal LeRobot dataloader to inspect batch shapes and modalities.
3. Add dataset-specific preprocessing hooks (e.g., image resizing, action scaling) inside the nanoaction training loop.

The combination of this README and `scripts/download_openx.py` should be enough to bootstrap experimentation with Open-X Embodiment inside this repository without revisiting the upstream docs every time.
