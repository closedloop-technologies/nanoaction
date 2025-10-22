#!/usr/bin/env python3
"""
Profile a subset of OpenX-Embodiment datasets hosted on Hugging Face by
downloading a handful of archives per dataset, unpacking representative samples,
and generating structured summaries (including Pydantic schemas) for quick inspection.

The profiler intentionally fetches only a couple of files per dataset to keep
runtime and storage requirements modest while still surfacing the core schema,
modalities, and episode statistics.

Example usage:

    uv run python scripts/profile_openx.py --datasets cmu_franka_exploration_dataset_converted_externally_to_rlds cmu_stretch imperialcollege_sawyer_wrist_cam

    uv run python scripts/profile_openx.py --limit 3 --archives-per-dataset 2

By default the script deletes each downloaded archive right after profiling so
disk consumption stays minimal. Pass --keep-archives to retain the files.
"""

from __future__ import annotations

import argparse
import io
import pickle
import statistics
import sys
import tarfile
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from huggingface_hub import HfApi, snapshot_download
from PIL import Image


DEFAULT_REPO_ID = "jxu124/OpenX-Embodiment"


def humanize_bytes(num_bytes: int) -> str:
    """Return a human readable byte count."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} {units[-1]}"


def is_printable_text(blob: bytes, *, min_ratio: float = 0.85) -> bool:
    """Heuristic to decide whether bytes likely represent UTF-8 text."""
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(ch.isprintable() or ch.isspace() for ch in text)
    return printable / max(len(text), 1) >= min_ratio


def iter_dataset_files(
    repo_files: Sequence[str],
) -> Mapping[str, list[str]]:
    """Group repo files by top-level dataset folder."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in repo_files:
        if "/" not in path:
            continue
        dataset, remainder = path.split("/", 1)
        if not remainder:
            continue
        grouped[dataset].append(path)
    return grouped


def select_datasets(
    *,
    available: Sequence[str],
    requested: Sequence[str] | None,
    limit: int | None,
) -> list[str]:
    """Return the list of dataset names to process."""
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SystemExit(
                f"Requested datasets not found in {DEFAULT_REPO_ID}: {', '.join(missing)}"
            )
        return list(requested)
    if limit is None:
        limit = 3
    return list(sorted(available))[:limit]


@dataclass
class FieldSummary:
    name: str
    type_hint: str
    description: str | None = None


@dataclass
class ObservationSummary(FieldSummary):
    width: int | None = None
    height: int | None = None
    channels: int | None = None
    image_format: str | None = None


@dataclass
class StepSummary:
    fields: list[FieldSummary] = field(default_factory=list)
    observation_fields: list[FieldSummary] = field(default_factory=list)


@dataclass
class EpisodeSample:
    name: str
    num_steps: int
    duration_seconds: float | None
    instructions: list[str]
    step_summary: StepSummary


@dataclass
class ArchiveProfile:
    path: Path
    size_bytes: int
    entries: int
    sample_names: list[str]
    episode_samples: list[EpisodeSample]
    tree_preview: str


@dataclass
class DatasetProfile:
    name: str
    archives: list[ArchiveProfile] = field(default_factory=list)
    total_bytes: int = 0
    total_downloaded_files: int = 0
    total_episodes: int = 0
    sampled_episode_count: int = 0
    step_counts: list[int] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    instruction_counter: Counter[str] = field(default_factory=Counter)
    camera_summaries: dict[str, ObservationSummary] = field(default_factory=dict)
    timeseries_summaries: dict[str, FieldSummary] = field(default_factory=dict)
    schema_sample: EpisodeSample | None = None


def describe_numpy_array(array: np.ndarray) -> tuple[str, str]:
    """Return (type_hint, description) for a numpy array."""
    base_type = "float"
    if np.issubdtype(array.dtype, np.integer):
        base_type = "int"
    elif np.issubdtype(array.dtype, np.bool_):
        base_type = "bool"
    type_hint = f"list[{base_type}]"
    description = f"numpy.ndarray shape={array.shape}, dtype={array.dtype}"
    return type_hint, description


def describe_image(blob: bytes) -> ObservationSummary | None:
    """Inspect bytes to see if they represent an image."""
    try:
        with Image.open(io.BytesIO(blob)) as img:
            width, height = img.size
            channels = len(img.getbands())
            summary = ObservationSummary(
                name="",
                type_hint="bytes",
                description=f"{img.format} image, mode={img.mode}",
                width=width,
                height=height,
                channels=channels,
                image_format=img.format,
            )
            return summary
    except Exception:  # noqa: BLE001 - best-effort inspection
        return None
    return None


def build_tree_preview(file_names: Sequence[str], max_lines: int = 12) -> str:
    """Produce a minimal ascii tree listing from a list of file names."""
    if not file_names:
        return ""
    display = list(file_names)
    truncated = False
    if len(display) > max_lines:
        display = display[: max_lines - 1]
        truncated = True
    lines = []
    for idx, name in enumerate(display):
        prefix = "└── " if idx == len(display) - 1 and not truncated else "├── "
        lines.append(f"{prefix}{name}")
    if truncated:
        lines.append(f"└── … {len(file_names) - len(display)} more")
    return "\n".join(lines)


def summarize_step(step: Mapping[str, Any]) -> StepSummary:
    """Collect human-readable summaries for a single step dictionary."""
    step_fields: list[FieldSummary] = []
    observation_fields: list[FieldSummary] = []

    for key, value in step.items():
        if key == "observation" and isinstance(value, Mapping):
            for obs_key, obs_val in value.items():
                if isinstance(obs_val, (bytes, bytearray)):
                    if is_printable_text(obs_val):
                        text = obs_val.decode("utf-8", errors="replace")
                        observation_fields.append(
                            FieldSummary(
                                name=obs_key,
                                type_hint="str",
                                description=f"text sample (len={len(text)})",
                            )
                        )
                        continue
                    image_summary = describe_image(obs_val)
                    if image_summary:
                        image_summary.name = obs_key
                        image_summary.description = (
                            f"{image_summary.image_format} image "
                            f"{image_summary.width}x{image_summary.height} (mode={image_summary.description.split('mode=')[-1]})"
                        )
                        observation_fields.append(image_summary)
                        continue
                    observation_fields.append(
                        FieldSummary(
                            name=obs_key,
                            type_hint="bytes",
                            description=f"binary payload (len={len(obs_val)})",
                        )
                    )
                elif isinstance(obs_val, np.ndarray):
                    type_hint, description = describe_numpy_array(obs_val)
                    observation_fields.append(
                        FieldSummary(
                            name=obs_key,
                            type_hint=type_hint,
                            description=description,
                        )
                    )
                else:
                    observation_fields.append(
                        FieldSummary(
                            name=obs_key,
                            type_hint=type(obs_val).__name__,
                            description=None,
                        )
                    )
        else:
            if isinstance(value, np.ndarray):
                type_hint, description = describe_numpy_array(value)
                step_fields.append(
                    FieldSummary(name=key, type_hint=type_hint, description=description)
                )
            elif isinstance(value, (np.floating, float)):
                step_fields.append(FieldSummary(name=key, type_hint="float"))
            elif isinstance(value, (np.integer, int)):
                step_fields.append(FieldSummary(name=key, type_hint="int"))
            elif isinstance(value, (np.bool_, bool)):
                step_fields.append(FieldSummary(name=key, type_hint="bool"))
            elif isinstance(value, (bytes, bytearray)):
                if is_printable_text(value):
                    text = value.decode("utf-8", errors="replace")
                    step_fields.append(
                        FieldSummary(
                            name=key,
                            type_hint="str",
                            description=f"text sample (len={len(text)})",
                        )
                    )
                else:
                    step_fields.append(
                        FieldSummary(
                            name=key,
                            type_hint="bytes",
                            description=f"binary payload (len={len(value)})",
                        )
                    )
            else:
                step_fields.append(
                    FieldSummary(name=key, type_hint=type(value).__name__)
                )

    return StepSummary(fields=step_fields, observation_fields=observation_fields)


def extract_duration(metadata: Mapping[str, Any]) -> float | None:
    """Extract a duration in seconds from episode metadata if present."""
    for key, value in metadata.items():
        key_lower = key.lower()
        if not isinstance(value, (int, float, np.floating, np.integer)):
            continue
        if "duration" in key_lower or "seconds" in key_lower:
            return float(value)
    return None


def decode_instruction(value: Any) -> str | None:
    """Return the language instruction as a string if available."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def analyze_episode(name: str, payload: Mapping[str, Any]) -> EpisodeSample:
    """Profile a single episode payload loaded from pickle."""
    steps: Sequence[Mapping[str, Any]] = payload.get("steps", [])  # type: ignore[assignment]
    num_steps = len(steps)
    metadata = payload.get("episode_metadata", {})  # type: ignore[assignment]
    duration_seconds = extract_duration(metadata) if isinstance(metadata, Mapping) else None

    step_summary = (
        summarize_step(steps[0])
        if steps
        else StepSummary(fields=[], observation_fields=[])
    )

    instructions: list[str] = []
    for step in steps:
        instruction = decode_instruction(step.get("language_instruction"))
        if instruction:
            instructions.append(instruction)

    return EpisodeSample(
        name=name,
        num_steps=num_steps,
        duration_seconds=duration_seconds,
        instructions=instructions,
        step_summary=step_summary,
    )


def channel_count_from_mode(mode: str | None) -> int | None:
    """Return the number of image channels for a PIL mode."""
    if mode is None:
        return None
    band_counts = {
        "1": 1,
        "L": 1,
        "P": 1,
        "RGB": 3,
        "RGBA": 4,
        "CMYK": 4,
        "YCbCr": 3,
        "LAB": 3,
    }
    return band_counts.get(mode)


def update_camera_summaries(
    profile: DatasetProfile, observation_fields: Iterable[FieldSummary]
) -> None:
    """Update dataset-level camera summaries based on observation fields."""
    for field_summary in observation_fields:
        if isinstance(field_summary, ObservationSummary):
            profile.camera_summaries.setdefault(field_summary.name, field_summary)


def update_timeseries_summaries(
    profile: DatasetProfile, step_fields: Iterable[FieldSummary], observation_fields: Iterable[FieldSummary]
) -> None:
    """Track non-image numeric arrays (timeseries) across the dataset."""
    for field_summary in list(step_fields) + list(observation_fields):
        if isinstance(field_summary, ObservationSummary):
            # camera already tracked
            continue
        if "numpy.ndarray" in (field_summary.description or "") or field_summary.type_hint.startswith("list["):
            profile.timeseries_summaries.setdefault(field_summary.name, field_summary)


def profile_tar_archive(
    dataset_profile: DatasetProfile,
    archive_path: Path,
    *,
    max_episode_samples: int,
) -> ArchiveProfile:
    """Extract representative episodes from a tarball of pickled episodes."""
    with tarfile.open(archive_path) as tar:
        all_members = [
            member
            for member in tar.getmembers()
            if member.isfile() and member.name.endswith(".data.pickle")
        ]
        sample_members = all_members[:max_episode_samples]
        episode_samples: list[EpisodeSample] = []

        for member in sample_members:
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            payload = pickle.load(extracted)
            episode_sample = analyze_episode(member.name, payload)
            episode_samples.append(episode_sample)

            dataset_profile.step_counts.append(episode_sample.num_steps)
            if episode_sample.duration_seconds is not None:
                dataset_profile.durations.append(episode_sample.duration_seconds)
            dataset_profile.instruction_counter.update(episode_sample.instructions)
            update_camera_summaries(
                dataset_profile, episode_sample.step_summary.observation_fields
            )
            update_timeseries_summaries(
                dataset_profile,
                episode_sample.step_summary.fields,
                episode_sample.step_summary.observation_fields,
            )
            if dataset_profile.schema_sample is None:
                dataset_profile.schema_sample = episode_sample

    dataset_profile.total_episodes += len(all_members)
    dataset_profile.sampled_episode_count += len(episode_samples)

    tree_preview = build_tree_preview([member.name for member in all_members])

    return ArchiveProfile(
        path=archive_path,
        size_bytes=archive_path.stat().st_size,
        entries=len(all_members),
        sample_names=[member.name for member in sample_members],
        episode_samples=episode_samples,
        tree_preview=tree_preview,
    )


def summarise_instructions(instruction_counter: Counter[str], max_examples: int = 5) -> list[str]:
    """Return the most common instructions with counts."""
    return [
        f"{text!r} (×{count})"
        for text, count in instruction_counter.most_common(max_examples)
    ]


def summarize_statistic(values: Sequence[int | float]) -> str:
    """Return a concise summary (min/mean/max) for numeric sequences."""
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]}"
    mean_val = statistics.fmean(values)
    return f"min={min(values)}, avg={mean_val:.2f}, max={max(values)}"


def render_camera_table(cameras: Mapping[str, ObservationSummary]) -> str:
    """Render a Markdown table describing cameras."""
    if not cameras:
        return "_no camera streams detected_"
    header = "| camera | resolution (WxH) | channels | encoding |\n| --- | --- | --- | --- |"
    rows = [
        f"| {name} | {info.width}×{info.height} | {info.channels} | {info.image_format or 'n/a'} |"
        for name, info in cameras.items()
    ]
    return "\n".join([header, *rows])


def render_timeseries_table(timeseries: Mapping[str, FieldSummary]) -> str:
    """Render a Markdown table describing numeric timeseries."""
    if not timeseries:
        return "_no numeric timeseries detected_"
    header = "| signal | type | notes |\n| --- | --- | --- |"
    rows = [
        f"| {name} | {summary.type_hint} | {summary.description or ''} |"
        for name, summary in timeseries.items()
    ]
    return "\n".join([header, *rows])


def render_pydantic_schema(dataset_profile: DatasetProfile) -> str:
    """Generate Pydantic BaseModel definitions as a string."""
    sample = dataset_profile.schema_sample
    if sample is None or not sample.step_summary:
        return "# Unable to infer schema (no episode samples captured)\n"
    base_name = "".join(part.capitalize() for part in dataset_profile.name.split("_"))

    observation_lines = []
    for field_summary in sample.step_summary.observation_fields:
        desc = (
            f"Field(description={field_summary.description!r})"
            if field_summary.description
            else ""
        )
        annotation = (
            f"Annotated[{field_summary.type_hint}, {desc}]"
            if desc
            else field_summary.type_hint
        )
        observation_lines.append(f"    {field_summary.name}: {annotation}")

    step_lines = []
    for field_summary in sample.step_summary.fields:
        desc = (
            f"Field(description={field_summary.description!r})"
            if field_summary.description
            else ""
        )
        annotation = (
            f"Annotated[{field_summary.type_hint}, {desc}]"
            if desc
            else field_summary.type_hint
        )
        if field_summary.name == "observation":
            continue  # handled separately
        step_lines.append(f"    {field_summary.name}: {annotation}")

    instructions_summary = (
        summarize_statistic(dataset_profile.step_counts)
        if dataset_profile.step_counts
        else str(sample.num_steps)
    )

    schema = textwrap.dedent(
        f"""
        ```python
        from typing import Annotated, List
        from pydantic import BaseModel, Field


        class {base_name}Observation(BaseModel):
        """
    ).strip("\n")

    if observation_lines:
        schema += "\n" + "\n".join(observation_lines)
    else:
        schema += "\n    model_config = {{\"extra\": \"allow\"}}"

    schema += textwrap.dedent(
        f"""


        class {base_name}Step(BaseModel):
    """
    ).rstrip("\n")

    if step_lines:
        schema += "\n" + "\n".join(step_lines)

    schema += f"\n    observation: {base_name}Observation"

    schema += textwrap.dedent(
        f"""


        class {base_name}Episode(BaseModel):
            episode_metadata: dict
            steps: Annotated[List[{base_name}Step], Field(description="sampled {instructions_summary}")]
            image_list: list[str]
        ```
        """
    )
    return schema


def build_dataset_markdown(profile: DatasetProfile) -> str:
    """Create a Markdown summary for a dataset profile."""
    lines = [
        f"### `{profile.name}`",
        "",
        f"- downloaded files: {profile.total_downloaded_files}",
        f"- downloaded bytes: {humanize_bytes(profile.total_bytes)}",
        f"- episodes in sampled archives: {profile.total_episodes}",
        f"- sampled episodes analysed: {profile.sampled_episode_count}",
        f"- steps per episode: {summarize_statistic(profile.step_counts)}",
    ]
    if profile.durations:
        lines.append(f"- episode durations (seconds): {summarize_statistic(profile.durations)}")
    else:
        lines.append("- episode durations (seconds): n/a")

    if profile.instruction_counter:
        lines.append("- frequent instructions: " + ", ".join(summarise_instructions(profile.instruction_counter)))
    else:
        lines.append("- frequent instructions: n/a")

    lines.append("")
    lines.append(render_camera_table(profile.camera_summaries))
    lines.append("")
    lines.append(render_timeseries_table(profile.timeseries_summaries))

    for archive in profile.archives:
        lines.append("")
        lines.append(f"**Archive** `{archive.path.name}` ({humanize_bytes(archive.size_bytes)})")
        lines.append("")
        lines.append("```")
        lines.append(archive.tree_preview or "<empty archive>")
        lines.append("```")

    lines.append("")
    lines.append(render_pydantic_schema(profile))
    lines.append("")

    return "\n".join(lines)


def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def cleanup_archive(path: Path, root: Path) -> None:
    """Remove an archive from disk and prune empty directories up to root."""
    if path.exists():
        path.unlink()
    # Remove empty parent directories but never climb above the root.
    current = path.parent
    try:
        root_resolved = root.resolve()
    except FileNotFoundError:
        root_resolved = root
    while current != current.parent and current.exists():
        try:
            if current.resolve() == root_resolved:
                break
        except FileNotFoundError:
            break
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and profile a subset of OpenX-Embodiment datasets."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repository ID (default: %(default)s).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific dataset folders to profile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Number of datasets to profile when --datasets is not provided (default: 3).",
    )
    parser.add_argument(
        "--archives-per-dataset",
        type=int,
        default=2,
        help="Maximum number of archive files to download per dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--episodes-per-archive",
        type=int,
        default=3,
        help="Number of episodes to sample from each archive (default: %(default)s).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/openx_profiles"),
        help="Base directory where downloaded archives are stored (default: %(default)s).",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/openx_profile_summary.md"),
        help="Path to the Markdown summary that will be written (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip downloads and only show which datasets/files would be processed.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives on disk instead of deleting them after profiling.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=args.repo_id,
        repo_type="dataset",
    )
    dataset_files = iter_dataset_files(repo_files)
    available_datasets = sorted(dataset_files)
    selected = select_datasets(
        available=available_datasets, requested=args.datasets, limit=args.limit
    )

    ensure_directory(args.download_dir)
    ensure_directory(args.output_markdown.parent)

    profiles: list[DatasetProfile] = []
    for dataset in selected:
        files = sorted(dataset_files.get(dataset, []))
        if not files:
            print(f"[warn] No files found for dataset {dataset}", file=sys.stderr)
            continue
        profile = DatasetProfile(name=dataset)
        chosen_files = [
            path
            for path in files
            if path.lower().endswith((".tar", ".tgz", ".tar.gz"))
        ][: args.archives_per_dataset]
        if not chosen_files:
            print(
                f"[warn] {dataset}: no archive files (.tar/.tgz) detected; skipping",
                file=sys.stderr,
            )
            continue

        if args.dry_run:
            print(f"{dataset}: would download {len(chosen_files)} archive(s)")
            continue

        for path in chosen_files:
            local_path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                allow_patterns=[path],
                local_dir=args.download_dir,
                local_dir_use_symlinks=False,
            )
            archive_path = args.download_dir / path
            if not archive_path.exists():
                # huggingface_hub>=0.23 returns the directory path; ensure file resolves
                archive_path = Path(local_path)
                if archive_path.is_dir():
                    archive_path = archive_path / Path(path).name
            if not archive_path.exists():
                raise FileNotFoundError(f"Archive not found after download: {archive_path}")

            profile.total_downloaded_files += 1
            profile.total_bytes += archive_path.stat().st_size
            archive_profile = profile_tar_archive(
                profile,
                archive_path,
                max_episode_samples=args.episodes_per_archive,
            )
            profile.archives.append(archive_profile)
            if not args.keep_archives:
                cleanup_archive(archive_path, args.download_dir)

        profiles.append(profile)

    if args.dry_run:
        return 0

    markdown_sections = [
        "# OpenX Dataset Profile Summary",
        "",
        f"Analysed datasets: {', '.join(profile.name for profile in profiles) if profiles else 'none'}",
        "",
    ]
    for profile in profiles:
        markdown_sections.append(build_dataset_markdown(profile))
        markdown_sections.append("")

    args.output_markdown.write_text("\n".join(markdown_sections))
    print(f"Wrote summary to {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
