"""
Utility helpers to visualize the module hierarchy of a PyTorch model.

The main entry point is `visualize_model`, which renders a Graphviz diagram
showing the connections between submodules and summarizing parameter shapes.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import torch

from nanochat.checkpoint_manager import load_model


def _human_readable(num: int) -> str:
    """Pretty print parameter counts."""
    if num == 0:
        return "0"
    units = ["", "K", "M", "B", "T"]
    magnitude = max(0, min(len(units) - 1, int(math.log10(num) // 3)))
    scaled = num / (1000 ** magnitude)
    return f"{scaled:.1f}{units[magnitude]}"


def _sanitize_node_id(name: str) -> str:
    """Graphviz-friendly node id."""
    return "root" if name == "" else name.replace(".", "__")


def _module_depth(name: str) -> int:
    if not name:
        return 0
    return name.count(".") + 1


def visualize_model(
    model: torch.nn.Module,
    output_path: Path | str,
    *,
    max_depth: Optional[int] = 2,
    include_parameters: bool = True,
    max_params_per_module: int = 3,
    graph_format: Optional[str] = None,
) -> Path:
    """
    Render the module hierarchy of `model` to `output_path`.

    The function exports a Graphviz DOT file alongside the image. If the
    `dot` binary is available on the system, an image is produced immediately.
    Otherwise, only the DOT file is written and the caller can render it later.
    """
    if isinstance(output_path, str):
        output_path = Path(output_path)

    if graph_format is None:
        graph_format = output_path.suffix.lstrip(".")
        if not graph_format:
            graph_format = "png"
            output_path = output_path.with_suffix(".png")

    dot_path = output_path.with_suffix(".dot")

    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    allowed_modules = {}
    for name, module in modules.items():
        depth = _module_depth(name)
        if max_depth is not None and depth > max_depth:
            continue
        allowed_modules[name] = module

    lines = [
        "digraph ModelGraph {",
        '  rankdir=LR;',
        '  graph [fontname="Helvetica"];',
        '  node [shape=record, fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    for name, module in allowed_modules.items():
        node_id = _sanitize_node_id(name)
        class_name = module.__class__.__name__
        depth = _module_depth(name)
        if name == "":
            subtree_params = list(parameters.items())
        else:
            prefix = name + "."
            subtree_params = [
                (param_name, tensor)
                for param_name, tensor in parameters.items()
                if param_name == name or param_name.startswith(prefix)
            ]
        direct_params = [
            (param_name, tensor)
            for param_name, tensor in parameters.items()
            if (param_name.rsplit(".", 1)[0] if "." in param_name else "") == name
        ]
        param_total = sum(t.numel() for _, t in subtree_params)
        label_lines = [class_name, f"Params: {_human_readable(param_total)}"]

        if include_parameters and direct_params:
            listed = direct_params[:max_params_per_module]
            snippets = [
                f"{pname.split('.')[-1]} ({'×'.join(str(s) for s in tensor.shape)})"
                for pname, tensor in listed
            ]
            if len(direct_params) > max_params_per_module:
                snippets.append(f"... +{len(direct_params) - len(listed)} more")
            label_lines.append("\\n".join(snippets))

        label = "\\n".join(label_lines)
        lines.append(f'  {node_id} [label="{label}"];')

    for name in allowed_modules:
        if name == "":
            continue
        parent = name.rsplit(".", 1)[0]
        if parent not in allowed_modules and parent != "":
            # Walk up until a visible parent is found
            while parent not in allowed_modules and parent:
                parent = parent.rsplit(".", 1)[0]
        parent_id = _sanitize_node_id(parent)
        child_id = _sanitize_node_id(name)
        lines.append(f"  {parent_id} -> {child_id};")

    lines.append("}")
    dot_path.write_text("\n".join(lines))

    dot_exe = shutil.which("dot")
    if dot_exe is None:
        return dot_path

    subprocess.run(
        [dot_exe, f"-T{graph_format}", str(dot_path), "-o", str(output_path)],
        check=True,
    )
    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the nanochat model graph.")
    parser.add_argument("--source", default="sft", help="Model source (base|mid|sft|rl)")
    parser.add_argument("--model-tag", default=None, help="Checkpoint tag (e.g., d32)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for loading the model (default: cpu to avoid GPU OOM)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum module depth to include in the graph",
    )
    parser.add_argument(
        "--include-parameters",
        action="store_true",
        help="Annotate modules with direct parameter shapes",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path (DOT file saved alongside)",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Graphviz output format (png, svg, etc.). Defaults to output suffix.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    device = torch.device(args.device)
    torch.manual_seed(42)

    model, _, _ = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    try:
        output_path = visualize_model(
            model,
            args.output,
            max_depth=args.max_depth,
            include_parameters=args.include_parameters,
            graph_format=args.format,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Visualization written to {output_path}")


if __name__ == "__main__":
    _main()
