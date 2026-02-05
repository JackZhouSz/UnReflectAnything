"""Command-line interface for UnReflectAnything."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run_train(args: argparse.Namespace) -> None:
    """Dispatch to main.run_pipeline(mode='train') with current argv."""
    # run_pipeline parses sys.argv; replace with [prog, ...train args]
    argv = sys.argv
    sys.argv = [argv[0]] + args.passthrough
    try:
        import main
        main.run_pipeline(mode="train")
    finally:
        sys.argv = argv


def _run_test(args: argparse.Namespace) -> None:
    """Dispatch to main.run_pipeline(mode='test') with current argv."""
    argv = sys.argv
    sys.argv = [argv[0]] + args.passthrough
    try:
        import main
        main.run_pipeline(mode="test")
    finally:
        sys.argv = argv


def _run_inference(args: argparse.Namespace) -> None:
    """Dispatch to inference entry: parse config and run inference."""
    argv = sys.argv
    # inference.parse_cli() only parses --config; do not pass extra args
    sys.argv = [argv[0], "--config", str(Path(args.config).resolve())]
    try:
        import inference
        inference.main()
    finally:
        sys.argv = argv


def _run_sweep(args: argparse.Namespace) -> None:
    """Launch a Weights & Biases sweep."""
    import subprocess
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")
    cmd = ["wandb", "sweep", str(config_path)] + args.passthrough
    sys.exit(subprocess.run(cmd).returncode)


def _run_agent(args: argparse.Namespace) -> None:
    """Run a W&B sweep agent."""
    import subprocess
    cmd = ["wandb", "agent"] + args.passthrough
    sys.exit(subprocess.run(cmd).returncode)


def _run_completion(args: argparse.Namespace) -> None:
    """Print shell completion script."""
    try:
        from importlib.resources import files
        pkg = files("unreflectanything")
    except Exception:
        # Python < 3.9 fallback
        import importlib.resources
        pkg = importlib.resources.files("unreflectanything")
    shell = (args.shell or "").strip().lower()
    if "zsh" in shell:
        path = pkg / "data" / "unreflect-completion.zsh"
    else:
        path = pkg / "data" / "unreflect-completion.bash"
    text = path.read_text(encoding="utf-8")
    print(text, end="")


def _run_download_weights(args: argparse.Namespace) -> None:
    """Download pretrained weights to cache or specified directory."""
    from unreflectanything.weights import download_weights
    download_weights(
        output_dir=Path(args.output_dir),
        variant=args.variant,
        force=args.force,
    )


def main() -> None:
    """Entry point for the unreflectanything console script."""
    parser = argparse.ArgumentParser(
        prog="unreflectanything",
        description="UnReflectAnything: remove specular reflections from RGB images.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("passthrough", nargs="*", help="Arguments passed to train (e.g. --config, --resume-run)")
    p_train.set_defaults(func=_run_train)

    # test
    p_test = subparsers.add_parser("test", help="Run evaluation / testing")
    p_test.add_argument("passthrough", nargs="*", help="Arguments passed to test")
    p_test.set_defaults(func=_run_test)

    # inference
    p_inf = subparsers.add_parser("inference", help="Run inference on an image directory")
    p_inf.add_argument(
        "--config", "-c",
        type=str,
        default="config_inference.yaml",
        help="Path to inference YAML config (default: config_inference.yaml)",
    )
    p_inf.set_defaults(func=_run_inference)

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Launch a Weights & Biases sweep")
    p_sweep.add_argument(
        "--config",
        type=str,
        default="config_sweep.yaml",
        help="Path to sweep config YAML (default: config_sweep.yaml)",
    )
    p_sweep.add_argument("passthrough", nargs="*", help="Arguments passed to wandb sweep")
    p_sweep.set_defaults(func=_run_sweep)

    # agent
    p_agent = subparsers.add_parser("agent", help="Run a W&B sweep agent")
    p_agent.add_argument("passthrough", nargs="*", help="Arguments passed to wandb agent (e.g. sweep ID)")
    p_agent.set_defaults(func=_run_agent)

    # completion
    p_comp = subparsers.add_parser("completion", help="Print shell completion script")
    p_comp.add_argument(
        "shell",
        nargs="?",
        default="",
        help="Shell: bash or zsh (default: infer from $SHELL)",
    )
    p_comp.set_defaults(func=_run_completion)

    # download-weights
    p_dl = subparsers.add_parser("download-weights", help="Download pretrained weights")
    p_dl.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save weights (default: cache dir)",
    )
    p_dl.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Weights variant to download (default: default)",
    )
    p_dl.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-download even if already present",
    )
    p_dl.set_defaults(func=_run_download_weights)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)
    # Resolve output_dir for download-weights
    if args.subcommand == "download-weights" and args.output_dir is None:
        from unreflectanything.weights import get_weights_cache_dir
        args.output_dir = str(get_weights_cache_dir())
    args.func(args)
