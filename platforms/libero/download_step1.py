import argparse
import os
import sys
import time
import types
from pathlib import Path

LIBERO_ROOT = os.environ.get(
    "LIBERO_ROOT",
    str(Path(__file__).resolve().parents[2].parent / "LIBERO"),
)
if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

DEFAULT_DOWNLOAD_DIR = str(
    Path(os.environ.get("MASKWAM_DATA_ROOT", "realdata")).expanduser() / "libero"
)


def load_download_utils(download_dir):
    """Import LIBERO without its first-run interactive config prompt."""
    config_dir = Path(
        os.path.expanduser(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero"))
    )
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        benchmark_root = Path(LIBERO_ROOT).resolve() / "libero" / "libero"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text(
            "\n".join(
                [
                    f"assets: {benchmark_root / 'assets'}",
                    f"bddl_files: {benchmark_root / 'bddl_files'}",
                    f"benchmark_root: {benchmark_root}",
                    f"datasets: {Path(download_dir).expanduser().resolve()}",
                    f"init_states: {benchmark_root / 'init_files'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Initialized LIBERO config: {config_file}")

    # LIBERO only uses termcolor for terminal decoration. Keep downloading
    # functional in minimal environments where this optional package is absent.
    try:
        import termcolor  # pyright: ignore[reportMissingImports]  # noqa: F401
    except ModuleNotFoundError:
        termcolor_fallback = types.ModuleType("termcolor")
        termcolor_fallback.colored = lambda text, *args, **kwargs: text
        sys.modules["termcolor"] = termcolor_fallback

    import libero.libero.utils.download_utils as download_utils  # pyright: ignore[reportMissingImports]

    return download_utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=DEFAULT_DOWNLOAD_DIR,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100", "libero_10"],
        default="all",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face instead of original download links"
    )
    return parser.parse_args()


def main():

    args = parse_args()
    download_utils = load_download_utils(args.download_dir)

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    if args.use_huggingface:
        print("Using Hugging Face as the download source")
    else:
        print("Using original download links (note: these may expire soon)")
        input_str = input("Download from original links may lead to failures. Do you want to continue? (y/n): ")
        if input_str.lower() != 'y':
            print("Switching to Hugging Face as the download source...")
            args.use_huggingface = True

    # If not, download
    download_utils.libero_dataset_download(
        download_dir=args.download_dir,
        datasets=args.datasets,
        use_huggingface=args.use_huggingface
    )


    # wait for 1 second
    time.sleep(1)
    print("\n\n\n")

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
