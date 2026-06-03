"""Command-line entry point for Texthunt.

Subcommands are added as the engines and evaluation harness land; for now this wires up the
argument parser so `texthunt --help` works.
"""

import argparse

from texthunt import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="texthunt",
        description="Probabilistic authorship identification for short, informal text.",
    )
    parser.add_argument("--version", action="version", version=f"texthunt {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
