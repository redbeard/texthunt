"""Command-line entry point for Texthunt.

Two subcommands over a Slack export directory:

- ``evaluate`` — honest open-set metrics on held-out authors and (optionally) held-out channels.
- ``identify`` — train on the whole corpus, then attribute a snippet with calibrated probabilities.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

from texthunt import __version__
from texthunt.engine import Engine
from texthunt.evaluate import author_disjoint_split, evaluate_engine, topic_aware_split
from texthunt.features import build_stylometric_engine
from texthunt.pipeline import DEFAULT_MIN_CHARS, load_blocks, train_identifier
from texthunt.verify import Verdict


def _engine_factory(name: str) -> Callable[[], Engine]:
    if name == "luar":
        from texthunt.embeddings import build_luar_engine

        return build_luar_engine
    return build_stylometric_engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="texthunt",
        description="Probabilistic authorship identification for short, informal text.",
    )
    parser.add_argument("--version", action="version", version=f"texthunt {__version__}")
    subcommands = parser.add_subparsers(dest="command")

    evaluate = subcommands.add_parser("evaluate", help="measure open-set accuracy on a corpus")
    _add_corpus_options(evaluate)
    evaluate.add_argument(
        "--topic-aware",
        action="store_true",
        help="query on channels held out of the gallery, measuring style over topic",
    )
    evaluate.set_defaults(run=_run_evaluate)

    identify = subcommands.add_parser("identify", help="attribute a snippet of text")
    _add_corpus_options(identify)
    identify.add_argument("text", help="the snippet to attribute")
    identify.set_defaults(run=_run_identify)

    return parser


def _add_corpus_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True, help="path to a Slack export directory")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--engine",
        choices=("stylometry", "luar"),
        default="stylometry",
        help="scoring engine: classical stylometry or LUAR style embeddings",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "run"):
        parser.print_help()
        return 0
    return args.run(args)


def _run_evaluate(args: argparse.Namespace) -> int:
    blocks = load_blocks(args.data, args.min_chars)
    split = (
        topic_aware_split(blocks, unknown_fraction=0.3, seed=args.seed)
        if args.topic_aware
        else author_disjoint_split(blocks, unknown_fraction=0.3, query_fraction=0.5, seed=args.seed)
    )
    report = evaluate_engine(split, _engine_factory(args.engine))
    print(f"known authors:      {report.n_known_authors}")
    print(f"queries:            {report.n_queries}")
    print(f"top-1 accuracy:     {report.top1_accuracy:.1%}  (random {report.random_baseline:.1%})")
    print(f"top-5 accuracy:     {report.top5_accuracy:.1%}")
    print(f"verification AUC:   {report.verification_auc:.3f}")
    print(f"equal error rate:   {report.eer:.1%}")
    print(f"reject threshold:   {report.reject_threshold:.3f}")
    return 0


def _run_identify(args: argparse.Namespace) -> int:
    blocks = load_blocks(args.data, args.min_chars)
    identifier = train_identifier(blocks, _engine_factory(args.engine), seed=args.seed)
    _print_verdict(identifier.identify(args.text))
    return 0


def _print_verdict(verdict: Verdict) -> None:
    if verdict.is_unknown:
        print(f"likely unknown author (P={verdict.probability_unknown:.1%})\n")
    else:
        print(f"most likely: {verdict.prediction}\n")
    print(f"{'unknown':<20} {verdict.probability_unknown:.1%}")
    for candidate in verdict.ranked:
        print(f"{candidate.author_id:<20} {candidate.probability:.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
