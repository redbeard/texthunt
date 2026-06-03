# Texthunt

Probabilistic authorship identification for short, informal text.

Given example writing from many people (e.g. Slack messages), Texthunt estimates **who wrote** an
unattributed snippet — and, crucially, can answer **"unknown"** when the true author is probably not
among the candidates. In the literature this is *open-set authorship verification*: the hard case,
because the text is short and the population of authors is large.

## How it works

A single short message rarely identifies anyone, so Texthunt **blocks** several messages per author
into a *profile* and scores a query block against every profile by cosine similarity. Two
interchangeable engines produce the vectors:

- **Engine A — classical stylometry.** Character/word n-gram TF-IDF plus explicit style features
  (punctuation, emoji, casing, length, function words). Fast, interpretable, CPU-only.
- **Engine B — neural style embeddings.** [LUAR](https://github.com/LLNL/LUAR), contrastively
  trained to capture *style* rather than *topic* and to aggregate many short posts into one author
  embedding.

If the best author's score falls below a calibrated **rejection threshold** (`τ_reject`, chosen at
the Equal Error Rate), Texthunt reports *unknown* rather than guessing. Scores are calibrated into
probabilities, so results are genuinely probabilistic.

> **Topic ≠ style.** Naive content features learn *what* a person talks about, not *how* they write.
> Texthunt mitigates this with character n-grams, style-trained embeddings, and topic-aware
> evaluation splits.

## Quickstart

```sh
uv sync                       # install
uv run pytest                 # run the tests
uv run texthunt --help        # CLI
```

## Privacy

Slack messages are personal data. Raw exports live under `data/` (gitignored) and are never
committed; author IDs are pseudonymised and mentions/URLs normalised. Only analyse writing you have
consent to use.

## Development

See [AGENTS.md](AGENTS.md) for engineering conventions (test-first, self-documenting code,
conventional commits, `uv` tooling). Run `./scripts/check.sh` before every commit.
