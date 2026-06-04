# Texthunt — Engineering Conventions

Probabilistic authorship identification for short, informal text (Slack messages). Given example
writing from many people, estimate who wrote an unattributed snippet — and say "unknown" when the
author is likely not among the candidates (open-set verification).

## Workflow

- **Test-first (TDD).** Write a failing `pytest` test that names the behaviour, make it pass, then
  refactor. A module and its tests land together. Keep tests deterministic with small in-repo
  fixtures; never download models or hit the network in unit tests — mark those `@pytest.mark.slow`
  or mock the boundary.
- **Small, focused changes**, each ending in one conventional commit.

## Code style

- Self-documenting: intention-revealing names, small single-purpose functions, types as
  documentation. Aim for code so clear that comments are nearly unnecessary.
- Comments explain *why* (a non-obvious trade-off), never restate *what* the code does.
- Full type hints on public functions. Model the domain with `pydantic` / dataclasses
  (`Message`, `AuthorProfile`, `Verdict`) rather than passing loose dicts/tuples.
- Pure functions where practical; isolate I/O (file reads, model loading) at the edges.

## Commits

Conventional Commits, concise, describing the **value added** — not a list of edits.

- Good: `feat: reject unknown authors below the EER threshold`
- Bad: `add threshold var, edit verify.py, update test`

Types: `feat`, `fix`, `refactor`, `test`, `chore`, `docs`.

*CRITICAL*: Do not commit any secrets or keys.

## Tooling (uv — never pip)

```sh
uv run pytest          # tests
uv run ruff check      # lint
uv run ruff format     # format
uv run ty check        # type check
./scripts/check.sh     # all of the above — run before every commit
```

Add deps with `uv add` / `uv add --dev`. Commit `uv.lock`.

Put throwaway scripts and scratch output in `./tmp` (gitignored), never `/tmp` — keeping them inside
the repo avoids sandbox permission prompts.

## Data & privacy

- Slack messages are personal data. Everything under `data/` is gitignored — never commit raw
  exports.
- Pseudonymise author IDs; normalise `@mentions`, URLs, and channel refs to placeholders so models
  learn *style*, not *who talks to whom*.
- Only use writing you have consent to analyse.

## Architecture

Two interchangeable scoring engines behind one interface, so they can be compared on identical
splits:

- **Engine A** — classical stylometry: char/word n-gram TF-IDF + explicit style features.
- **Engine B** — neural style embeddings (LUAR).

Both build a per-author profile vector, score a query by cosine similarity, and apply a rejection
threshold (`τ_reject`, chosen at the Equal Error Rate) to support the "unknown author" answer.
See the plan and `README.md` for the fuller rationale.
