# A/B Testing Kit — D3

**Tier:** 3 🔥 · **Category:** D — Analytics & data science · **Wave:** 3

Root rules in `../GUIDELINES.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

An experiment design and analysis toolkit for product A/B tests. Four things a product-DS
interview actually probes: sample-size/power analysis up front, what continuous monitoring
("peeking") does to the false-positive rate and how sequential methods fix it, CUPED
variance reduction using pre-experiment data, and frequentist vs Bayesian decision rules on
the same experiment.

Tier 3 means this must be visibly the author's: own framing, own simulation-based evaluation
method, own architecture writeup. Every claim about error rates is to be **demonstrated by
simulation in this repo**, not quoted from a blog post.

## Stack

Python 3.13 · numpy · scipy · pytest · ruff · uv. (matplotlib arrives with the notebook
in session 3; pandas is not needed and was not added.)
No services, no Docker, no GPU, no API keys. Runs entirely locally and in CI.

## Acceptance criteria

- [x] Power / MDE calculator for proportions and means, validated against simulation
- [x] Peeking demonstration: measured Type-I error inflation vs the nominal 5%
- [x] A sequential/always-valid test that holds its error rate under the same peeking
- [ ] CUPED implemented, with measured variance reduction vs the covariate correlation
- [ ] Bayesian analysis (posterior, P(B>A), expected loss) compared to the frequentist call
- [ ] README §5 carries real numbers, all reproducible from this repo
- [ ] Ship gate passes (`/ship`)

## Project-specific notes

- **Synthetic data by design.** There is no public dataset with the pre-experiment covariates
  CUPED needs plus known ground truth. Simulation *is* the evaluation method here — every
  error-rate claim needs a known true effect to be checkable against. Say this plainly in the
  README next to every number, per the anti-slop rule.
- **Seed everything.** Tests assert on simulation output; unseeded RNG means flaky CI.
- **Tests check correctness, the CLI produces results.** Do not raise `n_sims` in tests to
  sharpen a number — run the CLI at 20,000 reps instead and quote the command. Keeps the
  suite near 60s.
- `estimate_bias` is a *signed* mean, so it cancels to ~0 under a two-sided null. Read
  magnitude, not sign, when the true effect is zero. See README §5.3.
- Local pytest needs `--basetemp=<scratchpad>/pt` (sandbox blocks `%TEMP%`). Never put that
  in `pyproject.toml`.
