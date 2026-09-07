# Build Notes — A/B Testing Kit

Working notes: what broke, what was tried, why X over Y.
Not for recruiters — for me, six months from now, in an interview.

Keep it rough. Rough is the point.

---

## Log

### 2026-09-07 — session 1: simulation, design, peeking

- **Built:** `simulation.py` (seeded generation + shared result types), `design.py`
  (power/n/MDE), `peeking.py` (Type-I inflation + winner's curse), `cli.py`. 111 tests.
- **Learned / decided:** the single most important structural choice was making
  `simulation.py` own generation, and storing units **in arrival order**. Everything
  downstream falls out of that. `truncate(k)` is only an honest model of "the data as of
  day 3" because outcomes are i.i.d., so any prefix is itself a valid sample. Without that
  property peeking cannot be simulated at all — I would have had to regenerate a smaller
  experiment at each look, which is a different (and wrong) thing: it would make each look
  independent, and the whole point is that consecutive looks are *correlated* because they
  share data.

- **Broke:** Wilson interval returned a lower bound of `3.47e-18` for a study with zero
  rejections. Analytically it is exactly 0.
  **Fixed by:** pinning the bounds at the extremes (`successes == 0` → 0.0,
  `successes == n` → 1.0) rather than loosening the test to `approx`. A study that never
  rejected once should print `[0.0000, ...]`, not scientific-notation noise.

- **Broke:** wrote `test_raises_when_no_effect_is_detectable` asserting that 30 units per
  arm on a 1% base rate cannot detect anything. Test failed — because the premise was
  wrong, not the code. A lift from 1% to ~100% at n=30 is *enormously* detectable
  (power 1.0000). The guard only fires when there is no **headroom**: at baseline 0.90
  there are 10 points of room, and at n=5 consuming all of it still tops out near 10%
  power. Rewrote the test around `baseline=0.90, n=5`.
  **Learned:** the branch exists for high baselines, not small samples — I had confused
  "small n" with "no achievable effect". Worth remembering: for a rate metric the MDE
  search is bounded by `1 - baseline`, which is a constraint that simply does not exist
  for a mean.

- **Broke:** test suite hit 117s, mostly the peeking studies (41s in one test).
  **Fixed by:** cutting `n_sims` in the slow tests and widening tolerances to match the
  Monte Carlo error at the smaller size (kept every threshold at ≥2.5 SE, so nothing
  became flaky). Headline README numbers come from a separate long CLI run at 20,000
  replications instead — the tests check *correctness*, the CLI produces *results*. Those
  are different jobs and it was a mistake to have been asking the test suite to do both.

- **Noted, not fixed:** the per-look cost is dominated by scalar scipy calls
  (`norm.sf`, `t.ppf`). A vectorised path over replications would be maybe 10–50× faster.
  Not doing it: the code being obviously correct matters more here than it being fast, and
  a reader checking the claims can afford to wait. Recorded in README §7 as the first thing
  to change at scale.

- **Open at end of session:** `sequential.py`, `cuped.py`, `bayes.py`, the notebook.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| **mSPRT** for the sequential test | Locked to group-sequential alpha spending instead. mSPRT is more conceptually and implementationally heavy than this toolkit's teaching objective needs — it requires choosing a mixing distribution, and the resulting boundary is hard to inspect. With O'Brien–Fleming-style alpha spending a reader can *see* the significance boundary tighten as looks accumulate, and can check the spent alpha sums to 0.05. Explainability is the deliverable here. |
| Real public dataset instead of simulation | No public dataset carries both the pre-experiment covariates CUPED needs **and** a known true effect. Without known ground truth, "measured Type-I error" is not measurable — there is nothing to be wrong against. Simulation is not a shortcut here, it is the only way the central claim can be checked. Stated plainly in the README next to every number. |
| One-sided tests | Would need a matching one-sided path in `fixed_horizon_test`. Shipping a `two_sided=False` parameter that no simulation validates would directly undercut the point of the repo. Two-sided everywhere, documented as a limitation. |
| Unpooled SE for the binary p-value | Pooling assumes the null, which is correct for a test statistic. Kept pooled for the p-value and unpooled for the CI, which means the two can disagree at the margin. That is correct behaviour, not a bug — noted in the docstring so a future me does not "fix" it. |
| Clamping under-sized looks up to the minimum | Would silently produce duplicate analysis points (two looks at the same n), quietly changing the schedule. `peek_schedule` drops them instead. |
| Averaging estimates over all replications | Would hide the winner's curse completely. `SimulationSummary.mean_estimate` averages over **rejections only** — the subset an experimenter would actually act on. |
| Keeping the scaffold's `.env.example` | This project reads no environment variables. A template `.env.example` naming services the project never touches is worse than no file (the lesson from A6's ship gate). |

## Open questions

- [ ] For binary metrics the CUPED covariate is generated through a latent-normal copula, so
      the realised correlation between the two 0/1 indicators is **attenuated** relative to
      the requested `covariate_corr`. Currently documented and asserted as a range rather
      than a value. Should `covariate_corr` instead be solved numerically so the *realised*
      correlation matches what the caller asked for? Cleaner API, more machinery.
- [ ] Is `MIN_LOOK_SIZE = 30` defensible, or just the usual folklore number? It is doing
      real work (it keeps aggressive schedules from measuring the normal approximation
      breaking down instead of the peeking effect), so it deserves a better justification
      than "30 is the number people use".
- [ ] `analytic_power` uses the normal approximation for continuous metrics. Should it use
      the non-central t instead? The gap is measurable at n=12 and negligible by n≈100.
      Currently surfaced by a test rather than fixed.
