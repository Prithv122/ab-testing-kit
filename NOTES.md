# Build Notes — A/B Testing Kit

Working notes: what broke, what was tried, why X over Y.
Not for recruiters — for me, six months from now, in an interview.

Keep it rough. Rough is the point.

---

## Log

### 2026-09-07 — session 2c: bayes.py (posterior decision rules)

- **Built:** `bayes.py` — Beta-Binomial and normal posteriors, `P(B>A)`, expected loss,
  credible intervals, plus `agreement_study` and `bayes_peeking_study`. `bayes` and
  `agreement` CLI subcommands. 32 tests.

- **The framing I settled on:** do not argue about Bayes vs frequentist, *measure* three
  things. (1) How often do the two disagree on identical data? (2) What is the frequentist
  error rate of a Bayesian stopping rule? (3) Does going Bayesian fix peeking? All three are
  answerable with the substrate already in the repo, which is the payoff for having built
  `simulation.py` the way I did in session 1.

- **Caught myself writing a false claim, again, and this one was in a docstring.** I wrote
  that a 0.95 posterior threshold "is deliberately the mirror of a 5% two-sided alpha, so the
  comparison is like-for-like". It is not. `P(B>A) > 0.95` is a **one-sided** 5% rule. A
  two-sided alpha of 0.05 only ships on a positive result at 2.5%. So the industry-standard
  posterior threshold is **twice as permissive in the ship direction** as the
  industry-standard p-value threshold, and every "Bayesian methods ship more winners"
  comparison built on that pairing is measuring the thresholds, not the frameworks.
  **How I caught it:** the agreement study came back with disagreements running *entirely*
  one way (`bayes_only=174, frequentist_only=0`). A genuine philosophical difference would
  scatter in both directions. One-directional disagreement is the signature of a threshold
  mismatch, and it is worth remembering as a diagnostic.

- **The result that came out of fixing it is the best thing in the module.** At a matched
  threshold of 0.975, with a genuinely flat prior (continuous metric), the two frameworks
  agree on **2,000 out of 2,000** replications under the null *and* under a real effect. Not
  "rarely disagree" — zero. On a rate metric the Beta(1,1) prior shrinks the estimate a hair
  and the residual is about 1 in 2,000, which is a real Bayesian effect and a measurably tiny
  one. The mean `|P(B>A) - (1 - p/2)|` is ~4e-5. They are the same arithmetic.

- **And the one that matters most in practice:** a Bayesian stopping rule does **not** control
  the rate at which it ships losers under continuous monitoring. Same substrate, same analysis
  points, same stop-at-first-green shape as section 5.2, and the curve inflates in the same
  way the frequentist one does. The claim "posteriors can be monitored continuously" is true
  about the *posterior* and false about the *stopping rule* built on it, and the two get
  conflated constantly. Worth being precise: nothing about Bayesian machinery promises a
  frequentist error rate, so it is not a failure of Bayes — it is people expecting a guarantee
  that was never offered.

- **Expected loss is the part that survives.** It is the one genuinely better thing here: a
  threshold in units of the metric ("ship if the expected cost of being wrong is under 0.1pp")
  is a business statement, and a p-value cannot be one. It also measurably curbs the peeking
  inflation, though it does not remove it. That is the honest recommendation this module ends
  on: use Bayesian machinery for the decision framing, and alpha spending for the error rate.

- **Broke:** `test_a_loss_budget_can_veto_a_confident_result` hard-coded `max_expected_loss=1e-12`
  and failed because the experiment I picked (effect 0.5 at n=4,000) had an expected loss of
  **1.9e-131**. Expected loss shrinks super-exponentially in the z-statistic, so any fixed
  threshold either never bites or always does depending on the effect size.
  **Fixed by:** setting the budget relative to the loss that experiment actually carries.
  **Learned:** worth remembering when advising anyone to set a loss threshold — the quantity
  spans 100+ orders of magnitude over an ordinary range of effect sizes, so the threshold has
  to be reasoned about in metric units against a real MDE, never picked as a round number.

- **Deliberately not done:** exact Beta-Binomial `P(B>A)` by the summation identity. The
  normal approximation to the posterior is accurate to ~1e-4 here and roughly 400x cheaper,
  which is what keeps a 20,000-replication monitoring study to minutes. Tested against direct
  sampling from the posterior rather than assumed.

---

### 2026-09-07 — session 2a: sequential.py (O'Brien–Fleming alpha spending)

- **Built:** `sequential.py` — two spending functions (O'Brien–Fleming, Pocock), the boundary
  solver, `run_sequential` / `sequential_study` / `sequential_error_curve`, plus `sequential`
  and `boundary` CLI subcommands. 44 new tests, still 100% statement coverage.

- **The one real decision: how to compute the boundary.** There is no closed form past the
  first look, because the looks are *correlated* — look 3 contains look 2's data. So the
  critical values come from the standard recursive numerical integration (Armitage–McPherson–
  Rowe): under the null the cumulative score B(t) is Brownian motion, the continuation region
  after each look carries a sub-density, and the next look convolves it forward with a
  Gaussian increment. Each critical value is then a root-find against the spending schedule.
  The alternative was quoting a published boundary table, which would have been the exact
  thing this repo exists not to do.

- **The trick that made it fast enough to root-find.** The naive form of the survival
  probability is a double integral (over the new continuation region, and over the old
  sub-density), which is O(n²) per trial boundary. But the inner integral is a Gaussian CDF
  difference in closed form, so it collapses to a *single* O(n) pass over the previous grid.
  The O(n²) convolution then only has to run once per look, after the boundary is known,
  rather than once per brentq iteration.

- **Broke:** the first look of a 20-look O'Brien–Fleming design budgets about **2e-18** of
  alpha. No quadrature scheme resolves that — the solver was being asked to fit its own
  rounding error, and `brentq` returned junk.
  **Fixed by:** a floor (`_MIN_SPEND = 1e-9`). Below it the look gets an `inf` boundary and
  simply cannot reject. Considered and rejected: clamping to a large finite z (arbitrary), or
  loosening the target (silently spends alpha the schedule did not authorise). The `inf`
  version is conservative by construction, and because the recursion targets *cumulative*
  spend the unspent budget rolls into the next look automatically — the total still lands on
  0.05 to 1e-9. The CLI prints those looks as `never`, which is honest and reads well.

- **Validated it two independent ways, which is the part I would talk about.** I could not
  remember a published boundary table with any confidence, and half-remembering one would
  have been worse than useless. So:
  1. **Against the process itself.** Simulate standardised Brownian motion at the information
     times, count boundary crossings. Shares no code with the recursion. 200k paths, six
     configurations — every one inside 2 SE of 0.05.
  2. **Against a value I could check independently.** The Pocock boundary is *supposed* to be
     near-constant, and the classical constants are things I could sanity-check by shape:
     the solver produces 2.4380/2.4268/2.4102/2.3966/2.3860 for K=5 (published constant
     2.413), 2.2794…2.2959 for K=3 (2.289), ~2.55 for K=10 (2.555). Landing on those without
     being told them is strong evidence the recursion is right.
  Plus the cheap ones: K=1 returns 1.959964 exactly (a one-look sequential design *is* a
  fixed-horizon test), and refining the quadrature from 401 to 1601 nodes moves nothing past
  the 5th decimal.

- **Adding that column immediately caught a bad comparison in my own README.** Session 1
  reported a "6.8x magnification" of the estimate under the null: 0.0368 for 20 looks against
  0.0054 for one look. Those two numbers come from **different populations**. The 0.0368 is
  the mean |estimate| among replications that *stopped and shipped*; the 0.0054 is the mean
  |estimate| across *all* replications, shipped or not — which is just the noise level of a
  3,841-per-arm experiment (se * sqrt(2/pi) = 0.00685 * 0.798 = 0.0055, so it checks out
  analytically). Ratioing one against the other silently attributes the entire
  selection-on-significance effect to peeking.
  **The like-for-like number, both conditioned on rejection at 20,000 reps: 0.0373 at twenty
  looks against 0.0160 at one — 2.3x.** The remaining 3x is what conditioning on significance
  costs at *any* schedule, and a single fixed-horizon look already pays it.
  **Learned:** the mistake was possible because the two figures were produced by different
  ad-hoc scripts in session 1, and nothing forced them onto the same footing. Now that
  `mean_abs_estimate` is a field on `SimulationSummary`, every table prints the conditioned
  version and the comparison cannot drift again. Rewrote README 5.3 to give both numbers with
  their denominators stated. This is the kind of error I would rather find in my own repo than
  have found for me in an interview.

- **Built `cuped.py` in the same session.** Straightforward next to the boundary solver — the
  method is four lines of arithmetic — so the work was all in checking it rather than writing
  it. Measured reduction matches rho^2 to four decimal places across the sweep, Type-I error
  is untouched (0.0495 against a nominal 0.05), and the adjusted estimate is unbiased over all
  replications even though theta is fitted on the same data it adjusts.

- **The interesting CUPED finding is the binary attenuation**, and it closes session 1's first
  open question. Requesting `covariate_corr=0.8` on a 10%-rate binary metric yields a realised
  correlation of about 0.51, because the generator correlates latent normals and *then*
  thresholds them. So the reduction is ~26%, not the ~64% the requested number suggests.
  CUPED is not underperforming — it delivers exactly 1 - rho^2 against the correlation the
  data actually has. I decided **not** to solve `covariate_corr` numerically so the realised
  value matches the request. Reporting the realised correlation next to the requested one is
  more honest and teaches the reader something true about binary covariates; hiding the gap
  behind a solver would make the API tidier and the lesson invisible.

- **`cuped_at_each_look` exists to make a bug hard to write.** `cuped_adjust` fits theta on
  whatever experiment it is handed, so adjusting first and truncating after would let look 1
  use a theta fitted on data that has not arrived. Nothing in this session composes CUPED with
  sequential testing, but session 3 might, and by then I will not remember. There is a test
  asserting the two orders give different answers.

- **Added `mean_abs_estimate` to `SimulationSummary`**, which session 1's notes said to do.
  The signed bias cancels under a two-sided null, so the magnitude column is the only one
  that shows the winner's curse there. Both are now printed by the `peeking` *and*
  `sequential` tables, which share a printer so the two are directly comparable on the page.

- **Kept `TestResult.significant` as the naive per-look verdict** rather than overwriting it
  with the sequential decision. `SequentialOutcome.rejected` is the sequential call. The two
  disagreeing constantly *is* the correction working, and there is a test that pins exactly
  that. Documented loudly in `SequentialOutcome.final` because it is an obvious footgun.

- **Suite time went from ~60s to ~105s** and that is the real cost of this session. Same root
  cause session 1 recorded: scalar scipy calls, one per look per replication, and a
  sequential run under the null almost never stops early so it pays for *every* look. Cut
  replications in the new tests until each assertion still sat at ≥2.5 SE and stopped there.
  With `cuped.py` and `bayes.py` still to come this is now the binding constraint, and the
  vectorisation logged in README §7 has stopped being theoretical.

---

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

- **Nearly shipped a misleading result.** The headline null run reported `est. bias` of
  `+0.0002` at 20 looks, which reads as "no winner's curse" and directly contradicts what
  `peeking.py`'s own docstring claims. Was about to write that up as a null finding.
  **What was actually happening:** the test is two-sided, so under a true null the early
  stops split symmetrically — measured 52.2% positive — and the signed mean cancels. The
  inflation is entirely in the **magnitude**, not the direction.
  **Superseded — the ratio written here originally was wrong.** I quoted "0.0368 among
  stopped runs against 0.0054 for a single look, a 6.8x magnification", which compares two
  different populations: the first conditions on stopping, the second does not. The
  like-for-like figure, both conditioned on rejection at 20,000 replications, is **0.0373
  against 0.0160 — 2.3x**. See the session 2 entry above for the diagnosis. Do not quote
  6.8x anywhere; it is not a number this repository supports.
  **Learned:** `SimulationSummary.estimate_bias` being a *signed* mean is the right default
  for the H1 case (where direction is the whole point) but is actively misleading under H0.
  Documented both in README §5.3 rather than quietly reporting only the H1 number. If I add
  one metric in session 2 it should be mean |estimate|, so the null case does not need a
  paragraph of explanation to be read correctly.

- **The peeking result is more interesting than "peeking is bad."** Under H1, 20 looks
  *raised* power 80.1% → 88.4% and cut data use by 56% (3,841 → 1,707 per arm). That is why
  people do it. The cost is 5x Type-I inflation under the null and a 68% overstatement of
  the true effect. Framing it as a tradeoff rather than a mistake is both more honest and a
  better answer to "so why does anyone peek?" — which is exactly what an interviewer asks
  next. It is also precisely the gap group-sequential alpha spending is designed to close,
  so §5.2/§5.3 set up session 2 rather than just scolding.

- **Open at end of session:** `sequential.py`, `cuped.py`, `bayes.py`, the notebook.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| **mSPRT** for the sequential test | Locked to group-sequential alpha spending instead. mSPRT is more conceptually and implementationally heavy than this toolkit's teaching objective needs — it requires choosing a mixing distribution, and the resulting boundary is hard to inspect. With O'Brien–Fleming-style alpha spending a reader can *see* the significance boundary tighten as looks accumulate, and can check the spent alpha sums to 0.05. Explainability is the deliverable here. |
| **Arguing** Bayesian vs frequentist rather than measuring the difference | Every blog post already does this and none of them run the comparison. The repo has a substrate that makes it measurable on identical data with a known truth, so not measuring it would waste the one advantage this project has. |
| Exact Beta-Binomial P(B>A) via the summation identity | ~400x slower for ~1e-4 of accuracy at these sample sizes, which would have put the monitoring study out of reach. The normal approximation is checked against direct posterior sampling instead of assumed. |
| Comparing a 0.95 posterior threshold against a two-sided alpha of 0.05 and calling it like-for-like | It is not like-for-like — 0.95 is a one-sided 5% rule and ships twice as readily. Both thresholds are now reported, and the matched one (0.975) is what demonstrates the frameworks agree. |
| **Quoting a published boundary table** instead of solving for it | The entire premise of the repo is that claims are checked here, not cited. A table I cannot re-derive is a citation. Solving it also means the boundary adapts to whatever schedule `peek_schedule` produces, including the uneven ones that come out of `min_look_size` dropping early looks. |
| Clamping an unresolvable alpha increment to a large finite z | Arbitrary — the number would be picked to look reasonable rather than derived. `inf` says the true thing: this look cannot reject. |
| Overwriting `TestResult.significant` with the sequential verdict | Would destroy the like-for-like comparison between the naive and corrected rules on identical data, which is the whole demonstration. Kept both, documented the distinction. |
| Only implementing O'Brien–Fleming | Pocock costs one function and turns "sequential testing controls error" into a measurable tradeoff between two shapes — early stopping against final-look power. The contrast is the finding. |
| mSPRT / always-valid p-values | Still rejected, same reason as session 1: explainability is the deliverable. Now with a concrete demonstration of what explainable buys — `ab-testing-kit boundary` prints the bar tightening look by look and the alpha summing to 0.050000. |
| Real public dataset instead of simulation | No public dataset carries both the pre-experiment covariates CUPED needs **and** a known true effect. Without known ground truth, "measured Type-I error" is not measurable — there is nothing to be wrong against. Simulation is not a shortcut here, it is the only way the central claim can be checked. Stated plainly in the README next to every number. |
| One-sided tests | Would need a matching one-sided path in `fixed_horizon_test`. Shipping a `two_sided=False` parameter that no simulation validates would directly undercut the point of the repo. Two-sided everywhere, documented as a limitation. |
| Unpooled SE for the binary p-value | Pooling assumes the null, which is correct for a test statistic. Kept pooled for the p-value and unpooled for the CI, which means the two can disagree at the margin. That is correct behaviour, not a bug — noted in the docstring so a future me does not "fix" it. |
| Clamping under-sized looks up to the minimum | Would silently produce duplicate analysis points (two looks at the same n), quietly changing the schedule. `peek_schedule` drops them instead. |
| Averaging estimates over all replications | Would hide the winner's curse completely. `SimulationSummary.mean_estimate` averages over **rejections only** — the subset an experimenter would actually act on. |
| Keeping the scaffold's `.env.example` | This project reads no environment variables. A template `.env.example` naming services the project never touches is worse than no file (the lesson from A6's ship gate). |

## Open questions

- [ ] The Bayesian peeking result uses a flat/weak prior. A genuinely informative prior
      (a real platform has hundreds of past experiments saying most effects are near zero)
      should shrink early estimates hard and blunt the inflation. That is the strongest
      version of the "Bayesian methods handle peeking better" argument and I have not tested
      it — `bayes_peeking_study` takes `prior_a`/`prior_b`, so it is a sweep away.
- [ ] Expected loss is computed against a point null of zero difference. Real decisions have a
      cost of switching, so the relevant comparison is usually against a threshold of
      practical significance rather than against zero. Would change the numbers and probably
      the conclusion about what the loss rule is worth.
- [ ] Information fraction is taken as `n_k / n_max`, which assumes the outcome variance is
      constant across looks. Under the null it is. Under a real alternative the treatment arm
      variance differs slightly, so the realised information fraction drifts from the planned
      one. Standard practice plans on the design-time value and the measured Type-I error says
      it does no harm here — but "it does no harm at a 10% base rate with a 2pp lift" is not
      the same as "it never matters", and I have not tested a case where it would.
- [ ] The confidence interval reported at a sequential stop is the naive one. It undercovers,
      for exactly the reason the point estimate is biased. Stage-wise-ordering intervals are
      the fix and are not implemented; right now the honest reading is that `SequentialOutcome`
      gives a trustworthy *decision* and an optimistic *estimate*. Should probably say that in
      the README next to the bias number rather than only here.
- [ ] `_MIN_SPEND = 1e-9` is set from a rough error estimate for 1001-node Simpson, not from a
      measured one. It is conservative in the right direction, but I have not actually measured
      the quadrature error to confirm the floor sits where I think it does.
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
