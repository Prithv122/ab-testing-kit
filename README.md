# A/B Testing Kit

> Experiment design and analysis where every claim about error rates is **measured by simulation in this repository**, not quoted from a textbook.

[![CI](https://github.com/Prithv122/ab-testing-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/Prithv122/ab-testing-kit/actions/workflows/ci.yml)

**Live demo:** not deployed — CLI + notebook, runs locally
**Stack:** Python 3.13 · numpy · scipy · pytest · ruff · uv

> **Status: session 2 of 3.** Power analysis, the peeking demonstration and group-sequential
> alpha spending are complete and validated. CUPED and the Bayesian comparison are the
> remaining modules; §5 grows as each lands.

---

## 1. The problem

A team runs a checkout experiment, watches the dashboard daily, and ships the variant the
morning it first goes green. They believe they are accepting a 5% chance of being wrong,
because that is the number on the tool. They are not.

This toolkit exists to answer the questions a product data scientist gets asked *before* an
experiment starts — how many users do we need, what can we actually detect, when are we
allowed to look — and to prove its answers rather than assert them. Statistical libraries
generally state that a method controls error at some rate. Here, the repository generates
experiments with a **known** true effect, applies the method thousands of times, and reports
the rate that actually came out, with a confidence interval on that rate.

That inversion is the whole design: a claim you can check is worth more than a claim you
can cite.

## 2. The data

| | |
|---|---|
| Source | Simulated in `src/ab_testing_kit/simulation.py`, seeded |
| Size | Generated per run; headline results use 20,000 replications of 3,841 users per arm |
| Licence | n/a — no external data |
| Refresh | Deterministic given `--seed`; every number below carries the command that reproduces it |

**The data is synthetic, and that is a design decision rather than a convenience.** A
measured Type-I error is only meaningful against a *known* truth: you cannot compute how
often a method is wrong without knowing what right was. No public A/B dataset ships a known
true effect, and none carries the pre-experiment covariates CUPED needs alongside one. The
tradeoff is real and worth stating plainly — these results characterise the **methods**
under clean, i.i.d., correctly-specified conditions. They say nothing about how those
methods behave under the messier failure modes of a live experimentation platform:
sample-ratio mismatch, network effects, delayed conversions, or heavy-tailed revenue
metrics. Those are limitations, not oversights, and §7 says what would change.

## 3. Architecture

```mermaid
flowchart LR
    subgraph core["simulation.py — shared substrate"]
        A["ExperimentSpec<br/>known ground truth"] --> B["generate_many<br/>spawned seed per replication"]
        B --> C["Experiment<br/>units in arrival order"]
        C --> D["fixed_horizon_test<br/>pooled z / Welch t"]
        D --> E["TestResult"]
        E --> F["summarize<br/>+ Wilson CI on the rate"]
    end

    C -.->|"truncate(k)"| G["peeking.py<br/>stop at first green"]
    G -->|"peek_schedule<br/>same analysis points"| S["sequential.py<br/>stop at the boundary"]
    P["alpha_spending_boundary<br/>Brownian recursion,<br/>solved per schedule"] --> S
    F --> H["design.py<br/>n, MDE, power"]
    G --> F
    S --> F
    H -->|"validate_power"| F

    F --> I["cli.py<br/>deterministic output"]
    I --> J["README numbers"]
    I --> K["notebook — charts"]

    style core fill:#f6f8fa,stroke:#57606a
```

Every module answers a statistical question against the *same* generated experiments. That
is what makes the comparisons in §5 apples-to-apples: the fixed-horizon, peeking and
sequential rows below differ only in the decision rule applied, never in the underlying data
or seed. `sequential.py` goes one step further and reuses `peeking.py`'s own `peek_schedule`,
so the corrected rule analyses the data at *exactly* the same sample sizes the naive one
does. The only difference left between §5.2 and §5.4 is the number each look is compared
against.

## 4. Key decisions & tradeoffs

| Decision | Chose | Over | Why |
|---|---|---|---|
| Where simulation lives | One shared `simulation.py` owning generation and result types | Each module rolling its own | Makes the central claim checkable. If peeking and fixed-horizon results came from separately-generated data, a difference between them could be a data artefact; sharing the substrate means the decision rule is the only thing that varies. |
| Unit storage | Outcomes stored **in arrival order**, sliced by `truncate(k)` | Regenerating a smaller experiment per look | Consecutive looks must be *correlated*, because in reality they share data. Regenerating would make looks independent and understate the inflation — the bug would have flattered the result. |
| Seeding | `SeedSequence.spawn` per replication | One generator drawn sequentially | Replication *i* is identical whether the study runs 100 or 100,000. A surprising replication can be pulled out and inspected alone. |
| Design formula | Sized for the **pooled** two-proportion z-test | Unpooled / continuity-corrected | It matches the test `fixed_horizon_test` actually applies. Sizing for one test and analysing with another is a quiet route to being under-powered; `validate_power` proves the two stay in step. |
| Test sidedness | Two-sided only | A `two_sided` flag | A one-sided design needs a matching one-sided analysis path. Shipping a parameter no simulation validates would undercut the point of the repo. Documented as a limitation instead. |
| Reporting the rate | Wilson CI on the **rejection rate itself** | Bare point estimate | "Measured error 0.0512" from 10,000 runs carries ±0.004 of Monte Carlo error. Without that interval, a claim that a method controls error is unfalsifiable. |
| Estimate averaging | Over **rejections only** | Over all replications | Averaging over everything hides the winner's curse entirely — the whole second finding in §5. |
| Sequential boundary | Solved here by recursive integration over the Brownian sub-density | Quoting a published boundary table | The premise of the repo is that claims are checked rather than cited, and a table I cannot re-derive is a citation. Solving it also means the boundary adapts to whatever schedule `peek_schedule` produces, including uneven ones. Validated against a direct simulation of the process it integrates — §5.4. |
| Unresolvable alpha budgets | Mark the look `inf`: it cannot reject | Clamp to a large finite z, or loosen the target | The first of twenty O'Brien–Fleming looks budgets ~2e-18, below any quadrature's noise floor. Clamping picks a number to look reasonable; loosening spends alpha the schedule never authorised. `inf` is conservative by construction and the unspent budget rolls forward, so the total still lands on 0.050000. |
| Spending functions | Both O'Brien–Fleming and Pocock | OBF alone | One function shows that sequential testing controls error. Two turn it into a measurable tradeoff — early stopping against final-look power — which is the part worth knowing. |
| Per-look verdicts | Keep the naive `significant` flag beside the sequential decision | Overwrite it with the corrected call | The two disagreeing is the correction doing its job, and keeping both is what makes the like-for-like comparison possible at all. Documented as a footgun and pinned by a test. |
| Where results come from | Long CLI runs | The test suite | Tests check correctness and must stay fast; results need many replications. Conflating them gave a 117s suite *and* imprecise numbers. |

## 5. Results

All numbers below are reproducible from a clean clone. Each carries its command. Simulated
data — see §2 for what that does and does not license.

**Scenario:** a checkout conversion rate of 10%, and we want to detect a 2 percentage-point
absolute lift (a 20% relative improvement), at α = 0.05 two-sided and 80% power.

### 5.1 The sample-size formula delivers what it promises

```bash
uv run ab-testing-kit design --metric binary --baseline 0.10 --mde 0.02
```

Design requires **3,841 users per arm** (7,682 total). Running that design 20,000 times
against a *known* 2pp effect:

```bash
uv run ab-testing-kit validate-power --metric binary --baseline 0.10 --n 3841 --effect 0.02 --sims 20000 --seed 0
```

| | Analytic | Simulated (20,000 reps) | 95% CI on the simulated rate |
|---|---|---|---|
| **Power** at the designed n | 0.8000 | **0.8007** | [0.7952, 0.8062] |
| **Type-I error**, one look, true effect = 0 | 0.0500 | **0.0497** | [0.0468, 0.0528] |

The formula falls inside the Monte Carlo interval in both cases. This is the baseline the
rest of the toolkit is measured against: analysed **once**, at the planned horizon, the test
does exactly what it says.

### 5.2 Checking the dashboard 20 times turns a 5% error rate into 25%

Same null data, same seed, same nominal alpha. The *only* thing that changes across rows is
how many times the experiment is analysed before the horizon, stopping at the first
significant look.

```bash
uv run ab-testing-kit peeking --metric binary --baseline 0.10 --n 3841 --looks 1,2,3,5,10,20 --sims 20000 --seed 0
```

| Looks | Measured Type-I error | 95% CI | Mean n per arm at stop |
|---:|---:|---|---:|
| 1 (no peeking) | 0.0497 | [0.0468, 0.0528] | 3,841 |
| 2 | 0.0837 | [0.0799, 0.0876] | 3,745 |
| 3 | 0.1076 | [0.1034, 0.1120] | 3,672 |
| 5 | 0.1403 | [0.1356, 0.1452] | 3,565 |
| 10 | 0.1915 | [0.1861, 0.1970] | 3,393 |
| 20 | **0.2470** | [0.2410, 0.2530] | 3,195 |

**Twenty looks inflates the false-positive rate roughly 5x, from 4.97% to 24.7%.** A team
running this experiment and calling it the first morning it goes green is wrong about one
time in four while believing they are wrong one time in twenty. The intervals are nowhere
near overlapping, so this is not Monte Carlo noise.

Note that the inflation is already material at **two** looks (8.4%) — this is not a problem
that requires an obsessive experimenter, only a curious one.

### 5.3 A subtlety the signed bias hides

The CLI output for the run above carries one more column than the table reproduces: a mean
estimate bias, which comes out near **zero** under the null. That looks at first like
evidence there is no winner's curse. There isn't one *in the signed mean*, and the reason is
worth stating: the test is two-sided, so under a true null the
replications that stop early split symmetrically between spuriously-positive and
spuriously-negative, and the two cancel in the average. The inflation is in the
**magnitude**, not the direction.

Give the experiment a real effect and the direction stops cancelling:

```bash
uv run ab-testing-kit peeking --metric binary --baseline 0.10 --n 3841 --effect 0.02 --looks 1,20 --sims 20000 --seed 0
```

| Looks | Power | Mean n per arm at stop | Mean reported effect | True effect | Overstatement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8007 | 3,841 | 0.0224 | 0.02 | +12% |
| 20 | **0.8839** | **1,707** | **0.0337** | 0.02 | **+68%** |

This is the honest version of the tradeoff, and it explains why peeking is tempting rather
than merely careless. Twenty looks **raised** power from 80.1% to 88.4% and reached a
decision on **56% less data** — 1,707 users per arm instead of 3,841. Those are real,
valuable things.

The price is paid twice. Under the null, a 5x inflated false-positive rate (§5.2). Under a
true effect, a shipped estimate of **3.4pp against a true 2pp — a 68% overstatement**. The
team ships a real winner and then misses its forecast by two thirds, which is its own kind
of expensive.

Note that even a single look at a fixed horizon overstates by 12%: conditioning on
significance is *always* mildly optimistic. Peeking multiplies an effect that already exists.

For completeness, the magnitude claim in the paragraph above, measured directly under the
null at 4,000 replications: 52.2% of stopping estimates were positive (i.e. symmetric, so
the signed mean cancels to +0.002), while the mean **absolute** estimate was 0.0368 against
0.0054 for a single look at the horizon — a **6.8x** magnification.



## 6. How to run

```bash
git clone https://github.com/Prithv122/ab-testing-kit.git
cd ab-testing-kit
uv sync
uv run pytest
```

No environment variables, no services, no API keys, no dataset download — the data is
generated. Then:

```bash
# How many users do I need?
uv run ab-testing-kit design --metric binary --baseline 0.10 --mde 0.02

# What can I detect with the traffic I have?
uv run ab-testing-kit mde --metric binary --baseline 0.10 --n 5000

# Does that design actually deliver 80% power?
uv run ab-testing-kit validate-power --metric binary --baseline 0.10 --n 3841 --effect 0.02

# What does checking the dashboard every day cost me?
uv run ab-testing-kit peeking --metric binary --baseline 0.10 --n 3841 --looks 1,2,3,5,10,20

# What bar does each look have to clear if I want to keep my 5%?
uv run ab-testing-kit boundary --n 3841 --looks 5

# Does spending the alpha actually hold the error rate?
uv run ab-testing-kit sequential --metric binary --baseline 0.10 --n 3841 --looks 1,2,3,5,10,20
```

`--help` on any subcommand lists its options. Every command is deterministic given `--seed`.

Runtime note: the headline `peeking` and `sequential` runs at `--sims 20000` take several
minutes each, and `sequential` is the slower of the two because under the null it almost
never stops early and so pays for every look. Drop to `--sims 2000` for a fast look — the
pattern is identical, the intervals are wider. `boundary` is instant; it does no simulation.

## 7. What I'd change at 100× scale

- **Vectorise the analysis loop first.** The bottleneck is scalar scipy calls
  (`norm.sf`, `t.ppf`) executed once per look per replication — a 20-look study at 20,000
  replications is up to 400,000 of them. Computing test statistics across all replications
  as arrays, and replacing per-call `ppf` with a precomputed critical value, is a 10–50×
  win with no change to the statistics. Deliberately not done: for a repository whose
  purpose is that a reader can verify the claims, obviously-correct beats fast.
- **Then parallelise across replications.** `SeedSequence.spawn` was chosen partly for this
  — the children are independent, so replications distribute across processes with no
  coordination and no change to results.
- **Memory becomes the wall before CPU does.** Each replication currently materialises four
  full arrays. At 10⁶ users per arm that is ~32 MB per replication. Streaming sufficient
  statistics (running sums and sums of squares) instead of raw arrays would remove the
  ceiling, at the cost of losing `truncate`'s simplicity — the looks would need to be
  computed forward in one pass.
- **The statistical limits matter more than the computational ones.** Everything here
  assumes i.i.d. units, correctly specified variance, and no interference. A real platform
  at this scale hits sample-ratio mismatch, network effects between users, delayed
  conversions that make early looks biased rather than merely noisy, and revenue metrics
  heavy-tailed enough that the CLT is not a safe assumption at the sample sizes involved.
  Each of those breaks a different assumption in this code, and none of them is visible in
  a simulation that generates clean data. Extending the generator to *produce* those
  pathologies — and measuring which methods survive them — is the version of this project
  that would actually be worth deploying.

---

## References

- Standard two-proportion sample-size formulation follows Fleiss, *Statistical Methods for
  Rates and Proportions* (pooled variance under the null, no continuity correction).
- Welch–Satterthwaite degrees of freedom, and the Wilson score interval, are implemented
  from their standard definitions and cross-checked against `scipy.stats` in the test suite.
- The alpha-spending formulation follows Lan and DeMets; the O'Brien–Fleming and Pocock
  spending functions are their standard closed forms. The boundary solver is the recursive
  numerical integration described by Armitage, McPherson and Rowe, implemented here from the
  Brownian-motion characterisation rather than transcribed. No published boundary table was
  consulted — §5.4 says how the result was checked instead.
