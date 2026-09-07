# Resume Bullets — A/B Testing Kit

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Built a simulation-backed experimentation toolkit (Python, numpy/scipy) that generates A/B
  tests with a **known** true effect, so error-rate claims are measured rather than cited;
  quantified that stopping at the first significant look across 20 interim analyses inflates
  Type-I error from 4.97% to **24.70%** (20,000 replications, Wilson CI [0.2410, 0.2530]) and
  overstates the shipped effect by **68%**.

- Implemented O'Brien–Fleming and Pocock **alpha-spending boundaries** by recursive numerical
  integration over the Brownian sub-density (Armitage–McPherson–Rowe) rather than from published
  tables, validating them against an independent 200,000-path Monte Carlo of the underlying
  process; the boundary holds Type-I at **5.15%** across the same 20 looks for a cost of 2.4
  points of power, returning **26% of the sample** (3,841 → 2,842 users per arm).

- Implemented **CUPED** variance reduction and verified it delivers its theoretical `1 - ρ²` to
  four decimal places across a correlation sweep, more than doubling power at fixed sample size
  (0.350 → 0.754 at ρ = 0.8, equivalent to **2.8× the traffic**) with Type-I error unchanged at
  0.0495; surfaced that thresholding a continuous covariate into a binary one attenuates ρ from
  0.80 to 0.52 and cuts the benefit from 64% to 27%.

- Diagnosed and corrected a **false framework comparison** in the Bayesian-vs-frequentist
  literature by measuring both rules on identical seeded data: a conventional `P(B>A) > 0.95`
  threshold is a *one-sided* 5% rule and ships twice as readily as a two-sided α = 0.05 test;
  matching thresholds at 0.975 collapses the disagreement rate from 1-in-13 to **2-in-10,000**
  (mean |posterior − frequentist| = 4e-5), and demonstrated that a Bayesian stopping rule
  inflates under continuous monitoring just as the frequentist one does.

- Shipped as a documented CLI + notebook with **100% statement coverage** (738 statements, 200+
  tests) and green CI; every figure in the README carries the seeded command that regenerates
  it from a clean clone.

## Which roles this supports

- [x] Data Scientist / ML
- [ ] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

_Only list what you actually used and could be questioned on._

A/B testing · experiment design · statistical power & MDE · sample-size calculation ·
Type-I / Type-II error · sequential testing · group-sequential design · alpha spending
(O'Brien–Fleming, Pocock, Lan–DeMets) · Brownian motion / numerical integration ·
CUPED variance reduction · Bayesian inference · conjugate priors · posterior probability ·
expected loss · Monte Carlo simulation · Wilson score intervals · winner's curse ·
numpy · scipy · pytest · ruff · uv · GitHub Actions

---

### Bad vs good

❌ "Built a machine learning model to predict customer churn using Python."
✅ "Built a churn classifier on 240k accounts (LightGBM, 1:40 class imbalance) with isotonic calibration and cost-sensitive thresholding, lifting precision@10% from 0.31 to 0.58 over the business's existing rules baseline."

The second one is answerable in an interview. The first invites the question you can't answer.
