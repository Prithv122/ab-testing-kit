# Interview Prep — A/B Testing Kit

**Five questions, five answers.** An unanswered question means this project is not shipped.

If you can't answer one, you don't understand that part of your own project yet — go back and understand it. This file is the difference between a portfolio that survives a technical screen and one that collapses in it.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ There's one module, `simulation.py`, that generates experiments from a spec carrying a
**known** true effect, and owns the shared result types. Everything else is a question asked
against that same substrate.

Two structural choices in it do all the work. First, outcomes are stored **in arrival order**,
so the first *k* rows of an arm are themselves a valid random sample — which is what makes
`truncate(k)` an honest model of "the data as it looked on day 3", and therefore what makes
peeking simulable at all. If I'd regenerated a smaller experiment per look, the looks would be
independent, and the whole effect I'm measuring comes from the fact that they *aren't* —
look 3 contains look 2's data. Second, each replication draws from its own spawned seed, so
replication 47 is identical whether the study runs 100 times or 100,000, and I can pull a
surprising one out and inspect it alone.

On top of that: `design.py` does power and sample size; `peeking.py` measures what continuous
monitoring costs; `sequential.py` fixes it with alpha spending; `cuped.py` is the odd one out
because it changes the *metric* rather than the decision rule, and it returns an ordinary
`Experiment` so everything downstream composes with it; `bayes.py` runs posterior decision
rules on the same experiments. `cli.py` is where results actually come from — every number in
the README carries the command that regenerates it.

The payoff is that comparisons are apples-to-apples by construction. `sequential.py` reuses
`peeking.py`'s own `peek_schedule`, so the corrected rule analyses the data at *exactly* the
same sample sizes the naive one does. The only difference between those two rows in the README
is the number each look gets compared against.

### Q2. Why did you choose alpha spending over mSPRT / always-valid p-values?

_A:_ Explainability was the deliverable. mSPRT needs you to pick a mixing distribution, and the
boundary that falls out is hard to inspect — you largely have to trust it. With O'Brien–Fleming
alpha spending I can print the thing: `ab-testing-kit boundary --n 3841 --looks 5` shows the
critical value dropping 4.38 → 3.10 → 2.55 → 2.25 → 2.06, and the alpha column summing to
exactly 0.050000. A reader can check that. For a repository whose whole premise is "claims you
can verify beat claims you can cite", that mattered more than mSPRT being the more modern tool.

The related decision: I **solve** the boundary rather than quoting a published table. There's no
closed form past the first look because the looks are correlated, so it's the standard recursive
integration over the Brownian sub-density — the Armitage–McPherson–Rowe scheme. A table I can't
re-derive is just a citation, and solving it also means the boundary adapts to whatever schedule
comes out of `peek_schedule`, including the uneven ones you get when small early looks are
dropped.

Elsewhere: two-sided tests only, because a one-sided design would need a matching one-sided
analysis path and shipping a parameter no simulation validates would undercut the point. Pooled
SE for the binary p-value, unpooled for the interval — pooling assumes the null, which is right
for a test statistic and wrong for an estimate, so the two can disagree at the margin and that's
correct rather than a bug.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ Computationally, the bottleneck is scalar scipy calls — `norm.sf`, `t.ppf` — one set per
look per replication. A 20-look study at 20,000 replications is up to 400,000 of them, and the
sequential case is the worst because under the null it almost never stops early, so it pays for
every look. Vectorising across replications and precomputing critical values instead of calling
`ppf` in a loop is a 10–50× win with no change to the statistics. I deliberately didn't do it:
for a repo whose purpose is that a reader can verify the claims, obviously-correct beat fast.
It's the first thing in README §7, and it's stopped being theoretical: once sequential and
Bayesian monitoring landed the suite went to 209 seconds under coverage, against the 60 I'd set
myself as a budget. I'd rather say that plainly than quietly relax the budget — it's over, and
vectorising the analysis loop is the fix.

Memory becomes the wall before CPU does at real scale — each replication materialises four full
arrays, so at 10⁶ users per arm that's ~32 MB per replication. Streaming sufficient statistics
would fix it at the cost of `truncate`'s simplicity.

But the honest answer is that the **statistical** limits matter more than the computational
ones. Everything here assumes i.i.d. units, correctly specified variance, and no interference. A
real platform hits sample-ratio mismatch, network effects between users, delayed conversions
that make early looks biased rather than merely noisy, and revenue metrics heavy-tailed enough
that the CLT isn't safe at these sample sizes. Each breaks a different assumption in this code
and none of them shows up in a simulation that generates clean data. Extending the generator to
*produce* those pathologies — and measuring which methods survive them — is the version of this
I'd actually deploy.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Three layers, and they check different things.

**Against theory, where a closed form exists.** The sample-size formula says 3,841 per arm for
80% power; simulating that design 20,000 times against a known 2pp effect gives 0.8007, with the
analytic 0.8000 inside the Monte Carlo interval. A one-look Type-I error comes out 0.0497
against a nominal 0.05. Every rate I report carries a Wilson interval **on the rate itself**,
because "measured error 0.0512" from 10,000 runs carries ±0.004 and without that the claim is
unfalsifiable.

**Against an independent implementation, where no closed form exists.** The alpha-spending
boundary is the risky code — it can be subtly wrong and still look plausible. So it's checked
against a direct Monte Carlo of the Brownian motion it claims to integrate: 200,000 paths across
six configurations, every crossing rate within 2 SE of 0.05, sharing no code with the recursion.
And separately, the Pocock boundary is *supposed* to come out near-constant, and the solver
returns 2.438/2.427/2.410/2.397/2.386 for five looks — bracketing the classical constant of
2.413 without having been told it.

**The headline results themselves.** Naive peeking takes Type-I from 4.97% to 24.70% at twenty
looks. Alpha spending on identical data holds 4.97% → 5.15%, and every row's interval contains
0.05. It costs 2.4 points of power and returns 26% of the sample. CUPED reproduces `1 - ρ²` to
four decimal places while leaving Type-I at 0.0495.

Tests check correctness and the CLI produces results — deliberately separate jobs. Conflating
them gave me a 117-second suite *and* imprecise numbers, so headline figures come from long CLI
runs at 20,000 replications and the suite stays fast enough to actually run.

### Q5. Your Bayesian null rule ships at ~4.85%, and your two-sided frequentist test rejects at ~4.97%. Those look identical — so where's the "twice as permissive" claim coming from?

_A:_ They're not comparable numbers, and that's exactly the trap.

The frequentist 4.97% counts rejections in **both** directions — it includes the runs where the
estimate came out negative, which nobody ships. Filter to positive rejections and the
frequentist *ship* rate is about 2.5%. The Bayesian 4.85% is all ships by construction, because
`P(B > A) > 0.95` is inherently one-sided. So the like-for-like comparison is 4.85% against
2.5%, and the conventional 0.95 posterior threshold is a one-sided 5% rule — roughly twice as
permissive in the ship direction as the two-sided 5% test it usually gets compared against.

What convinced me it was a threshold artefact and not a real difference between frameworks: the
disagreements ran **entirely one way** — 1,544 bayes-only and 0 frequentist-only out of 20,000.
A genuine philosophical difference would scatter in both directions. One-directional
disagreement is the signature of a threshold mismatch. And the confirmation is that matching
them at 0.975 collapses the disagreement rate from about 1 in 13 to about 2 in 10,000, with the
residual traceable to two named approximations — the Beta(1,1) prior shrinking the estimate a
hair on a rate metric, and Welch's t versus a normal posterior differing slightly in the tail.
Mean `|P(B>A) - (1 - p/2)|` is 4e-5. They're computing the same number.

I'll flag that I originally wrote the wrong thing in a docstring — I claimed 0.95 was "the
mirror of a 5% two-sided alpha, so the comparison is like-for-like". The one-directional
disagreement is what caught it.

---

## 30-second pitch

_Problem → what you built → measured outcome. Say it out loud until it's fluent._

Teams watch experiment dashboards daily and ship the variant the morning it goes green,
believing they're accepting a 5% chance of being wrong. I built a toolkit that generates
experiments with a **known** true effect, so every claim about an error rate can actually be
checked rather than cited.

Measured: that habit takes the false-positive rate from 5% to **24.7%** at twenty looks, and
inflates the effect you report by 68%. Group-sequential alpha spending — with boundaries solved
in the repo and validated against an independent Brownian simulation, not copied from a table —
holds the rate at **5.15%** on identical data, costing 2.4 points of power and giving back 26%
of the sample. CUPED reproduces its promised `1 - ρ²` variance reduction to four decimals, worth
2.8× the traffic at ρ = 0.8. And going Bayesian doesn't fix peeking: a posterior stopping rule
inflates the same way, because it's the same arithmetic.

Everything reproduces from a clean clone with the command printed next to each number.
