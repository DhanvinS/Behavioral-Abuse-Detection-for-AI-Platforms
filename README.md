# Behavioral Abuse Detection for AI Platforms

Detecting abusive usage of an LLM API platform from behavioral signals: spam
bots, account-farming rings, prompt sprayers (jailbreak cycling), and token
theft (account takeover). Built around a high-fidelity **simulation** of
100k-scale user populations with **adaptive attackers**, because real abuse
logs aren't public — and because designing the attacker is half of
understanding the defense.

The framing question is not "which model is best" but: **how do you reason
about abuse at scale, under class imbalance and adversarial pressure, when a
false positive suspends a legitimate user?** Metrics are PR-AUC and
precision@k, never accuracy.

## Headline results (20k users, 28 days, adaptive attackers)

| detector | signal type | abuse PR-AUC | what it uniquely catches |
|---|---|---|---|
| XGBoost (tabular) | account-local stats | 0.975 | naive volume/timing/infra abuse |
| GRU sequence model | event streams + content clusters | 0.990 | takeover discontinuity (ATO AUC 0.91, localized to ~42h median) |
| Hetero GraphSAGE | shared infrastructure | 0.985 | coordination (farmers: 1.000) |
| ring mining (unsupervised) | identity graph components | — | 394/400 farm accounts, ~100% pure components |
| spray detector (MiniLM + clustering) | prompt semantics | — | 398/400 sprayers, incl. 161/163 "slow" ones invisible to tabular; 0 normal FPs |
| **fused risk + decision layer** | all of the above | **0.999** | flat across attacker evasion levels |

The adversarial result that matters: the tabular baseline degrades with
attacker evasion (PR-AUC 0.995 → 0.952 from naive to maximally evasive),
while the **fused system stays flat (≈0.996–0.998 at every evasion level)**.
Timing and volume are cheap to fake; shared devices/payments, prompt
semantics, and behavioral discontinuity are not.

The decision layer converts risk into tiered actions with explicit costs.
On held-out accounts: suspend tier at 1.000 precision absorbing ~86% of
abuse, 94.7% expected-cost reduction vs. no enforcement — and an honest
negative finding: the rate-limit tier caught zero abusers while adding
friction to 124 legit users, arguing for a 3-tier policy on this attack mix.

## Pipeline

```
sim/                 event-log simulator (archetypes + evasion knob -> Parquet)
features/            per-account behavioral features (Polars, content-blind)
models/train_xgb.py  tabular baseline (GPU XGBoost, noisy labels in / clean eval)
models/train_seq.py  GRU: 5-class + self-supervised next-event LM for ATO
models/train_gnn.py  hetero GraphSAGE over user-IP-device-payment (PyG)
models/mine_rings.py unsupervised union-find ring mining w/ NAT fanout cap
models/spray_detect.py  MiniLM embeddings -> campaign clusters
models/fuse.py       stacked risk + cost-based enforcement tiers
pipeline/run_all.py  one-command pipeline w/ persistent per-stage logs
pipeline/make_report.py  consolidated report: summary.md + figures + metrics
dashboard/app.py     Streamlit review console
```

Quickstart (RTX-class GPU, ~10 min end to end at 20k users):

```bash
pip install -r requirements.txt
python -m pipeline.run_all --users 20000 --days 28 --evasion 1.0 --data data/run2
streamlit run dashboard/app.py -- --data data/run2
```

The runner streams each stage live, tees logs to `reports/<run>/logs/`,
writes a `manifest.json` with timings/exit codes, and finishes by building
`reports/<run>/summary.md` with stored figures (detector comparison,
evasion-robustness curves, per-class PR-AUC, risk distributions, an ATO
localization timeline) plus merged `metrics.json`. Raw data stays in
`data/` (gitignored); everything reviewable lives in `reports/` (tracked).
Stages can also be run individually (`python -m models.train_gnn --data
data/run2`); each writes its own `metrics.json` and `scores.parquet`, and
`python -m pipeline.make_report --data data/run2` rebuilds the report from
whatever exists.

100k users / 28 days (~30M events) also runs locally; everything fits in
8GB VRAM (the GNN is full-batch; the embedder is MiniLM).

## Simulation design (the part most projects skip)

Five archetypes, each parameterized by distributions, not fixed rules:

| archetype | share | core signals |
|---|---|---|
| normal | 92% | diurnal sessions, log-normal gaps, diverse prompt topics, stable home/work/mobile networks |
| spam_bot | 3% | jittered fixed-period timing, low prompt entropy, shared datacenter IPs + device fps per farm |
| account_farmer | 2% | burst-created rings, shared devices/payment hashes, sequential usernames, dormant-then-activated (30% never activate) |
| prompt_sprayer | 2% | bursts cycling 8 jailbreak intents with paraphrase variation, rotating proxies |
| token_thief | 1% | normal account until takeover, then new device + foreign IP + extraction prompts |

Hard negatives baked into the normal population: power users (fast,
repetitive, near-bot duplication), campus NAT (hundreds of users behind one
university IP — AS27 is UMD), corporate NAT, VPN users on datacenter ASNs.
Training labels carry enforcement noise (3% missed abusers, 0.3% false
reports); evaluation always uses clean ground truth.

### The evasion knob (`--evasion`)

Each attacker entity samples an evasion level e ∈ [0, max]. The design rule:
as e → 1, attackers converge to the normal envelope on everything that is
**cheap to fake**, and keep only what is **expensive to give up**:

- evasive spam bots run inside a fully normal session envelope (often a
  "purchased aged account"), inject spam into a fraction of prompts, and
  only route ~25% of days through shared farm infrastructure (proxies cost
  money — that residual sharing is what the GNN finds);
- evasive sprayers go low-and-slow: normal account, normal timing, a random
  ~5% of events swapped for paraphrased jailbreaks. Only content gives them
  away — and a safety refusal is an HTTP 200, invisible in transport logs;
- evasive farmers buy aged accounts, stagger creation over weeks, spread
  across more devices/cards (but never fully — shared identity is the
  business model);
- evasive thieves use same-country residential proxies, mimic the victim's
  prompt style and token sizes, and ramp volume gradually. The new device
  fingerprint and the conditional-behavior break are what remain.

This knob was forced by a failure: the v1 simulation saturated every
detector at PR-AUC 1.0. Three iterations of hardening (overlapping nuisance
distributions, envelope mimicry, and removing a label leak where tabular
features used ground-truth prompt clusters) produced a sim where the
baseline has honest headroom — see "lessons" below.

## What each phase showed

**Tabular baseline (XGBoost).** Catches effectively all naive abuse —
worth stating plainly: most real-world abuse volume is naive, and a
content-blind tabular model on transport logs gets you 0.975 PR-AUC.
Degrades on evasive attackers (0.952) and is near-blind to slow sprayers
(class PR-AUC 0.879).

**Sequence model (GRU, multi-task).** Supervised head + a next-event
language model trained only on normal accounts. Per-event NLL spikes detect
takeovers the attacker can't smooth over: they can fake marginals, not the
victim's conditional behavior. Thief-vs-normal AUC 0.912 from the anomaly
score alone, takeover localized to a median 42h window. Sees content via
embedding-cluster IDs (the content-aware tier of the system).

**GNN (hetero GraphSAGE).** User nodes carry exactly the baseline's
features, so gains are attributable to message passing. Perfect on farmers
(1.000 vs 0.995); weaker on sprayers (0.846) — content attacks aren't
structural, which is precisely why you run both.

**Unsupervised passes.** Union-find over shared devices/payments/low-fanout
IPs recovers 98.5% of farm accounts in near-pure components, with a NAT
fanout cap so the campus NAT (a designed hard negative) isn't flagged. The
spray detector flags embedding clusters that are semantically tight, rare
across the user base, but used intensively — recovering ground-truth intent
clusters at ~100% purity with zero normal false positives.

**Fusion + decisions.** Logistic stack (fit on a held-out half of test, on
noisy labels, to avoid leaking the detectors' overfit train scores), then
precision-targeted thresholds per enforcement tier with explicit FP/FN
costs (false suspension = 100, missed thief = 200, etc.).

## Lessons that generalize (the interview section)

1. **A saturated benchmark proves nothing.** The first sim gave every model
   PR-AUC 1.0. Hardening the attacker until the baseline shows a gradient
   is what makes any later comparison meaningful.
2. **Label leakage hides in convenient features.** Ground-truth prompt
   clusters in the feature set quietly inflated everything; real systems
   only get noisy embedding clusters. (The spray detector then validated
   that intent clusters *are* recoverable from text — but you have to earn
   it, not assume it.)
3. **Signals have costs to fake, and that ordering is the defense.**
   Timing jitter is free; residential proxies are cheap; separate devices
   and payment instruments per account break the economics of farming.
   Detector portfolios should be weighted toward expensive-to-fake signals.
4. **Refusals are not failures.** Modeling the log schema honestly (a
   safety refusal is HTTP 200) moved sprayer detection from "trivial" to
   "content-only" — and changed which detector owns that attack class.
5. **The decision layer is where ML meets reality.** Same scores, different
   thresholds per action tier; a tier that catches nothing but adds
   friction is a policy bug even when the classifier above it is excellent.

## Design notes

- Ground-truth prompt cluster IDs (`spam_*`, `jb_*`) exist for evaluation
  and as a stand-in for an online embedding-clustering service; the tabular
  baseline never sees them.
- Coordination signals are deliberately excluded from tabular features:
  that gap is the GNN's job, and the comparison is the point.
- Jailbreak payloads are abstract placeholders (`[RESTRICTED-TOPIC-n]`) —
  the simulation models the behavioral shape of spraying, not harmful
  content.
- Real-time vs batch: tabular features and sequence scoring are computable
  online per-event; graph and embedding-cluster passes are batch (hourly/
  daily) — the fusion layer is where both cadences meet.

## Possible extensions

- Sweep `--evasion` 0 → 1 and plot per-detector degradation curves.
- Cross-fitted stacking and per-class fusion heads.
- Cold-start: how few events before each detector becomes useful?
- Replace hash-bucket content IDs in the GRU with learned MiniLM cluster
  assignments end to end.
