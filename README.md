# Behavioral Abuse Detection for AI Platforms

Detecting abusive usage of an LLM API platform from behavioral signals: spam
bots, account-farming rings, prompt sprayers (jailbreak cycling), and token
theft (account takeover). Built around a high-fidelity **simulation** of
100k-scale user populations, because real abuse logs aren't public — and
because designing the attacker is half of understanding the defense.

The framing question is not "which model is best" but: **how do you reason
about abuse at scale, under class imbalance and adversarial pressure, when a
false positive suspends a legitimate user?** Metrics are PR-AUC and
precision@k, never accuracy.

## Pipeline

```
sim/        event-log simulator (behavioral archetypes -> Parquet)
features/   per-account behavioral features (Polars)
models/     XGBoost baseline (noisy labels in, clean labels for eval)
data/       generated runs (gitignored)
```

Quickstart:

```bash
pip install -r requirements.txt
python -m sim.generate --users 20000 --days 28 --out data/run1
python -m features.build --data data/run1
python -m models.train_xgb --data data/run1
```

100k users / 28 days (~30M events) runs locally in minutes; everything fits
in 8GB VRAM.

## Simulation design (Phase 1)

Five archetypes, each parameterized by distributions, not fixed rules:

| archetype | share | core signals |
|---|---|---|
| normal | 92% | diurnal sessions, log-normal gaps, diverse prompt topics, stable home/work/mobile networks |
| spam_bot | 3% | jittered fixed-period timing, low prompt entropy (slot-filled templates), shared datacenter IPs + device fps per farm |
| account_farmer | 2% | rings created in bursts, shared devices/payment hashes, sequential usernames, dormant-then-activated (30% never activate) |
| prompt_sprayer | 2% | bursts cycling 8 jailbreak intents with paraphrase variation, rotating datacenter/residential-proxy IPs, high refusal rate |
| token_thief | 1% | normal account until takeover, then new device + foreign datacenter IP + extraction prompts + velocity spike |

Hard negatives baked into the normal population: **power users** (fast,
repetitive, near-bot-like duplication), **campus NAT** (hundreds of users
behind one university IP — AS27 is UMD), corporate NAT, and VPN users on
datacenter ASNs.

Training labels (`label_noisy`) simulate enforcement reality: 3% of abusers
are unlabeled (missed enforcement), 0.3% of normals are mislabeled abusive
(bad reports). Evaluation always uses clean ground truth.

Output: `events.parquet` (per-request log: user, ts, ip/asn/country, device
fingerprint, UA, endpoint, prompt cluster + text, tokens, success) and
`users.parquet` (ground truth + signup metadata, ring ids, takeover times).

## Baseline results (Phase 2)

~35 account-local features over the event log (timing entropy, inter-event CV,
burstiness, prompt-cluster entropy, duplicate-prompt fraction, infra diversity,
datacenter fraction, dormancy, hourly peak velocity...). XGBoost, GPU,
class-balanced weights, trained on noisy labels.

**Finding: the v1 simulation is too easy.** At 20k users the baseline
saturates — PR-AUC ≥ 0.999 on every class, precision@500 = 0.96 (= max
achievable at 8% abuse base rate). Top features are exactly the designed
signals (cluster entropy, timing entropy, token stats), confirming the
features work — and confirming that naive attackers are trivially separable
in tabular space. Nothing interesting can be claimed until the attackers
get smarter.

## Roadmap

- [x] Phase 1 — simulator (archetypes, shared infra, label noise, hard negatives)
- [x] Phase 2 — per-account features + XGBoost baseline
- [ ] **Phase 2.5 — harden the simulation** (next): evasive variants — spam
      bots that sample human-like session timing and rotate templates, slow
      sprayers hiding in normal traffic, thieves that mimic the victim's
      usage style and ramp gradually. Target: drive the tabular baseline's
      PR-AUC low enough on coordinated/evasive classes that later models
      have headroom to demonstrate value.
- [ ] Phase 3 — sequence model (GRU/small transformer): supervised + a
      self-supervised next-event perplexity score for takeover detection
      (perplexity spike at the takeover point).
- [ ] Phase 4 — heterogeneous graph (users–IPs–devices–payment–prompt
      clusters) with PyTorch Geometric (GraphSAGE/R-GCN + NeighborLoader);
      headline comparison vs XGBoost on *farming/coordination* classes.
      Plus unsupervised pass: connected components / dense-cluster flags.
      Prompt-spray detection via MiniLM embeddings + FAISS similarity
      clusters across accounts.
- [ ] Phase 5 — decision layer: score fusion -> tiered actions
      (monitor / rate-limit / challenge / suspend), per-tier thresholds from
      explicit FP/FN cost analysis, adversarial round (adaptive attackers,
      measure degradation, identify which signals are robust), real-time vs
      batch feature split.

## Design notes

- Ground-truth prompt cluster IDs are namespaced (`spam_*`, `jb_*`) for
  evaluation only — models must never key on the namespace; embedding-based
  clustering has to recover the structure on its own.
- Coordination signals (shared IP/device/payment across accounts) are
  deliberately **excluded** from the tabular features: that's the GNN's job,
  and the gap between the two on farming rings is the point of the project.
- Jailbreak payloads are abstract placeholders (`[RESTRICTED-TOPIC-n]`) —
  the simulation models the *behavioral shape* of spraying, not actual
  harmful content.
