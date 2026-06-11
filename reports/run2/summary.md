# Run report — `data/run2`
_generated 2026-06-11 23:13 UTC_

## Headline

| detector | abuse PR-AUC |
|---|---|
| fused | 0.9987 |
| xgb_abuse | 0.9752 |
| seq_abuse | 0.9903 |
| gnn_abuse | 0.9860 |

Expected cost: **460.0** vs do-nothing 8,950.0 (**94.9% reduction**)

### Enforcement tiers

| tier | n | abusive | precision |
|---|---|---|---|
| suspend | 210 | 210 | 1.000 |
| challenge | 67 | 31 | 0.463 |
| rate_limit | 133 | 0 | 0.000 |
| monitor | 2590 | 0 | 0.000 |

## XGBoost baseline

| class | PR-AUC |
|---|---|
| account_farmer | 0.9950 |
| normal | 0.9978 |
| prompt_sprayer | 0.8789 |
| spam_bot | 0.9961 |
| token_thief | 1.0000 |

## Sequence model

| class | PR-AUC |
|---|---|
| account_farmer | 0.9999 |
| normal | 0.9998 |
| prompt_sprayer | 1.0000 |
| spam_bot | 0.9906 |
| token_thief | 1.0000 |

ATO anomaly AUC: **0.9108**, median localization 41.77h

## GraphSAGE

| class | PR-AUC |
|---|---|
| account_farmer | 1.0000 |
| normal | 0.9977 |
| prompt_sprayer | 0.8457 |
| spam_bot | 0.9817 |
| token_thief | 0.9992 |

## Ring mining (unsupervised)

- flagged 950 accounts, precision 0.792, recall 0.752 (IP fanout cap 10)

## Spray detection (embeddings)

- cluster ground-truth purity 98.54%, precision 0.724
- recall: prompt_sprayer 99.50%, spam_bot 72.00%, normal 0.00%, prompt_sprayer_slow(e>0.6) 98.77%

## Figures

![detector_comparison.png](figs/detector_comparison.png)
![evasion_robustness.png](figs/evasion_robustness.png)
![per_class_pr_auc.png](figs/per_class_pr_auc.png)
![risk_by_label.png](figs/risk_by_label.png)
![ato_timeline.png](figs/ato_timeline.png)