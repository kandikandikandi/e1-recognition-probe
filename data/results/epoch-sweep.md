# E1 Epoch Sweep

Three surfaces across all unlearn checkpoints. All results use corrected adapter loading.

## Behavioral (structural reasoning subcategory, n=20)

| Epoch | Behavior acc |
|---|---|
| FT | 0.920 |
| e025 | 0.850 |
| e03 | 0.830 |
| e035 | 0.670 |
| e04 | 0.455 |
| e045 | 0.150 |
| e05 | 0.000 |
| e1 | 0.035 |
| e2 | 0.000 |

## Forced-choice recognition (logprob, 4-way, chance = 0.250)

| Epoch | FC acc | vs FT | vs chance |
|---|---|---|---|
| FT | 0.625 | — | +0.375 |
| e025 | 0.537 | -0.088 | +0.287 |
| e03 | 0.537 | -0.088 | +0.287 |
| e035 | 0.537 | -0.088 | +0.287 |
| e04 | 0.512 | -0.113 | +0.262 |
| e045 | 0.487 | -0.138 | +0.237 |
| e05 | 0.475 | -0.150 | +0.225 |
| e1 | 0.325 | -0.300 | +0.075 |
| e2 | 0.450 | -0.175 | +0.200 |

## Representation — concept-vs-concept probe

Polythricidae (label=1) vs Cinerylithidae (label=0). Unlearning targeted Polythricidae only.

| Epoch | CvC best-layer acc | vs FT | note |
|---|---|---|---|
| FT | 1.000 (layer 31) | — | |
| e025 | 1.000 (layer 20) | +0.000 | |
| e03 | 0.989 (layer 16) | -0.011 | |
| e035 | 0.967 (layer 16) | -0.033 | |
| e04 | 0.967 (layer 16) | -0.033 | |
| e045 | 0.967 (layer 16) | -0.033 | |
| e05 | 0.967 (layer 16) | -0.033 | |
| e1 | 0.879 (layer 20) | -0.121 | |
| e2 | 0.901 (layer 20) | -0.099 | |

### CvC per-layer (FT vs e025)

| Layer | FT | e025 | delta |
|---|---|---|---|
| 4 | 0.747 | 0.747 | +0.000 |
| 8 | 0.945 | 0.670 | -0.275 |
| 12 | 0.989 | 0.890 | -0.099 |
| 16 | 0.989 | 0.989 | +0.000 |
| 20 | 0.989 | 1.000 | +0.011 |
| 24 | 0.989 | 0.989 | +0.000 |
| 28 | 0.989 | 0.989 | +0.000 |
| 31 | 1.000 | 1.000 | +0.000 |

## Note on generic probe

The earlier Polythricidae-vs-factual-controls probe saturates at 1.000 at all epochs because it detects prompt-domain type (taxonomy query vs general knowledge), not concept-specific representation. It also achieves 1.000 on Cinerylithidae prompts it was never trained on. The CvC probe above is the correct measure.
