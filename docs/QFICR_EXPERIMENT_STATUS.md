# QFi-CR Experiment Status

## Current Mainline

QFi-CR + ratio-cap 0.20.

## K64 Anchor Quick Check

- current_anchor Avg: 67.4168
- weak_anchor Avg: 67.4530
- no_anchor Avg: 67.4555
- no_anchor is slightly higher on average, but is almost tied with weak_anchor.
- weak_anchor is more stable.
- current_anchor is best on GQA.
- no_anchor has a small regression on SQA-IMG.
- Do not remove the p_j anchor directly yet. K32/K128 validation is still needed.

## Output Paths

These paths record important local outputs only. Large files are not committed.

- `outputs/qficr_ratio_cap020_anchor_quick_k64/`
- `logs/qficr_ratio_cap020_anchor_quick_k64/`

## Next Steps

- Validate weak_anchor and no_anchor at K32/K128 for GQA/SQA risk.
- Add TextVQA/MME K32/K128 checks later.
- Add direct greedy and SCOPE-style baselines later.
