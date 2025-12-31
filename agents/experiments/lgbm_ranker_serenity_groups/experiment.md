# Plan

Evaluate LGBMRanker objectives (lambdarank, rank_xendcg) and group definitions
using the medium feature set. Serenity features are already quintiles, so
use raw values (no binning) when forming era x feature groups. Ranker labels
use 5 bins derived from target percentiles.

## Requirements
- LGBMRanker with objectives lambdarank and rank_xendcg.
- Group strategies: era; era x each of the five serenity features.
- Use feature_set=medium for all runs.
- Keep CV/OOF pipeline and metrics unchanged.

## Scope
- In: ranker support in training, configs for objective/group sweeps, experiment outputs.
- Out: changing data version, non-LGBM models, re-binning serenity features.

## Files and entry points
- agents/agent_base_model.py
- agents/experiments/lgbm_ranker_serenity_groups/configs/*.py
- agents/experiments/lgbm_ranker_serenity_groups/experiment.md

## Data model / API changes
- Training config supports ranker grouping strategy and optional grouping feature.

## Action items
[ ] Add LGBMRanker support with group-key sorting and group sizes per fold.
[ ] Run objective sweep (era grouping): lambdarank vs rank_xendcg.
[ ] Run group sweep for each serenity feature; configs exist for both objectives,
    but run the winning objective first.
[ ] Compare results vs baseline; summarize metrics and any stability changes.
[ ] Promote the best grouping to deep LGBM if promising.

## Testing and validation
- Ensure v5.2 full datasets exist.
- Smoke-run one ranker config and verify OOF predictions/results.

## Risks and edge cases
- Group ordering must be contiguous; always sort by group key.
- Small groups in era x feature may reduce ranking signal.
- Rank objectives may not improve BMC/corr vs regression.

## Open questions
- None.

## Results (medium feature set)
Small-experiment view: focus on small_bmc_mean/sharpe and correlation vs baseline;
full bmc metrics are intentionally omitted at this stage.

| model | small_bmc_mean | small_bmc_sharpe | corr_mean | corr_w_baseline_avg |
|---|---|---|---|---|
| small_lgbm_ranker_rank_xendcg_era_x_feature_unwaked_toxophilitic_dilatability | 0.000414 | 0.033556 | 0.013771 | 0.471041 |
| small_lgbm_ranker_rank_xendcg_era_x_feature_kookiest_unrepealable_octuplet | 0.000406 | 0.033248 | 0.013661 | 0.473803 |
| small_lgbm_ranker_rank_xendcg_era_x_feature_scholastic_doglike_sipunculid | 0.000248 | 0.020022 | 0.013827 | 0.478070 |
| small_lgbm_ranker_rank_xendcg_era_x_feature_cycloid_zymotic_galloway | 0.000225 | 0.018013 | 0.013526 | 0.478463 |
| small_lgbm_ranker_rank_xendcg_era | 0.000164 | 0.013279 | 0.013702 | 0.475927 |
| small_lgbm_ranker_rank_xendcg_era_x_feature_busy_clear_divi | -0.000005 | -0.000416 | 0.013711 | 0.479165 |
| small_lgbm_ranker_lambdarank_era | -0.001327 | -0.146421 | 0.005723 | 0.254247 |

Baseline reference: `small_lgbm_ender20_baseline` (small_bmc_mean = 0.000000).

## Decision
`rank_xendcg` clearly outperforms `lambdarank` on the small-experiment metrics.
If we scale any ranker runs, prioritize:
- `small_lgbm_ranker_rank_xendcg_era_x_feature_unwaked_toxophilitic_dilatability`
- `small_lgbm_ranker_rank_xendcg_era_x_feature_kookiest_unrepealable_octuplet`

`small_lgbm_ranker_rank_xendcg_era_x_feature_scholastic_doglike_sipunculid` is a
secondary candidate if we want a third option. The remaining variants are not
worth scaling.
