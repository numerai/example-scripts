# Experiment Plan

## Current baseline
- Baseline model: `small_lgbm_ender20_baseline` (medium feature set; 2k‑tree LGBM).
- Goal: compare modifications against baseline using `bmc_last_200_eras` and
  `small_bmc_last200`, keeping `feature_set` aligned for comparability.

## Active project: catboost_specific_params
- Objective: verify whether CatBoost variants can beat the medium-feature LGBM baseline.
- Status: rerun complete; `small_catboost_random_strength_zero` edges the baseline on bmc_mean.
- Follow-up: consider a deep-scale run or multi-seed confirmation for the best variant.

## Active project: lgbm_ranker_serenity_groups
- Objective: test learning-to-rank objectives and era x serenity groupings with medium features.
- Status: complete; `rank_xendcg` best but still below baseline on bmc_mean.

## Next projects (near-term)
- TabPFN baseline (OOF CV) for feasibility on Numerai tabular data.
  - Blocked: gated model access on Hugging Face (must accept terms + auth).
  - Blocked: CPU-only guardrail (>1k samples) unless we set overrides or use GPU/client.
- Hyperparameter sweeps:
  - fewer trees / different learning rates
  - subsampling variants
  - era-weighting
  - missing-value handling (e.g., NaN filling vs leave-as-is)

## Future projects
- Alternate GBDT architectures / losses:
  - CatBoost (deprioritized after small_bmc_last200 sweep showed no gains)
  - XGBoost
  - rankedboost
- Neural network exploration
- Training on different targets

## Notes / findings (keep updated)
- Baseline standardized to medium feature set for all small experiments.
- CatBoost variant `small_catboost_random_strength_zero` slightly beats the LGBM baseline on bmc_mean and shows positive small_bmc_mean.
- LGBM ranker experiments (lambdarank/rank_xendcg) did not surpass the baseline; no deep scaling planned.
- TabPFN baseline still blocked due to gated model access + CPU-only dataset limits.
