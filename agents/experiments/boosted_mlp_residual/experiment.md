# Boosted MLP Residual Sweep

Baseline (phase 1): `small_lgbm_ender20_baseline`

Goal: reach `small_bmc_mean >= 0.01` on downsampled data before moving to the deep baseline phase.

## Results (phase 1: medium feature set)

| model | small_bmc_mean | small_bmc_last200 | bmc_mean | bmc_last200 | corr_mean | corr_w_baseline_avg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small_lgbm_ender20_baseline | 0.000000 | 0.000000 | 0.003670 | 0.002731 | 0.021804 | 0.554719 | baseline |
| medium_r1_base | 0.000664 | 0.000664 | 0.004178 | 0.004178 | 0.021406 | 0.504373 | base |
| medium_r1_lr_high | -0.000247 | -0.000247 | 0.000820 | 0.000820 | 0.019422 | 0.551770 | lr=2e-3 |
| medium_r1_dropout0 | 0.001809 | 0.001809 | 0.002838 | 0.002838 | 0.017732 | 0.442666 | dropout=0.0 |
| medium_r1_weight_decay5e-4 | 0.000664 | 0.000664 | 0.004178 | 0.004178 | 0.021404 | 0.504292 | weight_decay=5e-4 |
| medium_r2_base | 0.000941 | 0.000941 | 0.002603 | 0.002603 | 0.020648 | 0.530639 | deeper/wider |
| medium_r2_dropout0p05 | 0.001485 | 0.001485 | 0.003398 | 0.003398 | 0.020617 | 0.509722 | dropout=0.05 |
| medium_r2_dropout0p1 | 0.000788 | 0.000788 | 0.003720 | 0.003720 | 0.021285 | 0.517818 | dropout=0.1 |
| medium_r2_dropout0p2 | 0.000307 | 0.000307 | 0.003051 | 0.003051 | 0.022079 | 0.565748 | dropout=0.2 |
| medium_r3_base | 0.001590 | 0.001590 | 0.003026 | 0.003026 | 0.020045 | 0.496056 | lr=5e-4 |
| medium_r3_lr_1e-3 | 0.000579 | 0.000579 | 0.002334 | 0.002334 | 0.020834 | 0.546112 | lr=1e-3 |
| medium_r3_lr_2e-4 | 0.002779 | 0.002779 | 0.002453 | 0.002453 | 0.017627 | 0.442172 | lr=2e-4 |
| medium_r3_lr_1e-4 | 0.003040 | 0.003040 | 0.001723 | 0.001723 | 0.016889 | 0.446518 | lr=1e-4 |
| medium_r4_base | 0.004378 | 0.004378 | 0.003416 | 0.003416 | 0.015601 | 0.384111 | deeper/wider, lr=1e-4 |
| medium_r4_lr_5e-5 | 0.004505 | 0.004505 | 0.002495 | 0.002495 | 0.015532 | 0.410263 | lr=5e-5 (best small_bmc) |
| medium_r4_lr_2e-4 | 0.003469 | 0.003469 | 0.004522 | 0.004522 | 0.018128 | 0.399620 | lr=2e-4 |
| medium_r4_lr_3e-4 | 0.002322 | 0.002322 | 0.001576 | 0.001576 | 0.017460 | 0.492070 | lr=3e-4 |
| medium_r5_base | 0.004388 | 0.004388 | 0.003181 | 0.003181 | 0.014276 | 0.353215 | lr=5e-5, epochs=80 |
| medium_r5_batch1024 | 0.003253 | 0.003253 | 0.000230 | 0.000230 | 0.011584 | 0.341923 | batch_size=1024 |
| medium_r5_batch2048 | 0.004182 | 0.004182 | 0.001699 | 0.001699 | 0.012856 | 0.340978 | batch_size=2048 |
| medium_r5_batch8192 | 0.003757 | 0.003757 | 0.002151 | 0.002151 | 0.016574 | 0.437288 | batch_size=8192 |
| medium_r6_base | 0.004388 | 0.004388 | 0.003181 | 0.003181 | 0.014276 | 0.353215 | weight_decay sweep base |
| medium_r6_wd_0 | 0.004388 | 0.004388 | 0.003181 | 0.003181 | 0.014276 | 0.353215 | weight_decay=0.0 |
| medium_r6_wd_1e-5 | 0.004388 | 0.004388 | 0.003181 | 0.003181 | 0.014276 | 0.353215 | weight_decay=1e-5 |
| medium_r6_wd_5e-4 | 0.004388 | 0.004388 | 0.003181 | 0.003181 | 0.014276 | 0.353215 | weight_decay=5e-4 |

## Results (phase 2: all feature set)







| model | small_bmc_mean | small_bmc_last200 | bmc_mean | bmc_last200 | corr_mean | corr_w_baseline_avg | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deep_lgbm_ender20_baseline_downsampled | 0.008683 | 0.009230 | 0.002216 | 0.001310 | 0.025003 | 0.696588 | baseline |
| all_residual_only_size_big | 0.003541 | 0.003541 | 0.005079 | 0.005079 | 0.003504 | -0.073053 |  |
| all_r12_es240 | 0.001282 | 0.001282 | 0.002875 | 0.002875 | 0.025655 | 0.680799 |  |
| all_r12_es160 | 0.001282 | 0.001282 | 0.002875 | 0.002875 | 0.025655 | 0.680799 |  |
| all_r10_size_big | 0.001282 | 0.001282 | 0.002875 | 0.002875 | 0.025655 | 0.680799 |  |
| all_r12_es120 | 0.001282 | 0.001282 | 0.002875 | 0.002875 | 0.025655 | 0.680799 |  |
| all_r12_es200 | 0.001282 | 0.001282 | 0.002875 | 0.002875 | 0.025655 | 0.680799 |  |
| all_r8_lr_9e-4 | 0.001265 | 0.001265 | 0.002523 | 0.002523 | 0.024908 | 0.672189 |  |
| all_r11_size_xl2 | 0.001032 | 0.001032 | 0.002380 | 0.002380 | 0.025390 | 0.672273 |  |
| all_r11_size_xl1 | 0.000645 | 0.000645 | 0.002328 | 0.002328 | 0.026009 | 0.695344 |  |
| all_r11_size_xl3 | 0.000920 | 0.000920 | 0.002216 | 0.002216 | 0.026188 | 0.706380 |  |
| all_r8_lr_8e-4 | 0.001081 | 0.001081 | 0.002178 | 0.002178 | 0.025113 | 0.684175 |  |
| all_r10_depth_more | 0.000782 | 0.000782 | 0.002166 | 0.002166 | 0.022060 | 0.595906 |  |
| all_r9_lr_9p5e-4 | 0.000785 | 0.000785 | 0.001944 | 0.001944 | 0.023934 | 0.660013 |  |
| all_r7_lr_7e-4 | 0.001022 | 0.001022 | 0.001898 | 0.001898 | 0.023022 | 0.631820 |  |
| all_r8_lr_7e-4 | 0.001022 | 0.001022 | 0.001898 | 0.001898 | 0.023022 | 0.631820 |  |
| all_r6_lr_1e-3 | 0.000464 | 0.000464 | 0.001811 | 0.001811 | 0.026190 | 0.717324 |  |
| all_r7_lr_1e-3 | 0.000464 | 0.000464 | 0.001811 | 0.001811 | 0.026190 | 0.717324 |  |
| all_r10_depth_less | 0.000525 | 0.000525 | 0.001717 | 0.001717 | 0.023727 | 0.648685 |  |
| all_r11_size_xl4 | 0.000368 | 0.000368 | 0.001689 | 0.001689 | 0.025975 | 0.719210 |  |
| all_r8_lr_6e-4 | 0.000995 | 0.000995 | 0.001668 | 0.001668 | 0.022260 | 0.616561 |  |
| all_r4_dropout0 | 0.000239 | 0.000239 | 0.001639 | 0.001639 | 0.025957 | 0.720429 |  |
| all_r5_batch4096 | 0.000239 | 0.000239 | 0.001639 | 0.001639 | 0.025957 | 0.720429 |  |
| all_r3_lr_5e-3 | 0.000239 | 0.000239 | 0.001639 | 0.001639 | 0.025957 | 0.720429 |  |
| all_r9_lr_1p15e-3 | 0.000256 | 0.000256 | 0.001628 | 0.001628 | 0.026274 | 0.727834 |  |
| all_r9_lr_8p5e-4 | 0.000262 | 0.000262 | 0.001625 | 0.001625 | 0.025500 | 0.702038 |  |
| all_r10_size_small | 0.000146 | 0.000146 | 0.001608 | 0.001608 | 0.024659 | 0.672247 |  |
| all_r7_lr_1p3e-3 | 0.000250 | 0.000250 | 0.001590 | 0.001590 | 0.026360 | 0.730244 |  |
| all_r7_lr_1p6e-3 | 0.000237 | 0.000237 | 0.001578 | 0.001578 | 0.026357 | 0.731109 |  |
| all_r3_lr_1e-3 | 0.000247 | 0.000247 | 0.001563 | 0.001563 | 0.026309 | 0.728623 |  |
| all_r6_lr_5e-3 | 0.000140 | 0.000140 | 0.001422 | 0.001422 | 0.025802 | 0.723557 |  |
| all_r6_lr_2e-3 | 0.000064 | 0.000064 | 0.001414 | 0.001414 | 0.026151 | 0.726334 |  |
| all_r9_lr_1p05e-3 | 0.000615 | 0.000615 | 0.001398 | 0.001398 | 0.023339 | 0.658123 |  |
| all_r5_batch1024 | 0.000110 | 0.000110 | 0.001369 | 0.001369 | 0.026163 | 0.733374 |  |
| all_r3_lr_2e-3 | 0.000068 | 0.000068 | 0.001360 | 0.001360 | 0.026128 | 0.730466 |  |
| all_r5_batch2048 | 0.000056 | 0.000056 | 0.001352 | 0.001352 | 0.025882 | 0.726300 |  |
| all_r5_batch8192 | 0.000012 | 0.000012 | 0.001340 | 0.001340 | 0.025886 | 0.724081 |  |
| all_r2_lr_5e-4 | 0.000698 | 0.000698 | 0.001212 | 0.001212 | 0.022037 | 0.620268 |  |
| all_r3_lr_5e-4 | 0.000698 | 0.000698 | 0.001212 | 0.001212 | 0.022037 | 0.620268 |  |
| all_r4_dropout0p2 | -0.000197 | -0.000197 | 0.000868 | 0.000868 | 0.025193 | 0.720048 |  |
| all_r4_dropout0p3 | -0.000262 | -0.000262 | 0.000819 | 0.000819 | 0.025849 | 0.737804 |  |
| all_era_zscore_size_big | -0.000516 | -0.000516 | 0.000796 | 0.000796 | 0.014475 | 0.392045 |  |
| all_r2_lr_2e-4 | 0.000579 | 0.000579 | 0.000508 | 0.000508 | 0.019714 | 0.570686 |  |
| all_r6_lr_1e-2 | -0.000780 | -0.000780 | 0.000436 | 0.000436 | 0.016512 | 0.521025 |  |
| all_r2_lr_5e-5 | 0.001640 | 0.001640 | 0.000422 | 0.000422 | 0.017477 | 0.519290 |  |
| all_r1_one_cycle | 0.000953 | 0.000953 | 0.000275 | 0.000275 | 0.018551 | 0.544339 |  |
| all_r1_cosine_warm | 0.001386 | 0.001386 | 0.000248 | 0.000248 | 0.016668 | 0.505870 |  |
| all_r2_lr_1e-4 | 0.001386 | 0.001386 | 0.000248 | 0.000248 | 0.016668 | 0.505870 |  |
| all_residual_only_lgbm_deep | 0.002938 | 0.002938 | 0.000241 | 0.000241 | 0.011511 | 0.343609 |  |
| all_r1_cosine | 0.000957 | 0.000957 | 0.000130 | 0.000130 | 0.018770 | 0.562653 |  |
| all_r1_step | 0.000894 | 0.000894 | 0.000074 | 0.000074 | 0.019258 | 0.579811 |  |
| all_r4_dropout0p1 | -0.000983 | -0.000983 | -0.000293 | -0.000293 | 0.023669 | 0.710276 |  |
| all_scaled_size_big | -0.002423 | -0.002423 | -0.000772 | -0.000772 | 0.001882 | 0.073172 |  |

## Final Report

### Scope
- Downsampled-only experiments (no full data runs).
- Phase 1: medium feature set vs `small_lgbm_ender20_baseline`.
- Phase 2: all feature set vs `deep_lgbm_ender20_baseline_downsampled`.

### Best Result (current winner)
- Model: residual-only Torch MLP (no add-back of baseline predictions).
- Config: `agents/experiments/boosted_mlp_residual/configs/all_deep/r15_residual_only_size_big.py`
- Data: `v5.2/downsampled_full.parquet`, feature_set=`all`.
- Residual target: `target - baseline_pred` with baseline from `agents/baselines/predictions/deep_lgbm_ender20_baseline_downsampled.parquet`.
- Architecture: `(1536, 1024, 768, 512, 256)`, GELU, dropout=0.0, batch=4096, lr=9e-4, weight_decay=1e-4, epochs=80, early_stopping=True (patience=6), AMP on CUDA.
- Metrics: bmc_mean `0.005079`, bmc_last200 `0.005079`, small_bmc_mean `0.003541`, corr_mean `0.003504`, corr_w_baseline_avg `-0.073053`.

### Key Comparisons
- Add-back residual MLP (previous best) reached bmc_mean `0.002875`.
- Residual-only LGBM deep sanity check (same boosting strategy) reached bmc_mean `0.000241`.
- Residual-only MLP outperformed the add-back MLP and the LGBM residual control by a wide margin on BMC.

### What Worked
- Residual-only predictions (no add-back) improved BMC substantially.
- Larger MLP capacity helped (size_big variants dominated).

### What Did Not Work
- Unit-variance scaling of residuals (worse BMC).
- Per-era zscore scaling (worse BMC).
- LGBM residual-only control (BMC close to zero).

### Plots
- Cumulative BMC comparison (overlap-only eras): `agents/experiments/boosted_mlp_residual/plots/deep_lgbm_ender20_baseline_downsampled_vs_all_residual_only_size_big.png`
- Cumulative MMC (meta model) for final model: `agents/experiments/boosted_mlp_residual/plots/all_residual_only_size_big_mmc_cumsum.png`

### MMC (meta model)
- MMC mean: `0.001661` (meta column: `numerai_meta_model` from `v5.2/meta_model.parquet`).
