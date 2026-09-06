# Ender-60: techniques and output decisions

Date: 2026-09-06

This note explains the machine-learning work in this folder for someone with a **data-engineering** background. It is not a metrics dump (that lives in `experiment.md`). It is the *why*: what each technique does to the table, what can leak, and which artifacts we kept.

**Status.** Medium residual-1.0 is **live on Numerai** (`model.pkl`, serve-time per-era rank to `(0, 1]`). Watching live rounds. Next train is all-features downsample on HF Jobs `cpu-performance` — see `next_session.md` and `agents/skills/numerai-remote-compute/SKILL.md`.

---

## 1. What we are actually predicting

Numerai is a weekly stock-scoring contest. You do **not** get ticker history. Each row is one `(id, era)` pair: a stock in one week, with encrypted point-in-time features. `id` is unique per stock-era, so you cannot join a stock to its previous weeks. Sequence models, LSTMs, and “this ticker last month” features are invalid.

The **payout label** is `target_ender_60`: a 60-day, 2-day-lag residual stock return. Think of it as “how much this name beat a neutralized market over the next ~12 weeks,” already cleaned by Numerai.

Two columns look like the same thing and are not:

| column | what it is | use |
|---|---|---|
| `target` | unstable alias | do not train or score on this |
| `target_ender_20` | 20-day cousin (~0.47 corr with 60-day) | diversifier later, not the payout label |
| `target_ender_60` | live payout target (from round 1343) | **the only scoring label** |

**Data-contract rule we enforced:** every config names `target_ender_60` explicitly. Scoring BMC against `v53_lgbm_ender20` while claiming Ender-60 work is a silent metric swap.

The official competitor you are measured against is `v53_lgbm_ender60` — Numerai’s own deep LightGBM on the same table. Your payout is not “be accurate.” It is “be accurate **and** not a clone of that model.”

---

## 2. The two numbers that matter

Think of two KPIs on the same OOF predictions:

**CORR** — Spearman-style correlation of your scores vs `target_ender_60`, averaged across eras. This is “does the ranking work?” Sane scout range here is roughly **0.005–0.04**. The official model on this sample sits near 0.05. If CORR goes to ~0 or negative, the model is no longer useful as a stock ranker even if another metric looks great.

**BMC (Benchmark Model Contribution)** — how much unique signal you add *after* accounting for `v53_lgbm_ender60`. High CORR + high correlation with the official model + low BMC = you rebuilt their pipeline. **`bmc_last_200_eras.mean` is the decision metric** (recent regimes matter more than 2018). `bmc_mean` is the tie-break.

`avg_corr_with_benchmark` is the clone detector. ~0.47 means “same family as the official LGBM.” ~0.22 means “related but not a copy.” Negative means you are betting *against* them, which usually nukes CORR.

---

## 3. Data engineering choices (before any model)

### Grain and joins

- Feature table: `v5.3/downsampled_full.parquet` — every **4th era**, medium feature set (780 columns), 1.70M rows.
- Benchmark table: `v5.3/downsampled_full_benchmark_models.parquet`.
- Join key: `id` (stock-era). **Inner join.** Early eras exist in features but not in official predictions; those rows cannot be residualized or BMC-scored, so they are dropped. Overlap: 1.54M / 1.70M ids.

This is the same pattern as a fact table inner-joined to a slowly-arriving dimension. You do not impute the official model; you wait until it exists.

### Late-arriving labels

`target_ender_60` is null until the 60-day return matures. Recent validation eras look complete in the feature file and empty in the label. We `dropna` on the **explicit** target before fit and score. Training on filled zeros would be label leakage of a different kind: you would teach the model that “latest era = 0.”

### Leakage from overlapping returns

Eras are weekly. A 60-day return in era *t* shares almost two months of calendar with era *t+1*. Random row splits and ordinary k-fold are illegal: the validation fold would contain returns that were already in the training fold.

**Fix:** expanding-window CV on **eras**, plus an **embargo of 16 eras** (official 60-day purge). After each train window we skip 16 weeks before the validation window starts. Fold 0 had no train rows after the embargo, so we used 4 folds. OOF: 1.43M rows, 245 eras (245–1221).

That embargo is the time-series equivalent of a late-arriving-fact quarantine.

### Scout vs scale

Downsampled every 4th era is a **cheap cluster**. We only promote an idea after it wins on this table. Scale here meant “more trees on the same medium table,” not the official 30k-tree / all-features job — that parquet was never built (the official `full.parquet` download was aborted to avoid an 8GB pandas load).

---

## 4. Technique: gradient-boosted trees (the backbone)

**LightGBM** is a forest of decision trees trained one after another. Each new tree fits the *errors* of the current sum. For tabular, low-signal, mixed-type data, this family is the default. Numerai’s own best single model is a **deep** LightGBM (30k trees, depth 10, 1024 leaves, 1% of columns per tree).

We did **not** start with a neural net. Same features + same label + a fancier architecture mostly copies the official LGBM: high CORR, low BMC. Trees are also what the compute container can run.

Scout backbone (`scout_lgbm_ender60`):

- 400 trees, learning rate 0.02, depth 5, 31 leaves
- `min_data_in_leaf = 10000` — refuses tiny leaves. On noisy finance data, a leaf of 50 rows is a memorized fluke.
- `colsample_bytree = 0.1` — each tree sees 10% of columns. That is explicit regularization and the same trick the official model uses.

**Result:** `corr_mean` 0.0253 (works), `bmc_last_200` −0.00013 (adds nothing unique), `avg_corr_bench` 0.47 (clone-ish). This is the **CORR floor**, not the payout model. We needed it as a control: every clever idea is compared to “just train LGBM on the payout target.”

---

## 5. Technique: residual labels (the BMC engine)

This is the most important idea, and it is a **label transform**, not a new model class.

You already have the official scores `v53_lgbm_ender60` on the same rows. Instead of teaching a second LGBM to predict `y = target_ender_60`, you teach it to predict:

```text
y_residual = target_ender_60 − proportion × (what the official model already explains)
```

Implemented as a **per-era linear residual**: within each week, regress the target on the official prediction (with intercept), then keep the leftover. `proportion` is how much of that fitted piece you subtract (0.5 / 0.75 / 1.0).

DE analogy: you are training on the **error table** of an upstream production model, not on the raw fact. The features stay the same. The *job* of the new model is “what did the warehouse miss?”

Why this creates BMC:

- The official model already harvested the easy, shared signal.
- A second model on the raw target re-learns that easy signal (high corr with the official scores).
- A model on the residual is forced onto the orthogonal leftover. CORR drops (you gave away the easy part). BMC rises (your errors are no longer collinear with theirs).

**Sweep result (decision metric = last-200 BMC):**

| proportion subtracted | corr | last-200 BMC | corr with official |
|---:|---:|---:|---:|
| 0.50 | 0.0093 | 0.00091 | 0.15 |
| 0.75 | 0.0116 | 0.00248 | 0.17 |
| **1.00** | **0.0153** | **0.00558** | **0.22** |

Monotonic: more residualization → more unique contribution, and CORR stayed in the sane band. **We kept `proportion = 1.0` as the primary model.**

A harsher cousin, `subtract_benchmark_zscore` (subtract a scaled z-score of the official prediction), printed the highest BMC (0.0088) and **negative CORR** (−0.0023). That is over-neutralization: you are anti-correlated with a model that is itself correlated with the truth, so you flip the ranker. **Discarded.** High BMC with collapsed CORR is not a winner; it is a broken KPI.

---

## 6. Technique: sibling targets (multi-label, same features)

Numerai publishes several 60-day targets that are different *definitions* of residual return (`teager2b`, `victor`, `tyler`, `xerxes`). Same feature rows, different `y`.

DE analogy: several downstream KPIs computed from the same events table — related, not identical. Training a copy of the backbone on `target_teager2b_60` and scoring it on `target_ender_60` asks: “does this other KPI still rank the payout KPI?”

| sibling | role | last-200 BMC vs ender-60 |
|---|---|---:|
| teager2b | high-corr cousin | ~0 (almost the backbone) |
| victor | more diverse | +0.00032 (small unique lift) |
| tyler | more diverse | −0.00080 (hurts recent BMC) |

**Decision:** siblings are not standalone payout models. Victor is the only one worth a small tilt. Tyler is not. We never average them in raw score space (next section).

---

## 7. Technique: per-era rank-gauss blend (how we mix models)

Raw averaging of model scores is a bad join.

Each model’s output lives on its own scale. Model A’s 0.01 is not Model B’s 0.01. A 50/50 average lets the wider-scale model dominate. Eras also have different cross-sectional spreads.

**Official Numerai mix:**

1. Within each era, **rank** each model’s scores (percentile).
2. Map ranks through a **Gaussian inverse CDF** (`norm.ppf`) so the era is standard-normal.
3. Divide by the era std so variance is 1.
4. **Weighted dot product** of those standardized columns.
5. Rank-gauss the blend again.

That is `blend_rank_gauss` in `agents/code/modeling/utils/ensemble.py`. It is closer to “average the *ranks* after a normal-score transform” than to `0.5 * pred_a + 0.5 * pred_b`.

**Critical finding:** a 50/50 rank-gauss of two models trained on *related but different* targets **keeps CORR and kills BMC**. Example: scalar XGB on ender-60 + sibling teager, 50/50, last-200 BMC ≈ 0, correlation with the official model *rose* to 0.53. You averaged two slightly different clones and got a more official-looking clone.

So: mix weakly related targets **inside** a model (next section), or residualize **before** fit. Do not 50/50 them after the fact and call it diversity.

Useful blends we kept as *ensembles of already-unique members*:

- Backbone 70% + residual-1.0 30% — CORR stays ~0.025, last-200 BMC 0.00174.
- **40% backbone + 40% residual + 20% vector-leaf** — CORR 0.0252, last-200 BMC **0.00269**. Best “don’t give up CORR” mix.

---

## 8. Technique: XGBoost vector-leaf (one tree, two outputs)

XGBoost 3.4+ can grow **one tree that predicts several targets at once** (`multi_strategy=multi_output_tree`, “vector-leaf”). Shared splits, a vector of values in each leaf.

DE analogy: one DAG that writes two related facts, sharing the grouping keys, versus two independent jobs whose outputs you join and average.

We trained `{ender_60, teager2b_60}` (and tyler / victor variants) with the same tree budget as the scalar XGB. At serve time we read only the ender-60 head.

| how you mix ender + teager | CORR | last-200 BMC |
|---|---:|---:|
| scalar XGB on ender only | 0.0244 | −0.00009 |
| 50/50 rank blend of two scalars | 0.0269 | −0.00006 |
| **one vector-leaf model** | **0.0264** | **+0.00083** |

Same information, different computation graph. Shared trees keep a unique component that prediction-level averaging washes out.

**Decision:** vector-leaf is a **supporting slot**, not the primary model. Residual-1.0 still wins BMC by a wide margin. Vector-leaf is the thing you keep when you need backbone-like CORR *and* a little BMC.

Tyler/victor vector-leaf did not beat that teager pair. We did not promote them.

---

## 9. Technique: scale (capacity confirmation)

A 400-tree win on every-4th-era data can be a lucky scout. Confirmation: same residual-1.0 recipe, **2000 trees**, slightly deeper (depth 6 / 63 leaves), same medium features.

| run | trees | CORR | last-200 BMC | corr with official |
|---|---:|---:|---:|---:|
| residual scout | 400 | 0.0153 | 0.00558 | 0.22 |
| residual scale | 2000 | 0.0195 | 0.00372 | 0.33 |

More capacity **bought CORR and spent uniqueness**. The model started re-learning the official signal. It still beat the backbone on BMC (0.00372 vs −0.00013), so the residual idea is real, not a one-fold fluke. It did **not** beat the smaller residual on the decision metric.

**Decision:** do not ship the 2000-tree scale as the BMC model. The 400-tree residual is the upload artifact. This is the opposite of the usual “bigger model wins” instinct, and it matches the product KPI: uniqueness, not in-sample fit.

We stopped after that confirmatory step plus a blend plateau (40/40/20 vs 50/30/20). Two consecutive non-improving rounds.

---

## 10. What we refused to do

| idea | why not |
|---|---|
| Train on the `target` alias | identity is not stable; silent label drift |
| Score BMC vs `v53_lgbm_ender20` | wrong benchmark for a 60-day payout |
| Neural nets / transformers first | same features + same y ≈ clone of official LGBM; expensive |
| Stock-history / graph models | no persistent stock key across eras |
| Tune 20 hyperparameters at once | you cannot attribute a win |
| Treat one downsampled run as production | required the scale confirmation |
| Ship `subtract_scale_007` | BMC high because CORR went negative |

---

## 11. Decisions on outputs (what to actually use)

There are three artifacts, for three jobs.

### A. BMC-max model — **the one we pickled**

- **Who:** `residual_prop_100` — LGBM on `target_ender_60` after full per-era residual to `v53_lgbm_ender60`.
- **Why:** best last-200 BMC (0.00558) with CORR still sane (0.0153).
- **File:** `predictions/residual_prop_100.parquet` (OOF), `results/residual_prop_100.json`, `model.pkl` (fit on all 1.70M labeled rows).
- **Serve path:** `predict(live_features, live_benchmark_models)` uses **features only**. Residualization is a *training-label* transform. At inference the official scores are not required. The pickle is LightGBM + a 780-name column list; no repo imports.

This is the model uploaded to a Classic slot (after ranking live scores into `(0, 1]`). Watching live rounds. Next scale is remote all-features residual, not a MacBook retrain.

### B. CORR-preserving ensemble — **best “balanced” output**

- **Who:** `blend_arch_40_40_20` = rank-gauss mix of scout backbone (40) + residual-1.0 (40) + vector-leaf teager (20).
- **Why:** CORR matches the backbone (0.0252) while last-200 BMC is 0.00269 — about half the residual’s BMC, without giving up ranking quality.
- **File:** `predictions/blend_arch_40_40_20.parquet`.
- **Caveat:** this blend is OOF-only. A live ensemble would need three pickled models plus the rank-gauss join at serve time. We did not package that, because the plan’s upload target was the single residual winner.

### C. Controls we keep on disk but do not ship

| artifact | verdict |
|---|---|
| `scout_lgbm_ender60` | CORR floor / clone control |
| `residual_prop_050` / `075` | dominated by 1.0 |
| `subtract_scale_007` | reject (negative CORR) |
| sibling standalones and 10–30% tilts | do not beat residual; tyler hurts |
| `xgb_scalar_ender60` | no BMC |
| `blend_50_xgb_teager` | proof that naive mixes kill BMC |
| `scale_lgbm_ender60` | confirms residual, worse BMC than scout residual |

---

## 12. How to read the plots

`plots/v53_lgbm_ender60_vs_residual_prop_100_plus_4_dark.png` — cumulative CORR and BMC from era 575. The official model’s BMC line is flat zero (it cannot contribute to itself). Residual pulls away on BMC; the backbone stays near the axis; the 40/40/20 blend sits in between with a healthier CORR path.

`plots/v53_lgbm_ender60_vs_xgb_vector_ender_teager_plus_2_dark.png` — vector-leaf vs scalar vs 50/50 blend. The blend hugs the official model; vector-leaf is the only one of the three that accumulates BMC.

---

## 13. Mental model (one paragraph)

Treat the official deep LGBM as **production**. Your job is not to rebuild production. Your job is a **downstream residual job** on the same features, with a time-aware train window, late-label drop, and an inner join to production scores that only exist after a lag. Mix additional related labels *inside* a multi-output tree if you want diversity; do not average those jobs’ outputs and expect uniqueness. Promote the residual-1.0 LGBM for BMC. If a stakeholder needs the backbone’s CORR, serve the 40/40/20 rank-gauss ensemble instead.
