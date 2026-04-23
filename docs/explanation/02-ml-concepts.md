# 02, ML Concepts Primer

> **What this page answers:** Every ML and statistics idea the project
> uses, in the order they appear. Read it top to bottom the first time,
> then use it as a reference.

Each concept has four parts: **Intuition**, **Math (minimal)**, **In this
codebase**, and **Limits / gotchas**. If a concept only needs intuition,
the other sections are one line each.

---

## 1. Supervised vs unsupervised learning

**Intuition.** Supervised learning gets labeled examples ("this
transaction was fraud, this one was not") and learns to reproduce the
label. Unsupervised learning gets no labels, it has to find structure
on its own.

**Math.** Supervised: learn a function `f(x) -> y` that minimises a loss
between predicted `y_hat` and true `y`. Unsupervised (in the form used
here, autoencoders): learn `f(x) -> x`, i.e. reproduce the input from a
compressed representation. Things it reproduces poorly are "weird".

**In this codebase.** XGBoost (`training/train_xgboost.py`) is
supervised: it uses the `Class` column (0/1). The autoencoder
(`training/train_autoencoder.py`) is unsupervised: it only sees
non-fraud rows (line 209 masks `y_train == 0`) and is judged by how
badly it reconstructs fraud.

**Limits.** Supervised needs labels, which are expensive when fraud is
rare. Unsupervised catches novel patterns but usually has lower
precision (more false alarms).

---

## 2. Class imbalance

**Intuition.** When positives are rare, a model can get high accuracy by
ignoring positives entirely. 0.17% fraud means "always predict legit" is
99.83% accurate, and useless.

**Math.** Base rate = `positives / total`. With a 0.17% base rate, if
recall is 50% (you catch half the fraud) but precision is only 2%, you're
still drowning your fraud team in false alarms.

**In this codebase.** The project uses **three** countermeasures stacked,
because each alone is insufficient:

1. **SMOTE** oversampling in training (`train_xgboost.py:118-119`),
   see [§3](#3-smote).
2. **`scale_pos_weight`** passed to XGBoost (`train_xgboost.py:86, 92`),
   multiplies the loss contribution of positive examples by
   `n_neg / n_pos` (about 580 for this dataset). "Belt and suspenders"
   alongside SMOTE per the code comment on line 86.
3. **Threshold tuning** on the PR curve with an asymmetric cost
   (`evaluate.py:47-72`), see [§5](#5-decision-threshold-tuning).

**Limits.** Over-sampling can cause the model to over-fit to synthetic
minority examples; too-high `scale_pos_weight` over-predicts fraud
everywhere, destroying precision.

---

## 3. SMOTE

**Intuition.** Instead of duplicating rare positive examples (plain
oversampling) or throwing away majority examples (undersampling), SMOTE
(Synthetic Minority Oversampling Technique) generates **new synthetic
positive rows** by interpolating between existing positives and their
nearest neighbours.

**Math.** For each minority point `x_i`, find its `k` nearest minority
neighbours. Pick one, `x_j`. Generate a synthetic point along the line:
`x_new = x_i + u * (x_j - x_i)` where `u` is uniform in `[0, 1]`.
Repeat until classes are balanced.

**In this codebase.** `imblearn.over_sampling.SMOTE` is applied *after*
scaling and *only* to the training split (`train_xgboost.py:117-123`).
It is never applied to the validation set, doing so would leak
synthetic data into evaluation and give you a falsely optimistic score.

**Limits.**

- SMOTE assumes the minority class is locally continuous in feature
  space. When positives form tiny isolated clusters, interpolations can
  land in "empty" regions that aren't actually fraud.
- It amplifies any noise in the minority labels.
- In high dimensions, nearest-neighbour distances become less
  meaningful. This dataset is only 33 dimensions, so SMOTE is fine.

Further reading: Chawla et al., "SMOTE: Synthetic Minority
Over-sampling Technique", JAIR 2002.

---

## 4. ROC-AUC vs PR-AUC

**Intuition.** A classifier outputs a probability. You pick a threshold;
above it you predict positive. ROC-AUC and PR-AUC summarise how good
the probability ranking is across *all* possible thresholds, so they
don't depend on picking one.

- **ROC-AUC** plots True Positive Rate vs False Positive Rate. On heavily
  imbalanced data it is misleadingly optimistic because FPR has a huge
  denominator (all the legit rows). A terrible model can look great.
- **PR-AUC** plots Precision vs Recall. It ignores true negatives
  entirely, so rare-positive performance is honest.

**Math.**

- `TPR = TP / (TP + FN)` (a.k.a. recall)
- `FPR = FP / (FP + TN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- ROC-AUC = area under the TPR(FPR) curve
- PR-AUC = area under the precision(recall) curve
  (computed as `average_precision_score` in `evaluate.py:39`)

**In this codebase.** `training/evaluate.py:25-44` logs **both** metrics
for every run. The `plan.md` acceptance target (AUC-ROC > 0.95) is the
headline number; PR-AUC is the one the project uses to compare models
honestly (`retrain_dag.py:112` compares candidates on PR-AUC, not
AUC-ROC).

**Limits.**

- PR-AUC is sensitive to class prior, it can't be compared across
  datasets with different base rates.
- Neither metric tells you how a *single* threshold performs; pair them
  with confusion-matrix metrics (see [§5](#5-decision-threshold-tuning)).

---

## 5. Decision threshold tuning

**Intuition.** Your model emits `P(fraud) = 0.43`. Is that fraud or not?
Default answer is "threshold at 0.5" but that's an arbitrary choice.
If missing fraud hurts 10x more than a false alarm, you should lower
the threshold.

**Math.** Total cost at threshold `t`:

```
cost(t) = cost_fp * FP(t) + cost_fn * FN(t)
```

Optimum is the `t` that minimises this expression. If `cost_fn = 10 *
cost_fp`, the optimum is lower than 0.5 because FN is more expensive.

**In this codebase.** `training/evaluate.py:47-72` implements exactly
this:

```python
for t in thresholds:           # thresholds come from precision_recall_curve
    y_pred = (y_pred_proba >= t).astype(int)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    cost = cost_fp * fp + cost_fn * fn
    if cost < best_cost:
        best_cost = cost
        best_threshold = float(t)
```

The default ratio `cost_fn=10, cost_fp=1` (line 51) encodes "missing a
fraud is 10x worse than a false alarm". The chosen threshold is logged
as an MLflow metric (`train_xgboost.py:130, 143`) and later fetched by
the serving layer (`serving/app/models/loader.py:103`) so training and
serving use the **same** cutoff.

**Limits.**

- The 10:1 ratio is a guess. A real system would calibrate this from
  business cost data.
- If class priors shift in production (say, fraud rate doubles), the
  optimal threshold changes too, fixed-threshold deployments drift
  silently.

---

## 6. Gradient-boosted trees and XGBoost

**Intuition.** A single decision tree is a short, interpretable model
that often underfits. If you train tree after tree, each one focused on
the examples the previous trees got wrong, and add up their outputs,
you get a very strong ensemble. That's gradient boosting. XGBoost is a
fast, regularised implementation of it.

**Math (minimal).** You want to learn `F(x) = sum_k f_k(x)` where each
`f_k` is a regression tree. At each step `k`, the new tree is trained
to approximate the **negative gradient of the loss** with respect to
the current prediction, effectively correcting the residuals. For
binary classification with log-loss, those residuals are `y - p`.

**In this codebase.**

- `training/train_xgboost.py:88-97` configures XGBoost:
  300 trees, max depth 6, learning rate 0.05,
  `eval_metric="aucpr"` (optimises PR-AUC directly),
  `scale_pos_weight` for imbalance.
- SMOTE runs before `fit`, so the model effectively trains on a
  balanced dataset even though the raw one isn't.
- The trained booster is logged to MLflow as
  `mlflow.xgboost.log_model` (lines 163-168) along with the fitted
  `StandardScaler` as a separate artifact (lines 156-160). The scaler
  is logged separately because XGBoost's native format doesn't know
  about scikit-learn transforms.

**Limits.**

- Gradient boosting on tabular data is hard to beat, but on images,
  text, or time-series it's the wrong tool.
- Heavy depth (>8) or too many estimators (>1000) over-fits fast.
- No calibration is done; `predict_proba` outputs can drift from true
  probabilities, which matters if you threshold around 0.5, this is
  partly why the project tunes the threshold in [§5](#5-decision-threshold-tuning).

Further reading: Chen & Guestrin, "XGBoost: A Scalable Tree Boosting
System", KDD 2016.

---

## 7. Autoencoders for anomaly detection

**Intuition.** An autoencoder is a neural network shaped like a funnel:
it compresses its input through a narrow "bottleneck" and tries to
reconstruct the original from the compressed version. If you only show
it **normal** transactions during training, it gets really good at
reconstructing normal and really bad at reconstructing fraud. The
reconstruction error becomes your anomaly score.

**Architecture used here** (`train_autoencoder.py:70-98`):

```
Input(33) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(16) -> ReLU
          -> Linear(32) -> ReLU -> Linear(64) -> ReLU -> Linear(33)
```

The decoder mirrors the encoder, a common convention, not a hard rule.

**Math.** Training loss is MSE between input and reconstruction:
`L(x) = mean((x - f(x))^2)`. At inference, you compute
`e(x) = mean((x - f(x))^2)` and compare to a threshold.

**Turning error into probability.** A raw reconstruction error isn't
bounded. The pyfunc wrapper (`train_autoencoder.py:123-138`) normalises
by dividing by `2 * threshold` and clipping to `[0, 1]`. That's a
pragmatic choice, it keeps the response shape identical to XGBoost's
so both models look the same to the serving layer.

**In this codebase.**

- Trained on legit rows only (`train_autoencoder.py:209-213`).
- Saved via TorchScript (`torch.jit.script`) so it can load inside
  MLflow without the original PyTorch class definition (lines 273-274).
- Wrapped as an `mlflow.pyfunc.PythonModel` that handles scaling +
  inference + error-to-probability mapping in one `predict()` call.

**Limits.**

- No SHAP explanations (TreeExplainer doesn't apply to neural nets;
  KernelExplainer is too slow for serving, see [§9](#9-shap-values)).
- The 99th-percentile-of-legit trick (`train_autoencoder.py:228-232`)
  is pragmatic but not principled.
- Autoencoders can silently "memorise" training data if the bottleneck
  is too large; 33 -> 16 is aggressive enough here.

Further reading: Goodfellow, Bengio, Courville, *Deep Learning*,
chapter 14 on autoencoders.

---

## 8. Why both models (champion + challenger)

**Intuition.** The two models have different biases: XGBoost is strong
when labeled fraud looks like past labeled fraud; the autoencoder is
strong when *novel* fraud looks different from normal. Running both in
parallel gives you a portfolio of detectors and, critically, a way to
compare them on the same live traffic via A/B testing.

**In this codebase.**

- XGBoost is the **champion**, registered as `fraud-xgboost@champion`
  in MLflow. All non-A/B traffic goes here.
- The autoencoder is the **challenger**, registered as
  `fraud-autoencoder@challenger`. A configurable fraction of traffic
  (`AB_CHALLENGER_FRACTION`, default 0.20) is routed to it via the
  deterministic hash in `serving/app/models/ab_testing.py`.

See [§11](#11-ab-testing-in-ml-serving) and
[06, Serving API](06-serving-api.md#a-b-routing) for the routing
mechanics.

---

## 9. SHAP values

**Intuition.** You want to explain a single prediction: "why did the
model flag this transaction?" SHAP (SHapley Additive exPlanations)
assigns each feature a **contribution** to the prediction, using a
fair allocation rule from cooperative game theory: a feature's
contribution is its average marginal impact when added to every
possible subset of other features.

**Math (the idea, not the formula).** For `n` features, there are
`2^n` subsets. SHAP computes the average change in the model's output
when a feature is added, weighted across all orderings. For
general models, that's exponential. For tree-based models,
**TreeSHAP** computes the same result in polynomial time by walking
the trees smartly.

**Properties** (why SHAP, not just "feature importance"):

- Contributions **sum** to the prediction minus a baseline, so
  explanations are additive and not handwavy.
- They're **consistent**: if a model changes so a feature matters more,
  its SHAP value increases too.
- They handle feature interactions correctly (unlike permutation
  importance, which double-counts).

**In this codebase.** `serving/app/models/explainer.py:22` uses
`shap.TreeExplainer`. The `_explain()` call in
`serving/app/routes/predict.py:34-41` is invoked on every `/predict`
request for the champion (XGBoost). The top 3 features by absolute
contribution are returned (`explainer.py:41-44`).

**Why only for XGBoost?**

- `TreeExplainer` only works on tree models (XGBoost, LightGBM,
  Random Forests, etc.).
- `KernelExplainer` works on any model but makes ~2000 model calls
  per prediction, too slow for serving.
- The autoencoder's reasoning is "error was high on these dimensions",
  which is a different kind of explanation and isn't implemented.

**Limits.**

- SHAP costs 10-50 ms per prediction on this model size, not free.
- Contributions are in scaled feature space, which is slightly less
  intuitive than raw dollars / seconds (the serving layer returns the
  feature **name** though, so the reader knows which dimension).
- SHAP is post-hoc: it doesn't guarantee the model is "correct", only
  that you understand what it's doing.

Further reading: Lundberg & Lee, "A Unified Approach to Interpreting
Model Predictions", NeurIPS 2017. Online book:
<https://christophm.github.io/interpretable-ml-book/shap.html>.

---

## 10. Data drift and concept drift

**Intuition.**

- **Data drift** = the *inputs* to the model look different from
  training (e.g. average transaction amount doubles because the app
  expanded to a wealthier country).
- **Concept drift** = the *relationship between inputs and the label*
  changes (e.g. a previously benign merchant starts being used for
  fraud). Inputs can look identical and the model still breaks.

Both degrade the model silently, training metrics won't flag them.

**Math.** Data drift is usually detected with per-feature distributional
tests:

- **Kolmogorov-Smirnov (KS)** for continuous features, compares
  empirical CDFs, returns a p-value. Evidently uses it by default.
- **Chi-squared** for categorical features.
- **Population Stability Index (PSI)** as an overall bucketing
  measure.

**In this codebase.** `scripts/drift_report.py` uses Evidently's
`DataDriftPreset` to compare a reference (training) sample to a current
(serving) sample and emits an HTML report to `data/reports/`. It's run
manually via `make drift-report`. It is **not** integrated into
monitoring, that was an explicit scope decision (`plan.md:713`).

**Limits.**

- Feature-by-feature drift tests can miss *joint* drift
  (distributions of each feature look fine, but pairs no longer look
  the same).
- The current script doesn't ingest actual serving data, there is no
  prediction log. A real system would write predictions to the
  database and use the last 7 days as the "current" window.

---

## 11. A/B testing in ML serving

**Intuition.** You want to know which of two models is better in the
real world, not just on offline test data. You route a fraction of
traffic to the challenger, measure performance on both, and compare.

**The must-have property: deterministic bucketing.** The same
transaction (or user) must always go to the same model, otherwise you
can't reproduce bugs or compare fairly (some requests would get model
A's answer on retry and model B's on the first try).

**In this codebase** (`serving/app/models/ab_testing.py:13-26`):

```python
digest = hashlib.md5(transaction_id.encode(), usedforsecurity=False).hexdigest()
bucket = int(digest, 16) % 100
return bucket < int(challenger_fraction * 100)
```

- `hashlib.md5` is a good uniform hash (cryptographic strength is
  irrelevant here, hence `usedforsecurity=False`).
- `% 100` gives a bucket in `[0, 99]`.
- `fraction=0.20` puts buckets `[0, 19]` onto challenger, the rest on
  champion.

The test file `serving/tests/test_ab_testing.py` verifies
determinism (same ID -> same result 100 times in a row) and the
approximate split ratio (10,000 random IDs with fraction 0.20 land
within +/- 2% of 20% challenger).

**Limits.**

- Bucketing is per transaction, not per user. A real A/B test on user
  experience would bucket by user ID.
- No statistical-significance test is computed; you'd compare PR-AUC
  between model variants offline.
- No ramp-up logic, you'd normally start at 1% and widen.

---

## 12. Feature engineering for tabular fraud

**Intuition.** The raw Kaggle dataset provides 28 PCA components
(`V1`-`V28`) plus `Time` and `Amount`. The PCA features are already
highly informative; this project adds five derived features that
capture patterns the raw fields miss.

**The engineered features** (from
`airflow/plugins/feature_engineering.py`):

| Feature | Formula | What it captures |
|---|---|---|
| `amount_log` | `log1p(Amount)` | Reduces the heavy right-skew of Amount. Long-tail fraud loses its disproportionate influence on tree splits. |
| `amount_zscore` | `(Amount - mean) / std` | Deviation from typical spend, normalised so it's comparable across batches. In single-row serving this is 0 (see below). |
| `hour_of_day` | `(Time // 3600) % 24` | Fraud tends to cluster at odd hours. Time is seconds since the first transaction in the dataset, not a real clock, so this is an **approximation**. |
| `is_night` | `hour_of_day in [22, 23, 0..5]` | Boolean proxy for off-hours fraud. |
| `v1_v2_interaction` | `V1 * V2` | A hand-made product feature. V1 and V2 are the most predictive PCA components. Tree boosters learn interactions already, but an explicit term sometimes helps convergence. |

**The amount_zscore quirk.** In
`serving/app/models/loader.py:160`, single-row inference sets
`amount_zscore = 0.0`. There's no way to compute a meaningful
standard deviation from one row. Batch inference on line 193 does
compute a batch-level z-score. This is a real-world inconsistency:
a single-row serving request loses the zscore signal. The fix in a
real system is to compute z-score against a rolling window of recent
transactions.

**Limits.**

- No per-user features (card ID, merchant ID, prior transaction
  count), the dataset doesn't include them.
- No rolling aggregates (transactions in the last hour, mean amount
  in the last day). Those are the most predictive features in real
  fraud systems; this dataset doesn't support them.
- Time-of-day is derived from a cumulative-seconds field, not a real
  timestamp, so it's a *relative* clock, not a real one.

---

## 13. MLflow: experiments, runs, artifacts, registry

Not strictly ML theory but essential vocabulary:

- An **experiment** is a named group of runs (e.g.
  `fraud-detection-xgboost`, created in
  `train_xgboost.py:104`). You can think of it as a folder.
- A **run** is one execution of a training script. It captures
  parameters, metrics, tags, and artifacts. Each `with
  mlflow.start_run()` block is one run.
- An **artifact** is any file logged alongside a run, figures,
  pickled scalers, text files. Accessed by path (e.g.
  `scaler/scaler.pkl`).
- The **model registry** is a separate layer on top of runs.
  `mlflow.xgboost.log_model(..., registered_model_name=...)`
  registers a new **version** under a name. Aliases
  (`champion`, `challenger`) point to specific versions and can be
  moved atomically, the serving layer references the *alias*, not
  the version, so promoting a new model doesn't require a redeploy.

See [05, Training](05-training.md) for the full lifecycle.

---

## 14. Quick reference table

If you forget one of these, the [glossary](09-glossary.md) has
one-liners for everything on this page.

| Term | First appears in |
|---|---|
| Class imbalance | `train_xgboost.py:84-86` |
| SMOTE | `train_xgboost.py:118-119` |
| PR-AUC | `evaluate.py:39` |
| Threshold tuning | `evaluate.py:47-72` |
| XGBoost | `train_xgboost.py:88-97` |
| Autoencoder | `train_autoencoder.py:78-98` |
| SHAP | `serving/app/models/explainer.py:22` |
| A/B hash routing | `serving/app/models/ab_testing.py:24-26` |
| MLflow registry alias | `model_registry.py` + `train_xgboost.py:187` |
| Data drift (Evidently) | `scripts/drift_report.py` |
