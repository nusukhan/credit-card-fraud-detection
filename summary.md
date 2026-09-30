# Written Summary

**Real-Time Credit Card Fraud Detection System**
LIVE PAKISTAN — Machine Learning Internship, Week 2

---

## 1. Data and preprocessing

The dataset contains 284,807 transactions with 492 frauds (0.173%). After removing 1,081 exact duplicate rows, 283,726 transactions remained with 473 frauds. Duplicates were removed **before** splitting: if one copy lands in training and its twin in test, the model has already seen the answer.

`Time` and `Amount` were scaled with **RobustScaler** rather than StandardScaler. `Amount` is heavily right-skewed — median €21.58 against a maximum of €25,691.16. StandardScaler centres on the mean and scales by the standard deviation, and both statistics are distorted by those extremes, compressing ordinary transactions into a narrow band near zero. RobustScaler uses the median and interquartile range, neither of which is affected by outliers. V1–V28 were left untouched, being pre-scaled PCA components.

---

## 2. Why accuracy was rejected

A model that labels every transaction genuine is correct on 284,315 of 284,807 cases: **99.83% accuracy, zero frauds caught.**

**ROC-AUC fails for a subtler version of the same reason.** All five imbalance strategies scored between 0.9793 and 0.9814 — a spread of 0.002 — while PR-AUC spread across 0.15. With roughly 283,000 true negatives, several hundred false positives barely move the false positive rate.

This was confirmed independently: **Isolation Forest scored ROC-AUC 0.930, higher than Random Forest's 0.925, while its PR-AUC (0.106) was eight times worse.** Selecting on ROC-AUC would have chosen the weaker model.

---

## 3. Imbalance handling comparison

Five strategies compared using Logistic Regression under Stratified 5-Fold cross-validation, with resampling confined to training folds via an `imblearn` Pipeline to prevent synthetic samples leaking into validation.

| Strategy | PR-AUC | Precision | Recall | Train time |
|---|---|---|---|---|
| **Class weighting** | **0.7545** | 0.0605 | 0.9179 | 13 s |
| SMOTE | 0.7519 | 0.0568 | 0.9073 | 14 s |
| SMOTEENN | 0.7506 | 0.0551 | 0.9046 | ~20 min |
| ADASYN | 0.7476 | 0.0188 | 0.9416 | 13 s |
| Random undersampling | 0.6028 | 0.0441 | 0.9126 | 1 s |

**The simplest method won.** Class weighting generates no synthetic data, discards no rows, and completed in 13 seconds. SMOTEENN required over twenty minutes and still finished third. Complexity did not translate into performance.

**Random undersampling failed for a structural reason.** Balancing by discarding majority samples meant throwing away roughly 226,000 genuine transactions, removing most of the variety in normal behaviour.

**ADASYN demonstrates why recall cannot be read alone.** Highest recall (0.9416) alongside by far the worst precision (0.0188) — fewer than two in every hundred alerts were genuine fraud.

Stratified K-Fold was mandatory throughout. With only 378 frauds in the training set, plain K-Fold could produce folds with wildly different fraud counts, making fold-to-fold scores unstable and uninterpretable.

---

## 4. Model comparison

| Model | Supervised | PR-AUC | ROC-AUC |
|---|---|---|---|
| **XGBoost** | Yes | **0.8251** | 0.9728 |
| Random Forest | Yes | 0.7978 | 0.9246 |
| Logistic Regression | Yes | 0.6719 | 0.9657 |
| Isolation Forest | No | 0.1060 | 0.9300 |

XGBoost was selected. Note that Random Forest beat Logistic Regression on PR-AUC (0.7978 vs 0.6719) while losing on ROC-AUC (0.9246 vs 0.9657) — a second direct illustration of the two metrics disagreeing.

**On the unsupervised result:** Isolation Forest's weak performance reflects scope rather than failure. Anomaly and fraud are different targets — a €9,000 legitimate purchase is a statistical outlier; a €4 fraudulent test charge is not. Its value lies in flagging novel fraud patterns for which no labelled examples exist yet, precisely where a supervised model has nothing to learn from.

---

## 5. Threshold selection

**F1-optimal threshold: 0.4782** — precision 0.9615, recall 0.7895, F1 0.8671. Against the default 0.5 (precision 0.9737, recall 0.7789, F1 0.8655) the difference is marginal, 0.0016 in F1.

### Cost-sensitive analysis

Costs were assigned asymmetrically, since the two errors are not equivalent:

| Error | Cost | Basis |
|---|---|---|
| Missed fraud | Actual transaction amount | The money is lost outright |
| False alarm | €5 | Manual review time, customer friction |

Using the actual amount rather than an average penalises missing a €2,000 fraud more heavily than a €20 one, matching real financial exposure. Total fraud value in the test set was €14,766.31 across 95 frauds.

**Result: the cost-optimal threshold is 0.500, identical to the default, with zero saving.**

This is a finding rather than a failure, and the reason is instructive. At the optimum the model misses 21 frauds (€3,792.56 in lost value) while raising only 2 false alarms (€10). **False alarms account for 0.3% of total cost**; the curve is dominated almost entirely by missed fraud.

Critically, the 21 missed frauds are missed *confidently* — their predicted probabilities sit far below any plausible threshold. No threshold adjustment recovers them, which is why the cost curve is nearly flat rather than the U-shape the method normally produces.

**The correct conclusion is that threshold tuning offers no leverage on this particular model**, because it is already well enough calibrated that false alarms are negligible. The remaining error is a model-capacity problem, not a threshold problem.

The precision-recall curve shows precision collapsing from 0.9 to 0.02 beyond recall 0.8. The cost table confirms why this cannot be exploited: even at threshold 0.01 the model raises only 51 false alarms, yet still misses 18 frauds. The remaining frauds are not hidden behind a conservative cutoff — the model simply does not detect them.

---

## 6. Explainability

SHAP (TreeExplainer, 2,000-row test subsample) ranked the top features:

| Rank | Feature | Mean abs. SHAP |
|---|---|---|
| 1 | V14 | 2.843 |
| 2 | V4 | 1.408 |
| 3 | V12 | 1.171 |
| 4 | V10 | 0.953 |
| 5 | V11 | 0.867 |

V14 dominates at roughly twice the influence of the next feature. The beeswarm plot shows direction as well as magnitude: **low** V14, V12 and V10 values push predictions towards fraud, while **high** V4 values do the same. `Amount` appears at rank 12 — transaction size carries some signal, but far less than the PCA components.

**This explanation is mathematically valid and operationally limited.** V1–V28 are anonymised PCA components; the original features were confidential. SHAP establishes that V14 carries predictive weight but cannot say what V14 represents, and a fraud analyst cannot act on "V14 was low." On non-anonymised features the same analysis would yield directly actionable insight.

---

## 7. Cross-check with permutation importance

Both methods were run, since they answer the same question by unrelated routes — SHAP decomposes individual predictions, permutation importance measures the drop in PR-AUC when a feature's relationship with the target is destroyed by shuffling.

| Rank | SHAP | Permutation (Δ PR-AUC) |
|---|---|---|
| 1 | V14 (2.843) | V14 (0.2545) |
| 2 | V4 (1.408) | V4 (0.0769) |
| 3 | V12 (1.171) | Amount (0.0583) |
| 4 | V10 (0.953) | V11 (0.0557) |
| 5 | V11 (0.867) | V3 (0.0400) |

**V14 and V4 rank first and second under both methods**, with V14 roughly three times more influential than anything else under permutation. Agreement between two unrelated attribution methods is stronger evidence than either alone.

**The methods disagree on `Amount`**, which SHAP places 12th but permutation importance places 3rd. This is not a contradiction. SHAP averages per-prediction contributions, while permutation measures aggregate performance loss. A feature that matters intensely for a small subset of transactions will show a modest average SHAP value but cause a large PR-AUC drop when broken — a plausible description of transaction amount in fraud detection.

V14 also carries a large standard deviation (±0.0571) across permutation repeats, against V4's ±0.0006. Its influence is substantial but less stable, suggesting dependence on a relatively small number of decisive transactions.

---

## 8. Drift analysis

Data was ordered by `Time` and split at the midpoint, training on the earlier period and testing on the later one.

| Period | Rows | Frauds |
|---|---|---|
| Early | 141,863 | 262 |
| Late | 141,863 | 211 |

| Split | PR-AUC |
|---|---|
| Random | 0.8251 |
| Chronological | 0.7755 |
| Difference | −0.0496 |

`Time` was dropped as a feature under this split: every test value falls outside the training range, and tree models cannot extrapolate, so leaving it in would produce degradation from that artefact rather than real drift.

**Finding: a 6% relative decline, below the threshold for meaningful degradation.** Worth noting is that the fraud count itself dropped from 262 to 211 between the two halves, so part of the change reflects a shift in the base rate rather than model decay.

The dataset spans only two days. Genuine drift operates over months, so the methodology is what transfers here rather than the specific result. In production this comparison would run on a rolling basis with retraining triggered by a defined PR-AUC floor.

---

## 9. Real-time scoring simulation (stretch goal)

Transactions were scored one at a time from the saved `.joblib` artifact, loaded fresh rather than reused from memory — verifying the artifact is genuinely self-contained.

| Metric | Value |
|---|---|
| Transactions processed | 5,000 |
| Actual frauds present | 6 |
| Transactions flagged | 4 |
| Frauds correctly caught | 4 (100% precision) |
| Mean latency | 6.06 ms |
| 95th percentile latency | 7.01 ms |
| Throughput | 154 transactions/second |

All four flagged transactions were genuine fraud, with predicted probabilities above 0.999 — the model was highly confident on each. Two frauds in the sample were missed, consistent with the recall of ~0.79 measured earlier.

A 6 ms mean latency is comfortably inside the budget for live payment authorisation, which typically allows on the order of 100 ms.

---

## 10. Recommendation

**Deploy XGBoost with class weighting (`scale_pos_weight ≈ 600`) at threshold 0.50.**

Rationale:

1. **XGBoost** — highest PR-AUC (0.8251) of all models tested
2. **Class weighting over resampling** — matched or beat every resampling method while adding no synthetic data, discarding no rows, and training in a fraction of the time
3. **Threshold 0.50** — retained not by default but because the cost analysis confirmed it as the cost-minimising point (€3,802.56). The default happens to be correct here; that was established rather than assumed

### Operational caveats

- **The €5 false-alarm cost is an assumption.** It is not a measured figure. Because false alarms contribute only 0.3% of total cost at the optimum, the conclusion is robust to moderate changes — but the curve should be recalculated with the bank's actual review cost before deployment.
- **The binding constraint is model capacity, not threshold choice.** The 21 missed frauds are missed with high confidence. Improving recall requires better features or a different model, not a different cutoff.
- **Two days of data cannot establish drift behaviour.** Chronological evaluation should run on a rolling window in production.
- **Explainability is capped by the anonymised features.** On real features, SHAP would support analyst-facing explanations rather than only model diagnostics.