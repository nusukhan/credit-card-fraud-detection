Real-Time Credit Card Fraud Detection

Detecting fraudulent transactions in a severely imbalanced dataset (492 frauds in 284,807 transactions — 0.17%), with cost-based threshold selection and drift analysis.

Built for the LIVE PAKISTAN ML Internship (Week 2, Complicated Tier).

Results
Metric	Value
Best model	XGBoost (class weighting)
PR-AUC	0.8251
Precision	0.9737
Recall	0.7789
Inference latency	6 ms per transaction
Project structure
fraud-detection/
├── fraud.ipynb          # Main notebook — fully commented, runs top to bottom
├── summary.md           # Written summary (comparison, threshold, recommendation)
├── creditcard.csv       # Dataset (not included — see Setup below)
└── outputs/
    ├── fraud_model.joblib          # Saved artifact: model + scaler + threshold
    ├── 01_class_imbalance.png      # Imbalance visual (linear + log scale)
    ├── 02_imbalance_comparison.csv # Strategy comparison table
    ├── 03_strategy_comparison.png  # Strategy comparison chart
    ├── 04_precision_recall_curve.png
    ├── 05_cost_analysis.png        # Cost vs threshold curve
    ├── 05_cost_analysis.csv
    ├── 06_shap_importance.png      # SHAP global importance
    ├── 07_shap_beeswarm.png        # SHAP direction plot
    ├── 08_permutation_importance.png
    └── flagged_transactions.csv    # Real-time scoring log
Setup

1. Download the dataset

The dataset is the public ULB Credit Card Fraud set. Download creditcard.csv from Kaggle and place it in the project root:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

2. Install dependencies

bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap joblib

3. Run

Open fraud.ipynb and run all cells top to bottom (Restart + Run All).

Note: the SMOTEENN comparison cell is the slow step (~20 minutes on CPU). Everything else runs in seconds to a few minutes.

What the notebook does
Section	Step
1–7	Load data, show why accuracy is meaningless, remove 1,081 duplicates, RobustScaler, visualise imbalance
8	Stratified train/test split (resampling happens after, never before)
10	Compare 5 imbalance strategies (SMOTE, ADASYN, undersampling, class weighting, SMOTEENN)
11–12	Train Logistic Regression, Random Forest, XGBoost + Isolation Forest (unsupervised)
13	Precision-Recall curve and threshold selection
14	Cost-sensitive threshold analysis
15, 19	SHAP + permutation importance
16	Chronological drift simulation
17	Save model artifact
18	Real-time scoring loop (stretch goal)
Key decisions
RobustScaler over StandardScaler — Amount is heavily skewed; median/IQR resist outliers where mean/std do not.
Leakage-free resampling — every resampler runs inside an imblearn Pipeline, fitted on training folds only, never on the validation fold.
PR-AUC over ROC-AUC — at 0.17% fraud, ROC-AUC flatters every model. Shown twice in the results: all strategies cluster at ROC-AUC ~0.98, and Isolation Forest scores higher ROC-AUC than Random Forest despite 8× worse PR-AUC.
Cost-based threshold — the recommended 0.50 was confirmed by cost analysis, not assumed. On this model threshold tuning offers no leverage; the binding constraint is model capacity.

Full reasoning and all numbers are in summary.md.

Dependencies

Python 3.13 · pandas · numpy · scikit-learn · imbalanced-learn · xgboost · shap · matplotlib · joblib
