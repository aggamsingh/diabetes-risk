# GlucoLens: Explainable ML for Early Prediction of Diabetes Risk

A course project (B.Tech CSE, AI/ML) that uses routine clinical measurements to:

1. **Classify** whether a patient has diabetes (screening): *Pima Indians Diabetes* dataset, 768 patients.
2. **Predict** how much the disease progresses in one year: *Diabetes Progression* dataset (Efron et al., 2004), 442 patients.

It also **explains** the predictions with odds ratios, permutation importance and SHAP.

Everything is in one notebook: [`diabetes_risk.ipynb`](diabetes_risk.ipynb).

## Results (test set)

| Task | Final model | Test results | Success criterion |
|---|---|---|---|
| Classification | SVM (RBF, C = 1, gamma = 0.01), threshold 0.30 | Recall **0.796**, precision 0.566, ROC-AUC **0.810** | Recall ≥ 0.75 ✅, ROC-AUC ≥ 0.80 ✅ |
| Regression | Lasso (alpha = 0.1) | R² **0.456**, RMSE 53.71, MAE 42.81 | R² ≥ 0.40 ✅, beats mean baseline (R² −0.012) ✅ |

**Main drivers of diabetes risk:** Glucose (odds ratio 2.78 per SD), then BMI, pregnancies, family history and age. Four explanation methods agree on this order.
**Main drivers of progression:** triglycerides (s5), BMI and blood pressure.


## Second notebook: Kaggle dataset

[`diabetes_risk_kaggle.ipynb`](diabetes_risk_kaggle.ipynb) repeats the same steps on the Kaggle **Diabetes Prediction Dataset** (~100,000 patients) for more stable results. Put `diabetes_prediction_dataset.csv` in `data/kaggle_dataset/` before running; a full run takes about 2 minutes. Figures are saved in `figures/kaggle/`.

| Task | Final model | Test results |
|---|---|---|
| Classification, all features | Random Forest (tuned), threshold 0.7 | recall 0.784, precision 0.729, ROC-AUC 0.974 |
| Classification, no blood test | Logistic Regression, threshold 0.5 | recall 0.787, precision 0.211, ROC-AUC 0.825 |
| Ridge regression on BMI | Ridge (alpha = 100) | R² 0.171, RMSE 7.04 (mean baseline 7.73) |

- The diabetes label is largely defined by HbA1c and glucose, so the "all features" model is partly circular; the **no blood test** model is the realistic screening result. Its main drivers are **age and BMI**.
- HbA1c was first planned as the regression target but cannot be predicted from the other features in this dataset (R² 0.042), so Ridge predicts BMI instead.

## Rubric map

| # | Rubric item | Notebook section |
|---|---|---|
| 1 | Problem Understanding & Familiarization | A |
| 2 | Exploratory Data Analysis & Insights | B |
| 3 | Regression and Classification | C (preprocessing), D (regression), E (classification) |
| 4 | Comparative Performance Analysis | F |
| 5 | Result Interpretation & Presentation | G (explainability), H (conclusion) |

## How to run

Requires **Python 3.11**.

```bash
py -3.11 -m venv .venv            # on macOS/Linux: python3.11 -m venv .venv
.venv\Scripts\activate            # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook diabetes_risk.ipynb
```

Then choose **Kernel → Restart & Run All**. A full run takes about 2–3 minutes (SHAP takes about 35 s of that).
The datasets are already saved in `data/`, so no internet connection is needed. If the CSVs are deleted, the first run downloads them again.
All random steps use `RANDOM_STATE = 42`, so the results are the same on every run (only the timing column changes).

## Repository layout

```
diabetes_risk.ipynb       Pima notebook (main deliverable)
diabetes_risk_kaggle.ipynb   Kaggle notebook (same steps, larger dataset)
requirements.txt          pinned library versions
data/                     pima.csv, progression.csv, kaggle_dataset/
figures/                  all figures (fig01 ... fig23) for the report and slides
docs/DECISIONS.md         every design decision, with reasons
docs/FORMULAS.md          every formula used, with symbols and intuition
docs/VIVA_PREP.md         likely viva questions with answers (both notebooks)
docs/PRESENTATION_OUTLINE.md   slide plan
```

## Limitations

The Pima data contains only adult women of one ethnic group, both datasets are small, and much of the Insulin data is missing. The models show associations, not causes. This is a course prototype for decision support, **not a diagnostic tool**.
