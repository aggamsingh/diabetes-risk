# CLAUDE.md: Diabetes Risk Prediction
## Explainable Machine Learning for Early Prediction of Diabetes Risk Using Clinical Data

**Repository:** https://github.com/aggamsingh/diabetes-risk.git
**Main deliverable:** one Jupyter notebook, `diabetes_risk.ipynb`, that runs top to bottom with "Restart & Run All" and no errors.

---

## 0. Who We Are and How You Should Help

We are B.Tech CSE (AI/ML) students. This is a course project graded out of 20 marks (see Section 1), followed by a viva. We must be able to explain **every line of code and every number** in the notebook.

**You are an assistant, not the author.** Follow these rules strictly:

1. **Keep it simple and readable.**
   - Use plain, beginner-readable Python: pandas, NumPy, scikit-learn, matplotlib, seaborn, and `shap` only.
   - Do not write custom classes, decorators, clever one-liners, deep helper abstractions, or "framework" code.
   - A small helper function is fine only if it removes obvious repetition, and it must be explained.
2. **Do only what is necessary.** Every section must map to a rubric item. If something does not earn marks or help us understand the problem, leave it out. If you think something extra is worth adding, **ask first** and explain why.
3. **Keep cells short.** Each code cell does one thing, ideally in under about 15 lines. Before every code cell there must be a markdown cell explaining:
   - **What** the cell does
   - **Why** we are doing it
   - The **formula(s)** involved, written in LaTeX (see Section 6)
4. **Comment for learning.** Add short comments on any non-obvious line. After every result (table or plot), add a markdown cell titled **"Observation"** that interprets the result in 2–4 plain sentences.
5. **Never fabricate results.** Every number written in markdown must come from an output actually produced in the notebook. If you have not run something, say so. Never hard-code metric values into text.
6. **Build incrementally.** Build the notebook one section at a time (Section 4). After each section:
   - Stop and summarise what was added.
   - Explain anything tricky.
   - List the viva questions you added.
   - Wait for our "go".

   We want to read and run each section ourselves before moving on.
7. **Make it reproducible.**
   - Set `RANDOM_STATE = 42` in one place at the top of the notebook and use it everywhere.
   - Pin versions in `requirements.txt`.
   - Save the datasets to `data/` so the notebook works offline after the first run.
   - The notebook must run cleanly with "Restart & Run All".
8. **Ask before deviating.** If you believe a decision in this file is wrong, tell us and propose an alternative before changing it.

---

## 1. Rubric (everything we build must map to this)

| # | Rubric item | Marks | Where it lives in the notebook |
|---|---|---|---|
| 1 | Problem Understanding & Familiarization | 4 | Section A |
| 2 | Exploratory Data Analysis & Insights | 4 | Section B |
| 3 | Unit 1 / Unit 2 Topic Implementation (Regression **and** Classification) | 4 | Sections C, D, E |
| 4 | Comparative Performance Analysis | 4 | Section F |
| 5 | Result Interpretation & Presentation | 4 | Sections G, H |

Put a short rubric map at the top of the notebook so the evaluator can see where each item is covered.

---

## 2. Problem and Datasets

Diabetes is often detected late. We will build models that use routine clinical measurements to do two things:
- **(a) Classify** whether a person has diabetes, for early screening.
- **(b) Predict** how much the disease will progress, as a continuous value.

We will also **explain** which clinical factors drive these predictions.

| Task | Dataset | How to load | Target |
|---|---|---|---|
| **Classification** | Pima Indians Diabetes: 768 patients, 8 features (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) | `sklearn.datasets.fetch_openml(name="diabetes", version=1, as_frame=True)`, then save to `data/pima.csv` | `Outcome` (1 = diabetic, 0 = not diabetic) |
| **Regression** | Diabetes Progression (Efron et al., 2004): 442 patients, 10 baseline features (age, sex, BMI, blood pressure, 6 blood serum measurements) | `sklearn.datasets.load_diabetes(as_frame=True, scaled=False)`, then save to `data/progression.csv` | Quantitative measure of disease progression one year after baseline |

**Known data issue (must be handled and explained).** In the Pima data, a value of `0` for Glucose, BloodPressure, SkinThickness, Insulin, or BMI is physiologically impossible, so it means **missing**. Convert these zeros to NaN, report how many there are per column, and impute them with the **median**, fitted on the training data only.

**Limitations to state honestly:**
- Pima contains only adult women of Pima Indian heritage, so results may not generalise.
- Both datasets are small.
- This is a decision-support prototype, not a diagnostic tool.

---

## 3. Fixed Technical Decisions (keep these unless we agree otherwise)

1. **Train/test split.** Use an 80/20 split, stratified by class for classification, with `random_state=42`. The test set is used only for final evaluation.
2. **Prevent leakage.** Fit all preprocessing (imputation, scaling) inside a scikit-learn `Pipeline`, so it is learned from training data only.
3. **Scaling.** Use `StandardScaler` for models that depend on distance or coefficients (Linear/Ridge/Lasso, Logistic Regression, KNN, SVM). Tree models do not need scaling. Explain why in the notebook.
4. **Class imbalance.** Pima is about 35% positive. Use `class_weight="balanced"` where the model supports it. Do not use SMOTE; it adds complexity and the gain is not needed.
5. **Model selection.** Use 5-fold cross-validation on the training set (`StratifiedKFold` for classification, `KFold` for regression). Report the mean ± standard deviation across folds.
6. **Tuning.** Run a small `GridSearchCV` (2–3 hyperparameters, a few values each) on the **best two models per task only**.
7. **Primary metric for screening.** Use **Recall (sensitivity)**, together with ROC-AUC. Missing a diabetic patient (a false negative) is worse than a false alarm. Justify this in the notebook.
8. **Libraries.** Use only those listed in Rule 1. No XGBoost, LightGBM, Optuna, or SMOTE. scikit-learn's own `GradientBoostingClassifier` / `GradientBoostingRegressor` is enough.

---

## 4. Notebook Structure (build in this order; pause after each section)

### Section A: Problem Understanding & Familiarization *(Rubric 1)*
- Explain what diabetes is, the difference between Type 1 and Type 2, why early detection matters, and the known clinical risk factors (glucose, BMI, age, family history, blood pressure).
- Explain why **explainability** matters in healthcare: trust, accountability, and catching model errors.
- State the two ML tasks precisely: inputs, target, and type (classification vs regression), and why each type fits its target.
- Write a feature dictionary for both datasets: name, unit, clinical meaning, and whether it is modifiable (e.g., BMI is; Age is not).
- Define the success criteria before seeing any results (e.g., "a recall of at least 0.75 on the test set with a reasonable ROC-AUC").
- Load both datasets and display `.head()`, `.shape`, `.info()`, and `.describe()`. Add an Observation cell.

### Section B: Exploratory Data Analysis & Insights *(Rubric 2)*
For **each** dataset:
- **Missing values:** For Pima, count the zeros in each column that should not be zero, with a bar chart and an explanation of why they mean "missing".
- **Target distribution:** Show the class balance (Pima) or the histogram of the progression target.
- **Feature distributions:** Plot histograms or KDE plots of each feature. For Pima, colour them by Outcome.
- **Outliers:** Show box plots and flag outliers using the IQR rule.
- **Correlation:** Plot a heatmap of Pearson correlations, including the correlation of each feature with the target.
- **Two or three focused plots for the strongest relationships:** for example, Glucose vs Outcome, or BMI vs progression.
- End the section with a **"Key EDA Insights"** markdown cell: 5–7 numbered insights, each linking a finding to clinical knowledge and to a modelling decision. For example: "Glucose separates the classes most strongly, which matches medical knowledge, so we expect it to be the top feature."

### Section C: Preprocessing
- Replace the impossible zeros with NaN (Pima).
- Do the train/test split.
- Build the `Pipeline`(s): `SimpleImputer(strategy="median")` followed by `StandardScaler` where needed.
- Explain in markdown what leakage is and how the pipeline prevents it.

### Section D: Regression *(Rubric 3; Diabetes Progression dataset)*
- **Models:**
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - KNN Regressor
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- For each model, include a short markdown cell explaining the idea and its formula (Section 6).
- **Metrics:** MAE, MSE, RMSE, and R², as 5-fold CV mean ± std on the training set.
- Tune the best two models with `GridSearchCV`.
- **Final test-set evaluation**, with:
  - A plot of **predicted vs actual** values
  - A **residual plot**
- **Interpretation:**
  - Linear/Ridge coefficients, with an explanation of the sign and size of each
  - The features that **Lasso shrinks to zero** (this shows feature selection)

### Section E: Classification *(Rubric 3; Pima dataset)*
- **Models:**
  - Logistic Regression
  - KNN
  - Gaussian Naive Bayes
  - Decision Tree
  - Random Forest
  - SVM (RBF kernel, `probability=True`)
  - Gradient Boosting
- For each model, include a short markdown cell explaining the idea and its formula.
- **Metrics:** Accuracy, Precision, Recall, F1, and ROC-AUC, as 5-fold stratified CV mean ± std.
- Tune the best two models with `GridSearchCV`, scoring on recall or ROC-AUC (justify the choice).
- **Final test-set evaluation:**
  - Confusion matrix
  - Classification report
  - ROC curve
- **Threshold analysis:**
  - Show how precision and recall change as the decision threshold moves from 0.5 down to lower values.
  - Pick a threshold suited to screening and justify it in one cell.

### Section F: Comparative Performance Analysis *(Rubric 4)*
- **Comparison tables** (one per task, sorted by the primary metric) showing:
  - The CV mean ± std
  - The test score
  - The training time per model
- **Bar charts** of the main metric across models, with error bars showing the CV standard deviation.
- **Overlay plot** of the ROC curves for all classifiers on one chart.
- **Overfitting check:** compare train vs CV vs test scores for each model, and explain any gaps.
- **Before vs after tuning** comparison for the tuned models.
- **Written analysis:**
  - Which model wins, and why: relate this to how each model works.
  - Whether the difference is meaningful given the CV standard deviation.
  - The trade-off between accuracy and interpretability. For example, is Random Forest better *enough* than Logistic Regression to justify being harder to interpret?

### Section G: Explainability & Result Interpretation *(Rubric 5)*
Apply these to the **best classifier**, plus Logistic Regression for comparison.

- **Global explanations:**
  - Logistic Regression **coefficients and odds ratios** (e^β): "each 1-SD increase in Glucose multiplies the odds of diabetes by X".
  - **Permutation importance**, calculated on the test set.
  - **SHAP**: summary (beeswarm) plot and bar plot. Use `TreeExplainer` for tree models and `LinearExplainer` for Logistic Regression.
  - Do the methods agree on the top features?
- **Local explanations:** SHAP waterfall plots for three patients: one correctly flagged diabetic, one correctly cleared, and one missed (a false negative). Explain in plain words why the model decided what it did.
- **Clinical sanity check:** Do the top drivers and the direction of their effects match medical knowledge? Flag and discuss anything surprising.
- For regression, show SHAP or coefficient importance briefly for the best regressor.

### Section H: Conclusion & Presentation *(Rubric 5)*
- A **results summary** table with the final model, test metrics, and chosen threshold for each task.
- **Answers to the questions from Section A:**
  - Did we meet the success criteria?
  - What drives diabetes risk according to the models?
- **Limitations and future work:** dataset bias, small sample size, correlation is not causation, and not for clinical use.
- Save **all key figures** to `figures/` with clear names (e.g., `fig05_roc_comparison.png`) so they can go straight into the slides.

---

## 5. Repository Layout (keep it minimal)

```
diabetes-risk/
├── CLAUDE.md
├── README.md                  # project summary, how to run, results table, rubric map
├── requirements.txt           # pinned versions
├── diabetes_risk.ipynb            # THE deliverable
├── data/                      # pima.csv, progression.csv (saved on first run)
├── figures/                   # all saved plots for the report/slides
└── docs/
    ├── DECISIONS.md
    ├── FORMULAS.md
    ├── VIVA_PREP.md
    └── PRESENTATION_OUTLINE.md
```

If folders such as `src/`, `app/`, `tests/`, or a Makefile already exist from an earlier plan, list them and ask us before deleting anything.

---

## 6. Formulas: Show Them Where They Are Used

**In the notebook:** the markdown cell before the code that uses a formula must show it in LaTeX, with one line explaining each symbol.

**In `docs/FORMULAS.md`:** collect all the formulas in one place, grouped by topic. Each entry should give the formula, the meaning of each symbol, the intuition in one or two lines, and the notebook section where it is used.

Cover at least the following:

- **Statistics and EDA:**
  - Mean, median, and standard deviation
  - IQR and the outlier rule: below Q1 − 1.5·IQR or above Q3 + 1.5·IQR
  - Pearson correlation: r = cov(X,Y) / (σ_X σ_Y)
- **Preprocessing:**
  - Median imputation
  - Standardisation: z = (x − μ) / σ
- **Regression:**
  - The linear model: ŷ = β₀ + Σβᵢxᵢ
  - The OLS objective (minimise Σ(y − ŷ)²)
  - Ridge (adds λΣβ²) and Lasso (adds λΣ|β|), and why Lasso can set coefficients to zero
  - KNN regression (average of the k nearest neighbours)
  - Decision tree split criterion for regression (reduction in variance / MSE)
  - Random forest (averaging bagged trees)
  - Gradient boosting (each new tree fits the residuals of the previous ones)
- **Classification:**
  - The sigmoid: σ(z) = 1 / (1 + e^(−z))
  - Log-loss (binary cross-entropy)
  - Odds ratios: e^β
  - KNN with Euclidean distance
  - Naive Bayes (Bayes' theorem plus the independence assumption)
  - Gini impurity and entropy / information gain
  - SVM margin maximisation and the RBF kernel
  - The idea behind `class_weight="balanced"`: weight = n_samples / (n_classes × n_c)
- **Regression metrics:** MAE, MSE, RMSE, R² = 1 − SS_res / SS_tot
- **Classification metrics:**
  - The confusion matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, Specificity, F1
  - The ROC curve (TPR vs FPR) and AUC
  - The effect of the decision threshold
- **Validation:** k-fold cross-validation (mean ± std); how grid search works
- **Explainability:**
  - Permutation importance (the drop in score when a feature is shuffled)
  - The Shapley value formula and its intuition (fair credit for each feature, averaged over all feature orderings)
  - SHAP additivity: prediction = base value + Σ SHAP values

---

## 7. Viva Documentation (update after every section)

- **`docs/DECISIONS.md`:** add one entry for every choice made: median imputation, which models get scaled, `class_weight` vs SMOTE, metric priorities, the choice of k for cross-validation, threshold choice, which models to tune, and so on. Use this format:
  ```
  ### D-XX: <Decision>  (Section X)
  - Options considered:
  - Chosen:
  - Why (with evidence from the notebook if empirical):
  - Trade-off:
  - Likely viva question → short answer
  ```
- **`docs/VIVA_PREP.md`:** about 50 likely examiner questions with short, clear answers, grouped by rubric item. Include tough ones, for example:
  - "Why is accuracy misleading here?"
  - "Why not use the zeros as-is?"
  - "Does a high SHAP value mean glucose *causes* diabetes?"
  - "Why did model X beat model Y?"
  - "What does R² = 0.45 actually mean?"
- **`docs/PRESENTATION_OUTLINE.md`:** about 10–12 slides, mapped to the rubric, with the figure from `figures/` to use on each slide and 2–3 talking points per slide. Finish with a 2-minute verbal summary of the project.

---

## 8. Git Rules

- **Author identity:** every commit must be authored by **Aggam Singh Arora <aggamsingh1@gmail.com>**. Do not use a Claude author identity, and do not add `Co-Authored-By: Claude` or `Claude-Session` lines to commit messages.
- **Committing and pushing:** commit after each completed notebook section with a clear message (e.g., `Add Section B: EDA for both datasets`), and push to `origin main`.
- **Before committing the notebook:** clear any huge outputs if needed, but keep plots and tables visible. The evaluator should see the results without having to rerun the notebook.
- **Never commit:** `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.

---

## 9. End-of-Section Checklist (report this to us every time)

1. What was added (cells and outputs)
2. Key results, with numbers taken from the outputs
3. Decisions logged in `DECISIONS.md`
4. Formulas added to `FORMULAS.md`
5. New viva questions
6. Anything you are unsure about, or anything we should verify
7. The next section, waiting for our "go"

---

## 10. Second Notebook: Kaggle Dataset (added for the 3-review schedule)

`diabetes_risk_kaggle.ipynb` repeats the Pima notebook on the Kaggle **Diabetes Prediction Dataset** (~100k patients), which gives more stable scores. Rules:

- **Mirror the Pima notebook** (`diabetes_risk.ipynb`) exactly: same Sections A–H, writing style, cell layout, helper functions and checklist. Only the dataset (and what follows from it) changes. Do not modify the Pima notebook.
- **Data:** `data/diabetes_prediction_dataset.csv`, downloaded manually from Kaggle (no kagglehub). Exact duplicate rows are dropped; `"No Info"` in `smoking_history` is kept as a category.
- **Classification** (`diabetes`): Logistic Regression, Decision Tree, Random Forest, each on **two feature sets**: *all features* and *no blood test* (without `HbA1c_level` and `blood_glucose_level`), because the label is defined by those tests.
- **Regression:** **Ridge** on `HbA1c_level`, compared with a mean baseline. `diabetes` must never be a regression feature (leakage).
- **Preprocessing:** text columns (`gender`, `smoking_history`) are one-hot encoded inside the pipeline (`ColumnTransformer`).
- **Screening threshold:** chosen in code by a fixed rule on training-CV probabilities (highest threshold with recall ≥ 0.75).
- **Figures:** `figures/kaggle/figNN_*.png`.
- More models may be added in later reviews.

**Start by reading this file, then confirm your understanding and propose the plan for Section A.**