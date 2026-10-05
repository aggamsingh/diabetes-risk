# Decisions Log

Every design choice in `diabetes_risk.ipynb`, with the reason and the likely viva question.

---

### D-01: Dataset sources and offline caching  (Section A)
- Options considered: download every run; download once and save to CSV; commit the CSVs to the repo.
- Chosen: load Pima with `fetch_openml(name="diabetes", version=1)` and Progression with `load_diabetes(scaled=False)`, save both to `data/`, and read the CSV on later runs. The CSVs are also committed.
- Why: the notebook then runs offline after the first run, and everyone (including the evaluator) uses exactly the same data. We checked that a fresh-download run and a cached run give identical outputs.
- Trade-off: if OpenML ever changes the dataset, our saved copy will not pick up the change (which is what we want for reproducibility).
- Likely viva question → *"What happens if there is no internet?"* After the first run, the notebook reads `data/*.csv`, so no internet is needed.

### D-02: Rename Pima columns and encode the target as 0/1  (Section A)
- Options considered: keep OpenML's short names (`preg`, `plas`, ...); rename them to the standard names.
- Chosen: assign the clear names (`pima.columns = [...]`): `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`, and turn `class` into `Outcome` (1 = `tested_positive`, 0 = `tested_negative`).
- Why: clear names make plots and explanations readable; models need a numeric target.
- Trade-off: none of importance; the mapping is shown in the notebook.
- Likely viva question → *"How did you encode the target?"* `Outcome = 1` if the label is `tested_positive`, else 0.

### D-03: Keep the Progression column names as scikit-learn provides them  (Section A)
- Options considered: rename `s1`–`s6` to `tc`, `ldl`, `hdl`, `tch`, `ltg`, `glu`; keep `s1`–`s6`.
- Chosen: keep `s1`–`s6` and give the meaning of each in the feature dictionary.
- Why: matches the scikit-learn documentation the examiner may check; less code.
- Trade-off: plots show `s1`…`s6` instead of clinical names, so we refer to the dictionary when interpreting them.
- Likely viva question → *"What is s5?"* Most likely the log of the serum triglyceride level (`ltg` in the original paper).

### D-04: `scaled=False` for the Progression data  (Section A)
- Options considered: scikit-learn's default pre-scaled version; the raw clinical units.
- Chosen: `scaled=False` (raw units).
- Why: raw units are interpretable (BMI in kg/m², age in years). Scaling is done later inside a `Pipeline`, fitted on the training data only, to avoid leakage.
- Trade-off: we must remember to scale for distance- and coefficient-based models (Section C).
- Likely viva question → *"Why not use the already-scaled version?"* It was scaled using all 442 rows, including the future test set, which is a mild form of leakage, and it hides the clinical units.

### D-05: One global `RANDOM_STATE = 42`  (Section A)
- Options considered: no seed; different seeds in different places; one global seed.
- Chosen: `RANDOM_STATE = 42`, defined once in the setup cell and passed to every random step.
- Why: anyone who reruns the notebook gets the same splits, folds and models, and therefore the same numbers.
- Trade-off: results reflect one particular split; cross-validation (mean ± std) shows how much they would vary.
- Likely viva question → *"Would your results change with another seed?"* Slightly, and the CV standard deviation tells us by roughly how much.

### D-06: Success criteria fixed before modelling  (Section A)
- Options considered: no targets; set targets after seeing results; set targets in advance.
- Chosen: classification needs test **Recall ≥ 0.75** and **ROC-AUC ≥ 0.80**; regression must beat the mean baseline (R² = 0), aiming for **test R² ≈ 0.40 or higher**, with RMSE also reported.
- Why: fixing targets in advance stops us from moving the goalposts. Recall comes first because a missed diabetic (false negative) costs more than a false alarm. The R² target reflects that progression depends on many factors not in the data, so a perfect fit is not realistic.
- Trade-off: the thresholds are judgement calls, not clinical standards.
- Likely viva question → *"Why 0.75 recall?"* It means catching at least 3 of every 4 diabetic patients, a reasonable bar for a cheap first-stage screen followed by a confirmatory test.

### D-07: Tune classifiers on ROC-AUC, not recall  (agreed in planning; applies in Section E)
- Options considered: `GridSearchCV(scoring="recall")`; `scoring="roc_auc"`.
- Chosen: ROC-AUC for tuning; recall is still reported everywhere and raised afterwards in the threshold analysis.
- Why: recall alone can be "gamed": a model that predicts "diabetic" for everyone gets recall = 1. ROC-AUC rewards ranking patients well, independent of the threshold; the threshold is then chosen for screening.
- Trade-off: the tuned model at the default 0.5 threshold may not have the highest possible recall until the threshold is lowered.
- Likely viva question → *"Your primary metric is recall, so why not tune on it?"* (See the reason above.)

### D-08: Python 3.11 virtual environment with pinned versions  (Setup)
- Options considered: system Python 3.14; Python 3.11 in a `.venv`.
- Chosen: Python 3.11 in `.venv`, versions pinned in `requirements.txt`.
- Why: `shap` depends on `numba`, which has long-established wheels for 3.11; pinning makes the environment reproducible.
- Trade-off: users must create the venv with Python 3.11.
- Likely viva question → *"How would someone else reproduce your results?"* Create a 3.11 venv, `pip install -r requirements.txt`, then Restart & Run All.

### D-09: Save each figure when it is created  (all sections)
- Options considered: save all figures at the end (Section H); save each one where it is made.
- Chosen: save each figure to `figures/figNN_<name>.png` in the cell that creates it.
- Why: the numbering follows the notebook order, and no figure can be forgotten.
- Trade-off: none of importance.
- Likely viva question → n/a.

### D-10: Convert Pima's impossible zeros to NaN during EDA (Section B), impute later (Section C)  (Section B)
- Options considered: keep the zeros; drop rows with zeros; set them to NaN and impute.
- Chosen: replace the 0s in Glucose, BloodPressure, SkinThickness, Insulin and BMI with NaN in Section B; fill them with the **training-set median inside a Pipeline** in Section C.
- Why: the zeros are physiologically impossible (Insulin: 374 zeros = 48.7%; SkinThickness: 227 = 29.6%). Kept as 0 they distort plots, means and correlations. Dropping rows would lose about half the data. Replacing 0 with NaN learns nothing from the data, so doing it before the split causes no leakage; the median itself is learned only from the training set.
- Trade-off: almost half the Insulin values will be imputed, so Insulin-based conclusions are less reliable.
- Likely viva question → *"Why not use the zeros as they are?"* A zero insulin or BMI is impossible; it means "not recorded". Treating it as a real value would teach the model a false pattern.

### D-11: Median (not mean) for imputation  (Section B → used in C)
- Options considered: mean; median; model-based imputation.
- Chosen: median.
- Why: Insulin and SkinThickness are right-skewed (Insulin mean 79.80 vs median 30.50 before removing zeros), and the median is not pulled up by extreme values.
- Trade-off: imputing many identical values shrinks the column's spread.
- Likely viva question → *"Why median?"* It is robust to skew and outliers.

### D-12: Keep outliers  (Section B)
- Options considered: remove IQR outliers; cap them; keep them.
- Chosen: keep all rows.
- Why: the flagged values are plausible real patients (e.g. high insulin or pedigree scores), not errors, and the datasets are small (at most 29 flagged in any one column).
- Trade-off: extreme values can pull linear models; tree models are hardly affected.
- Likely viva question → *"You found outliers. Why didn't you remove them?"* An outlier is not the same as an error; removing real high-risk patients would bias the model.

### D-13: 80/20 split, stratified for Pima  (Section C)
- Options considered: 70/30; 80/20; stratified vs plain random.
- Chosen: 80/20 with `random_state=42`; `stratify=y` for Pima, plain random for Progression (continuous target).
- Why: 80% keeps enough data for training on small datasets; 20% (154 / 89 patients) is still a usable test set. Stratification kept the diabetic share at 0.349 (train) vs 0.351 (test).
- Trade-off: a single test set of 89–154 patients is noisy (Progression test mean 145.78 vs train 153.74), which is why we also rely on CV mean ± std.
- Likely viva question → *"Why stratify?"* So the small test set has the same ~35% of diabetics as the full data; otherwise recall and precision could be distorted by chance.

### D-14: All preprocessing inside a Pipeline, via two helper functions  (Section C)
- Options considered: preprocess the whole dataset once before splitting; fit preprocessing on the training set by hand; use `Pipeline`.
- Chosen: `make_scaled_pipeline(model)` (median imputer → StandardScaler → model) and `make_unscaled_pipeline(model)` (median imputer → model).
- Why: a Pipeline is refitted inside every CV fold and on the training set only, so medians, means and stds never see validation/test data. The check in C.3 shows training medians differ slightly from full-data medians (BMI 32.4 vs 32.3), and scaled test means are not exactly 0 (Insulin 0.188).
- Trade-off: slightly more code than preprocessing once; the two functions avoid repeating it 14 times.
- Likely viva question → *"What is data leakage and how did you prevent it?"* Test information leaking into training. Split first, then learn all preprocessing inside a Pipeline from the training data only.

### D-15: Which models are scaled  (Section C)
- Options considered: scale every model; scale only the models that need it.
- Chosen: scale Linear/Ridge/Lasso, Logistic Regression, KNN, SVM; do not scale Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes.
- Why: distance- and coefficient-based models are affected by feature units. Trees split on one feature at a time using thresholds, so units do not change the result. Gaussian NB fits a separate distribution per feature.
- Trade-off: none of importance (scaling the trees would be harmless, just unnecessary).
- Likely viva question → *"Why don't trees need scaling?"* A split like "Glucose > 127" picks the same patients whether Glucose is in mg/dL or standardised.

### D-16: 5-fold CV with shuffling  (Section C)
- Options considered: k = 5 or 10; shuffled or not.
- Chosen: `StratifiedKFold(5, shuffle=True, random_state=42)` for Pima, `KFold(5, shuffle=True, random_state=42)` for Progression.
- Why: with 614 / 353 training rows, 5 folds leaves about 123 / 71 rows per validation fold, which is large enough to give a stable score. 10 folds would make each validation fold very small. Shuffling with a fixed seed removes any ordering in the file while staying reproducible.
- Trade-off: 5 scores give a rough estimate of the std.
- Likely viva question → *"Why cross-validation rather than one validation split?"* One split depends on luck; averaging 5 folds gives a more reliable score plus a spread (std).

### D-17: No confidence band on the regression scatter plots  (Section B)
- Chosen: `sns.regplot(..., ci=None)`.
- Why: the band is computed by random resampling, which made the figure change on every run; without it, the plot is simpler and identical every time.

### D-18: Include a mean-only baseline  (Section D)
- Chosen: `DummyRegressor()` (always predicts the training mean) in the comparison.
- Why: our success criterion says "clearly beat the mean baseline"; it also shows that a model with negative R² (Decision Tree, CV −0.106) is worse than doing nothing.
- Likely viva question → *"What does negative R² mean?"* The model's predictions are worse than just predicting the average.

### D-19: Tune Ridge and Lasso on alpha only, scored by R²  (Section D)
- Options considered: tune the top two by CV R² (Ridge 0.481, Lasso 0.481); tune the best linear + best tree model.
- Chosen: Ridge and Lasso (the rule: best two models), grid `alpha ∈ {0.01, 0.1, 1, 10, 100}`, 5-fold CV, `scoring="r2"`.
- Why: alpha is the only hyperparameter that matters for these models. R² is the metric in our success criterion.
- Trade-off: both are linear, so tuning cannot add much; Ridge stayed at α = 1 (0.4808), Lasso moved to α = 0.1 (0.4812).
- Likely viva question → *"Why only one hyperparameter?"* Ridge and Lasso have just the penalty strength; the other settings (like the intercept) are not tuning choices.

### D-20: Final regressor = tuned Lasso (α = 0.1)  (Section D)
- Chosen: the tuned model with the higher CV R² (Lasso 0.4812 vs Ridge 0.4808).
- Why: a fixed rule (highest CV score) decided before looking at the test set. The difference is tiny (far inside the CV std of about 0.04), so Ridge or Linear Regression would be equally good.
- Result: test R² = 0.456, RMSE = 53.71, MAE = 42.81 (baseline R² −0.012), which meets the success criterion.
- Likely viva question → *"Is Lasso really better than Ridge?"* No, not meaningfully. They are within 0.0004 in CV R², so the choice barely matters.

### D-21: Show Lasso feature selection with stronger penalties  (Section D)
- Why: at the tuned α = 0.1 Lasso keeps all 10 features, so we refit with α = 1, 3, 5, 10 to show which features it drops first (s2 at α = 1; then age and s4 at α = 3).
- Likely viva question → *"Which features did Lasso remove and why?"* s2 first, because it is 0.90 correlated with s1 and so mostly redundant; then age and s4, which add little once the others are known.

### D-22: `class_weight="balanced"` where supported; no SMOTE  (Section E)
- Chosen: balanced class weights for Logistic Regression, Decision Tree, Random Forest and SVM. KNN, Gaussian NB and Gradient Boosting have no such option and are left as they are.
- Why: about 35% are diabetic; weighting each class by $n / (2 n_c)$ makes missing a diabetic cost more, with no synthetic data. The three unweighted models had the lowest CV recall (0.565–0.598).
- Trade-off: the comparison is not perfectly equal (three models lack weighting).
- Likely viva question → *"Why not SMOTE?"* Class weights solve the same problem with one argument and no fake patients; CLAUDE.md also rules SMOTE out.

### D-23: Tune Logistic Regression and SVM on ROC-AUC  (Section E)
- Chosen: the top two by CV ROC-AUC (LR 0.844, SVM 0.837; they also had the two highest recalls). LR grid: `C ∈ {0.01, 0.1, 1, 10}`. SVM grid: `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, 0.01, 0.1}`.
- Result: LR C = 0.1 → 0.8450; SVM C = 1, gamma = 0.01 → 0.8493. Final classifier = tuned SVM (higher CV ROC-AUC).
- Why ROC-AUC: see D-07; recall is raised afterwards with the threshold.
- Likely viva question → *"Is SVM really better than Logistic Regression?"* Not meaningfully: a gap of 0.004 is inside the CV std (about 0.02–0.03).

### D-24: Labels from probabilities, not `predict()`  (Section E)
- Chosen: `y_pred = (predict_proba >= t)`.
- Why: for SVM, `predict()` uses the decision score while `predict_proba` comes from a separately fitted probability model, so the two can disagree. Using probabilities everywhere means the 0.5 results and the threshold analysis follow the same rule.

### D-25: Screening threshold t = 0.30, chosen on training CV  (Section E)
- Options considered: keep 0.5; pick t on the test set; pick t from out-of-fold training probabilities.
- Chosen: the highest threshold whose training-CV recall ≥ 0.75 → t = 0.30 (CV recall 0.780, precision 0.592, 45.9% flagged).
- Why: picking t on the test set would be leakage. The highest qualifying t keeps false alarms as low as possible while meeting the target.
- Result on test: recall 0.50 → 0.796 (27 → 43 of 54 diabetics caught); false alarms 18 → 33; precision 0.566.
- Trade-off: almost half the patients are flagged for follow-up testing.
- Likely viva question → *"Why 0.30 and not 0.5?"* At 0.5 we missed half the diabetics (test recall 0.50). For screening, a missed diabetic is worse than an extra blood test.

### D-26: Hide scikit-learn's FutureWarning for `SVC(probability=True)`  (Section E)
- Why: scikit-learn 1.9 marks `probability=True` as deprecated (to be removed in 1.11) but it still works in our pinned version, and CLAUDE.md asks for it. The warning would only clutter the output.
- Trade-off: upgrading scikit-learn past 1.10 would require `CalibratedClassifierCV` instead.

### D-27: Keep the CV-selected final models even though the test set ranks others higher  (Section F)
- Observation: on the test set, untuned Lasso (0.4669) beats tuned Lasso (0.4555), and Gradient Boosting has the highest test ROC-AUC (0.831 vs 0.810 for tuned SVM).
- Chosen: keep tuned Lasso and tuned SVM, which were selected by cross-validation before the test set was used.
- Why: choosing a model *because* it scored best on the test set turns the test set into a validation set, and its score is then no longer an honest estimate. The test differences (≤ 0.02) are also within the CV noise.
- Likely viva question → *"Gradient Boosting had the best test AUC. Why didn't you pick it?"* We fixed the selection rule (best CV score) in advance; 0.02 on 154 patients is noise, and its CV ROC-AUC (0.819) was clearly lower.

### D-28: Test recall in the comparison table uses `predict()`  (Section F)
- Why: the CV recall in E.2 (from `cross_validate`) uses each model's `predict()`, so the test column uses the same rule to be comparable. The screening result with threshold 0.30 (E.5) is reported separately.

### D-29: SHAP `KernelExplainer` for the SVM, `LinearExplainer` for Logistic Regression  (Section G)
- Options considered: KernelExplainer on the tuned SVM; TreeExplainer on Random Forest instead; SHAP for LR only.
- Chosen (agreed with the team): KernelExplainer on the SVM we actually selected, plus LinearExplainer on LR for comparison.
- Details: 50 training patients as background (`shap.sample`, seed 42); explained on **imputed but unscaled** data so the plots show clinical units; output = probability. With 8 features KernelExplainer enumerates all feature coalitions, so the values are exact for this background and identical every run (about 30–40 s).
- Check: base value + sum of SHAP values = predicted probability (additivity). We checked this in a separate test script during development; it is not a cell in the notebook, but the waterfall plots show it (e.g. 0.287 → 0.932).
- Likely viva question → *"Why not TreeExplainer?"* Our best model is an SVM, not a tree; TreeExplainer only works for tree models.

### D-30: Permutation importance on the test set, scored by ROC-AUC, 20 repeats  (Section G)
- Why: computing it on the test set measures what the model relies on for *unseen* patients; ROC-AUC matches our tuning metric; 20 repeats average out the randomness of shuffling.
- Likely viva question → *"Why are some importances slightly negative?"* Shuffling a feature the model barely uses can improve the score by chance; values near 0 mean "not used".

### D-31: Choice of the three waterfall patients  (Section G)
- Rule (using the screening threshold 0.30): caught diabetic = the true positive with the highest probability; cleared healthy = the true negative with the lowest; missed diabetic = the false negative with the lowest probability (the clearest miss).
- Why: a fixed rule, not hand-picked; it shows the clearest example of each case.

### D-32: Seed NumPy before SHAP beeswarm plots  (Section G)
- Why: `shap.plots.beeswarm` shuffles the dot order using NumPy's global random generator, so the figure changed on every run. `np.random.seed(RANDOM_STATE)` before each call makes all 25 figures identical across runs (checked).

### D-33: Regression explained with Lasso coefficients, not SHAP  (Section G)
- Why: the final regressor is linear, so its coefficients on standardised features *are* its explanation (for a linear model, SHAP values are just coefficient × (value − mean)). CLAUDE.md asks for SHAP **or** coefficients here.

---

# Kaggle notebook (`diabetes_risk_kaggle.ipynb`)

### K-01: Add a second notebook on the Kaggle Diabetes Prediction Dataset  (all sections)
- Options considered: replace the Pima notebook; keep both.
- Chosen: keep Pima unchanged and build a separate notebook that mirrors it section by section.
- Why: ~100,000 patients give far more stable scores (CV std ≤ 0.01 vs 0.02–0.04 for Pima), and the project is assessed over three reviews.
- Likely viva question → *"Why two datasets?"* Pima is small but well documented; Kaggle is large and stable but of unknown origin. Together they show the same method on both.

### K-02: Drop exact duplicates; keep "No Info" as a category  (Section B)
- Found: 3,854 duplicate rows; 35,816 "No Info" smoking entries.
- Why: duplicates give some patients double weight and can leak between train and test. "No Info" is informative (it is mostly children), so it stays as its own category.

### K-03: BMI = 27.32 is a hidden missing value  (Section B)
- Found: 27.32 appears 25,495 times (21,666 after removing duplicates = 22.5%); the next most common value appears 103 times.
- Chosen: set it to NaN. Classification: the pipeline imputes the training median. Regression: those rows are dropped (BMI is the target).
- Likely viva question → *"How did you know 27.32 was fake?"* Mean and median were both exactly 27.32, and one value occurring thousands of times while all others occur about 100 times cannot be real measurements.

### K-04: Two feature sets for classification  (Sections A, E)
- Found: every patient with HbA1c above 6.6 or glucose above 200 is diabetic (B.6), so the label is largely defined by these tests.
- Chosen: "all features" and "no blood test" (age, gender, BMI, hypertension, heart disease, smoking).
- Why: the all-features model is partly circular; the no-blood-test model answers the real screening question, *who should get a blood test?*

### K-05: Regression target = BMI, not HbA1c  (Section D)
- Options considered: HbA1c (original plan); HbA1c with `diabetes` as an input; BMI.
- Found: Ridge for HbA1c reaches only CV R² 0.042; HbA1c values 6.8–9.0 occur only in diabetics, so HbA1c follows the diagnosis, not the other features. Using `diabetes` as an input would be leakage.
- Chosen (agreed with the team): show the HbA1c result briefly in D.1, then predict BMI with Ridge (test R² 0.171).
- Likely viva question → *"Why is R² only 0.17?"* Only age is clearly linked to BMI in this data (correlation 0.388); the other features say little about weight.

### K-06: One-hot encoding inside a ColumnTransformer  (Section C)
- Why: `gender` and `smoking_history` are text; one-hot turns each category into a 0/1 column. Doing it inside the pipeline means the categories are learned from the training data only; `handle_unknown="ignore"` protects against unseen categories.

### K-07: Models: LR, DT, RF (+ Ridge); tune LR and RF  (Sections D, E)
- Why: agreed scope for Review 1 (more models later). LR and RF were the best two by CV ROC-AUC in both feature sets.
- Grids: LR `C ∈ {0.01, 0.1, 1, 10}`; RF `max_depth ∈ {None, 10}`, `min_samples_leaf ∈ {1, 10}`; Ridge `alpha` from 0.01 to 10,000 (extended after the first best value, 1000, sat at the edge of the grid).
- Result: RF improved a lot (all features 0.9669 → 0.9758; no blood test 0.7558 → 0.8307); LR and Ridge did not change.

### K-08: Final classifiers  (Section E)
- All features: tuned **Random Forest** (CV ROC-AUC 0.9758 vs LR 0.9623).
- No blood test: tuned **Logistic Regression** (0.8313 vs RF 0.8307, practically a tie, so the simpler model).

### K-09: Threshold chosen in code by a fixed rule  (Section E)
- Rule: the highest threshold (from 0.9 down to 0.1) whose training-CV recall ≥ 0.75.
- Result: all features t = 0.7 (test recall 0.784, precision 0.729); no blood test t = 0.5 (test recall 0.787, precision 0.211).
- Why: no hand-picking and no test-set peeking; the grid starts above 0.5 so a *higher* threshold can be chosen when recall allows it.

### K-10: SHAP: path-dependent TreeExplainer for RF, LinearExplainer for LR, on a 500-patient sample  (Section G)
- Found: TreeExplainer with a background sample failed its own additivity check (0.979 vs 0.955); reading the trees directly (no background) is exact.
- LR SHAP values are computed on scaled data but shown in real units (inverse-transformed) so the plots are readable; they are in log-odds.
- Permutation importance uses 5 repeats (the large test set already gives stable results).

### K-11: Figures in `figures/kaggle/`  (all sections)
- Why: keeps them apart from the Pima figures, which use the same numbering.
