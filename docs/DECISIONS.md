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
