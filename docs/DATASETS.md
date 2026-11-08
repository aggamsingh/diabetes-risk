# Datasets

All counts below were recounted directly from the files in `data/`.

## Overview

| | Dataset | Used in | Why this one |
|---|---|---|---|
| 1 | Kaggle Diabetes Prediction Dataset | Project 1 (Review 1) | Large and labelled, suits prediction |
| 2 | Pima Indians Diabetes Dataset | Project 2 (Unit 3, Expectation-Maximisation) | Continuous measurements, suits bell-curve (Gaussian) clustering |
| 3 | University of California Irvine Diabetes Dataset (AIM-94) | Project 3 (Unit 4, Hidden Markov Model) | Readings in time order, suits sequence modelling |

Each project asked a different question, and each question needed a different kind of data:

- Project 1: can we predict diabetes? This needs many labelled patients.
- Project 2: do patients form natural groups without labels? This needs continuous numbers.
- Project 3: how does glucose change over time? This needs time-ordered readings.

---

## 1. Kaggle Diabetes Prediction Dataset (Project 1)

**Source:** Kaggle, "Diabetes prediction dataset". File: `data/kaggle_dataset/diabetes_prediction_dataset.csv`.

**Size:**

- 100,000 rows (one row = one patient) and 9 columns (8 attributes + 1 target)
- 3,854 exact duplicate rows removed, leaving 96,146 patients
- 0 missing cells

**Attributes:**

| # | Attribute | Type | Values |
|---|---|---|---|
| 1 | gender | category | Female 58,552, Male 41,430, Other 18 |
| 2 | age | number | 0.08 to 80 years, average 41.9 |
| 3 | hypertension (high blood pressure) | yes/no | 7.5% yes |
| 4 | heart_disease | yes/no | 3.9% yes |
| 5 | smoking_history | category, 6 types | never, former, current, not current, ever, No Info |
| 6 | bmi (body mass index) | number | 10.0 to 95.7, average 27.3 |
| 7 | HbA1c_level (average blood sugar over the last 2 to 3 months) | number | 3.5 to 9.0, only 18 distinct values |
| 8 | blood_glucose_level | number | 80 to 300, only 18 distinct values |
| 9 | diabetes (target) | yes/no | 8,500 diabetic (8.5%), 91,500 not |

**Why we chose it for Project 1:**

- Large: 100,000 patients give stable classification scores.
- It has a clear yes/no diabetes label, which supervised learning needs.
- It mixes medical and lifestyle attributes (smoking, heart disease).

**What it showed us:**

- Blood sugar measures (HbA1c and glucose) were the strongest predictors, followed by body mass index and age.
- Ridge regression predicted body mass index. HbA1c was the first regression target but could not be predicted from the other attributes (cross-validated R² about 0.04).
- Data issue: body mass index = 27.32 appears far too often. It is a filled-in placeholder and was treated as missing.

**Why we did not reuse it for Project 2:**

1. HbA1c and glucose have only 18 distinct values each: they are rounded to fixed levels, not smooth measurements.
2. 4 of the 8 attributes are categories or yes/no (gender, hypertension, heart disease, smoking).
3. A Gaussian Mixture Model fits bell curves, which need continuous numbers. On repeated fixed values a bell curve collapses onto that one value and the model breaks (the same thing happened with Pregnancies in Pima).

---

## 2. Pima Indians Diabetes Dataset (Project 2)

**Source:** National Institute of Diabetes and Digestive and Kidney Diseases (United States), loaded from OpenML through scikit-learn. File: `data/pima.csv`.

**Who:** women aged 21 and over of Pima Native American heritage (Arizona), a group with one of the highest diabetes rates in the world.

**Size:**

- 768 rows (patients) and 9 columns (8 attributes + 1 target)
- 0 duplicate rows
- Missing values are hidden as 0 (see the table)

**Attributes:**

| # | Attribute | Meaning | Range | Zeros (= missing) | Used in Project 2? |
|---|---|---|---|---|---|
| 1 | Pregnancies | number of times pregnant | 0 to 17 | 111 (a real 0) | Removed: whole-number counts collapse a bell curve |
| 2 | Glucose | blood sugar 2 hours after a sugar drink | 0 to 199 | 5 | Kept |
| 3 | BloodPressure | lower (diastolic) blood pressure | 0 to 122 | 35 | Kept |
| 4 | SkinThickness | skin fold at the back of the arm (body fat) | 0 to 99 | 227 (29.6%) | Removed: too much missing |
| 5 | Insulin | insulin 2 hours after the sugar drink | 0 to 846 | 374 (48.7%) | Removed: half missing |
| 6 | BMI | body mass index | 0 to 67.1 | 11 | Kept |
| 7 | DiabetesPedigreeFunction | family history score | 0.078 to 2.42 | 0 | Kept, log-transformed (skewed) |
| 8 | Age | years | 21 to 81 | 0 | Kept, log-transformed (skewed) |
| 9 | Outcome (target) | 1 = diabetic | | 268 diabetic (34.9%), 500 not | Never used for fitting, only to check the groups afterwards |

**Final data for Project 2:** 768 patients × 5 attributes (Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age).

**Why we chose it for Project 2:**

- The rubric says Project 2 uses an existing dataset, and Pima was in the project from the start.
- Its key attributes are continuous: Glucose has 136 distinct values and the family history score 517, against 18 in Kaggle.
- The diagnosis can be hidden during clustering and used afterwards to check whether the groups make medical sense.
- It is small enough (768 patients) to check the algorithm by hand and plot every patient.

**What it showed us:**

- Without labels, patients form 3 risk groups: 10.8%, 38.0% and 56.9% diabetic.
- The drivers were glucose, body mass index and family history, the same drivers Project 1 found on Kaggle.
- 30.6% of patients are borderline between groups.
- Data shape matters: Pregnancies broke the model until it was removed.

**Why it could not be used for Project 3:** each patient is one row, measured once. There is no time order, so a Hidden Markov Model has nothing to learn about change over time.

---

## 3. University of California Irvine Diabetes Dataset, AIM-94 (Project 3)

**Source:** University of California Irvine Machine Learning Repository. Collected by Dr. Michael Kahn (Washington University, St. Louis) for the 1994 Artificial Intelligence in Medicine symposium (AIM-94). Folder: `data/uci_diabetes/`.

**Who:** 70 patients with type 1 (insulin-dependent) diabetes, recorded between March 1988 and September 1991.

**Size and structure:**

- 70 files, one per patient (`data-01` to `data-70`)
- 29,330 rows in total (one row = one recorded event) and 4 columns:

| # | Column | Example |
|---|---|---|
| 1 | Date | 04-21-1991 |
| 2 | Time | 09:09 |
| 3 | Code (type of event) | 58 |
| 4 | Value | 100 |

**Event types (20 documented codes, plus 4 undocumented ones in the files):**

| Group | Codes | Rows |
|---|---|---|
| Insulin doses (regular, NPH, UltraLente) | 33, 34, 35 | 14,401 |
| Glucose readings (unspecified, before/after breakfast, lunch, supper, snack) | 48, 57 to 64 | 13,531 |
| Low-sugar (hypoglycaemic) symptoms | 65 | 331 |
| Meals (typical, more, less) | 66, 67, 68 | 514 |
| Exercise (typical, more, less) | 69, 70, 71 | 305 |
| Special events | 72 | 94 |
| Not in the code list (ignored) | 0, 4, 36, 56 | 154 |
| **Total** | | **29,330** |

Some records came from an electronic meter with real times; others came from paper diaries with fixed slot times (breakfast 08:00, lunch 12:00, dinner 18:00, bedtime 22:00).

**Cleaning:**

1. Keep only the glucose readings: 13,531 rows.
2. Remove 13 rows with broken entries (for example the date 06-31-1991, which does not exist).
3. Remove 3 readings below 20 milligrams per decilitre (not physically realistic).
4. Result: 13,515 clean readings from 70 patients, sorted by time within each patient.
5. Keep the 68 patients with at least 50 readings.
6. Split by patient: 54 for training, 14 for testing. No patient appears in both.

**Key numbers:** median of about 120 readings per patient, median gap of about 6 hours between readings.

**Why we chose it for Project 3:**

- The rubric says Project 3 needs a new sequential dataset.
- A Hidden Markov Model learns how states change over time, so it needs readings in time order for the same person. This dataset has about 120 per patient over weeks.
- It is still diabetes, so the project story continues: the same disease, now seen over time.
- It is a well-known public benchmark, so the results can be checked by others.

**What it showed us:**

- Order matters: each reading is linked to the previous one (lag-1 autocorrelation 0.186), and shuffling the order drops this to 0.
- 3 hidden glucose states (averages 75 low, 126 in range, 206 high) that match medical ranges.
- Known medical effects found without being told: the rebound after a low (low is followed by high 56% of the time), morning highs (dawn phenomenon) and evening lows.
- On the 14 unseen patients, the Hidden Markov Model fit better than the same model without time order.

**Not used (future work):** insulin doses, meals and exercise are in the dataset. We used glucose only to keep the model clean; adding them could explain why states change.

---

## Optional: the fourth file

`data/progression.csv` is the Diabetes Progression dataset (Efron and others, 2004, built into scikit-learn).

- 442 patients × 11 columns: 10 attributes (age, sex, body mass index, blood pressure, 6 blood serum measurements s1 to s6) + 1 target (disease progression one year later).
- Used only in the earlier Pima notebook (`diabetes_risk.ipynb`) for Ridge regression. Not used in Projects 2 or 3.

---

## How the datasets build up the project

We used three datasets, each picked for what its model needs. For Project 1, prediction, we needed many labelled patients, so we used the Kaggle dataset: 100,000 patients, 8 attributes and a diabetes label. It told us blood sugar, body mass index and age are the main risk factors. For Project 2, clustering with bell curves, Kaggle did not work: its glucose and long-term blood sugar have only 18 values each and half its attributes are categories. So we used Pima: 768 patients with continuous measurements. We hid the diagnosis, and the model still found three risk groups driven by the same factors as Project 1, confirming the earlier findings with a completely different method. For Project 3, Pima had only one measurement per patient, and a Hidden Markov Model needs time order. So we used the University of California Irvine dataset: 13,515 glucose readings from 70 patients over weeks. It showed glucose moves between hidden low, in-range and high states in medically known patterns. The project goes from predicting diabetes, to finding risk groups without labels, to understanding how glucose behaves over time.
