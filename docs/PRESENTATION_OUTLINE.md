# Presentation Outline: GlucoLens

12 slides, about 10–12 minutes. Every number below comes from the notebook outputs.

---

### Slide 1: Title  *(Rubric 1)*
**GlucoLens: Explainable ML for Early Prediction of Diabetes Risk**. Team names, course, date.
- Two tasks: *who has diabetes?* (classification) and *how fast will it progress?* (regression).
- Goal: accurate **and** explainable predictions from routine clinical data.

### Slide 2: The problem  *(Rubric 1)*
Figure: none (or a simple icon slide).
- Type 2 diabetes can go unnoticed for years while damaging the eyes, kidneys and heart; early detection lets patients delay it.
- Known risk factors: glucose, BMI, age, family history, blood pressure.
- In healthcare a model must be *explainable*: doctors need to know *why* it flagged a patient.

### Slide 3: Data and success criteria  *(Rubric 1)*
Figure: none (table of the two datasets).
- Pima: 768 women, 8 features, target = diabetic (1/0). Progression: 442 patients, 10 features, target = progression score (25–346).
- Criteria fixed **before** modelling: recall ≥ 0.75 and ROC-AUC ≥ 0.80; regression R² ≥ 0.40 and beat the mean baseline.

### Slide 4: Hidden missing values  *(Rubric 2)*
Figure: `figures/fig01_pima_missing_zeros.png`
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible, so they mean "missing".
- Insulin is missing for 48.7% and SkinThickness for 29.6%. Fixed with median imputation inside a pipeline (no leakage).

### Slide 5: Key EDA findings  *(Rubric 2)*
Figures: `figures/fig06_pima_top_features.png` (+ `fig05_pima_correlation.png` if space allows)
- Glucose separates the classes best (correlation 0.49; median 140 vs 107).
- Classes are imbalanced (65.1% / 34.9%), so accuracy is misleading: we use recall, ROC-AUC and class weights.
- Progression: strongest links are bmi (0.59) and s5 (0.57); s1 and s2 are 0.90 correlated.

### Slide 6: Method  *(Rubric 3)*
Figure: none (a simple flow diagram: split → pipeline → 5-fold CV → tune best two → test once).
- 80/20 stratified split; imputation and scaling inside a `Pipeline` (no leakage).
- 7 regressors and 7 classifiers compared with 5-fold CV; best two per task tuned with `GridSearchCV`.

### Slide 7: Regression results  *(Rubric 3)*
Figure: `figures/fig12_reg_test_plots.png`
- Linear models win (CV R² about 0.48); final Lasso: **test R² 0.456**, RMSE 53.71 vs 73.22 for the baseline.
- Predictions are pulled towards the middle; the residuals show no pattern.

### Slide 8: Classification and the threshold  *(Rubric 3)*
Figures: `figures/fig14_clf_threshold.png` and `figures/fig15_clf_cm_screening.png`
- Tuned SVM: test ROC-AUC 0.810, but at threshold 0.5 it caught only 27 of 54 diabetics.
- The threshold was chosen on training CV (not the test set): **t = 0.30 → 43 of 54 caught (recall 0.796)**, for 15 extra false alarms.

### Slide 9: Model comparison  *(Rubric 4)*
Figures: `figures/fig16_model_comparison.png` and `figures/fig17_roc_comparison.png`
- Top models are tied within the CV noise; trees overfit (Decision Tree train R² 1.000 vs CV −0.106).
- Tuning gave tiny CV gains and no test gain. Simple models are as good as complex ones here, so prefer the interpretable one.

### Slide 10: What drives the predictions?  *(Rubric 5)*
Figures: `figures/fig19_shap_beeswarm_svm.png` (+ `fig18_permutation_importance.png`)
- Glucose dominates (odds ratio 2.78 per SD), then BMI, pregnancies, family history and age. **Four methods agree** on this order.
- High values push the risk up, which matches medical knowledge.

### Slide 11: Explaining single patients  *(Rubric 5)*
Figures: `figures/fig22_waterfall_caught_diabetic.png` and `figures/fig22_waterfall_missed_diabetic.png`
- Caught diabetic: glucose 194 alone adds +0.51 to the probability (0.287 → 0.932).
- Missed diabetic: glucose 78, age 26, everything looks healthy, so no model with these features could flag her.
- Surprise: the SVM lowers the risk for very high insulin; explainability caught a weakness that accuracy hid.

### Slide 12: Conclusion and limitations  *(Rubric 5)*
Figure: none (the results table from Section H).
- All success criteria met (ROC-AUC only just). BMI is the key *modifiable* risk factor in both tasks.
- Limitations: one population, small data, missing insulin, correlation ≠ causation, not for clinical use.
- Future work: larger and more diverse data; prefer Logistic Regression for deployment.

---

## 2-minute verbal summary

"Type 2 diabetes is often found late, so we asked whether routine clinical measurements can flag people at risk, and explain *why*. We used two public datasets: 768 Pima women for classifying diabetes, and 442 patients for predicting disease progression.

In the data we found hidden missing values: zeros where a real value is impossible, almost half of them in insulin. We filled them with the median inside a scikit-learn pipeline so no test information leaked into training. The classes were imbalanced, about 35% diabetic, so we focused on recall and ROC-AUC rather than accuracy.

We compared seven models for each task using five-fold cross-validation. For progression, simple linear models won, and our final Lasso model explains about 46% of the variation on unseen patients, well above the baseline. For diabetes, a tuned SVM and Logistic Regression were practically tied. The SVM ranked patients well, with a test ROC-AUC of 0.81, but at the default threshold it missed half the diabetics. So we lowered the threshold to 0.30, chosen on the training data, and it then caught 43 of 54 diabetics, a recall of about 0.80, at the cost of more false alarms. That is the right trade-off for screening.

To explain the models, we used odds ratios, permutation importance and SHAP. All of them agree: glucose is the strongest driver, followed by BMI, pregnancies, family history and age. This matches medical knowledge. BMI also drives progression, which makes it the key factor patients can change. Explainability also caught a weakness: the SVM treats very high insulin oddly, which the accuracy numbers did not show.

Our limitations are a narrow population, small datasets and missing data, and the results show association, not causation. This is a decision-support prototype, not a diagnostic tool."
