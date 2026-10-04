# Viva Preparation

Likely examiner questions with short answers, grouped by rubric item. Numbers quoted here come from notebook outputs.

---

## Rubric 1: Problem Understanding & Familiarization (Section A)

**Q1. What is the difference between Type 1 and Type 2 diabetes?**
Type 1: the immune system destroys the insulin-producing cells, so the body makes little or no insulin; it usually starts young and is not caused by lifestyle. Type 2: the body becomes resistant to insulin; it is linked to obesity, inactivity and age, and is often preventable or delayable. Our adult datasets mostly reflect Type 2.

**Q2. Why does early detection matter?**
Type 2 diabetes can go unnoticed for years while high glucose damages the eyes, kidneys, nerves and heart. Finding high-risk people early allows lifestyle changes or treatment that delay the disease and its complications.

**Q3. Why is one task classification and the other regression?**
`Outcome` takes only two values (0/1), so we predict a class. The progression `target` is a continuous score (25 to 346 in our data), so we predict a number.

**Q4. Why does explainability matter in healthcare?**
Doctors need to trust and justify decisions, explanations reveal when a model relies on artefacts (like zeros meaning "missing"), and they show which modifiable factors a patient could change.

**Q5. Which features can a patient change?**
In Pima: Glucose, BloodPressure, SkinThickness, BMI and (partly) Insulin. Not modifiable: Pregnancies (history), DiabetesPedigreeFunction (genetics) and Age. In Progression: everything except age and sex.

**Q6. What is the DiabetesPedigreeFunction?**
A score summarising diabetes history in a person's relatives, weighted by how closely related they are. It is a proxy for genetic risk; it has no unit.

**Q7. Why did you set the success criteria before modelling?**
So we could not move the goalposts after seeing the results. If we set targets afterwards, we could always claim success.

**Q8. Why recall ≥ 0.75 and not accuracy?**
About 35% of Pima patients are diabetic (mean of `Outcome` = 0.35). A model that predicts "not diabetic" for everyone gets about 65% accuracy while catching nobody. For screening, missing a diabetic patient is worse than a false alarm, so recall (the share of diabetics caught) matters most.

**Q9. Pandas `.info()` showed no missing values in Pima. Is the data complete?**
No. The missing values are hidden as `0`. `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` all have a minimum of 0 in `.describe()`, which is physiologically impossible.

**Q10. Why use `scaled=False` for the Progression data?**
To keep the original clinical units for interpretation, and because scikit-learn's pre-scaled version was scaled using all rows, including the future test set (leakage). We scale inside a pipeline, on training data only.

**Q11. Who are the Pima patients, and why does that matter?**
Adult women (21 or older) of Pima Indian heritage in Arizona, a population with a very high diabetes rate. Results may not generalise to men, children or other ethnic groups.

**Q12. Why might glucose dominate the classifier?**
Diabetes is diagnosed from glucose tests, so the label is partly defined by glucose. A strong glucose effect is expected and partly circular.

**Q13. What does an R² of 0 mean?**
The model is no better than always predicting the mean progression (152.13 in our data). Our goal is to beat that clearly, aiming for R² ≈ 0.40.

---

## Rubric 2: Exploratory Data Analysis (Section B)

**Q14. How many values were missing in Pima, and where?**
Hidden as zeros: Insulin 374 (48.7%), SkinThickness 227 (29.6%), BloodPressure 35, BMI 11, Glucose 5.

**Q15. Why not use the zeros as they are?**
A zero glucose, blood pressure or BMI is impossible for a living person, so it means "not recorded". Treating it as a real value would drag averages down and teach the model a false pattern (for example, that "insulin = 0" means something).

**Q16. Why not just drop the rows with missing values?**
Insulin alone is missing for almost half the patients. Dropping those rows would throw away about half of an already small dataset.

**Q17. You replaced zeros with NaN before the train/test split. Isn't that leakage?**
No. Replacing 0 with NaN uses no statistics from the data; it applies the same fixed rule to every row. The **median** used to fill the gaps is learned inside the Pipeline from the training set only (Section C).

**Q18. Why median imputation and not mean?**
Insulin and SkinThickness are right-skewed; the median is not pulled up by the long tail of large values.

**Q19. Why is accuracy misleading for Pima?**
The classes are 65.1% non-diabetic and 34.9% diabetic. A model that always says "not diabetic" scores 65.1% accuracy while catching nobody.

**Q20. Which feature separates diabetics best, and how do you know?**
Glucose: highest correlation with Outcome (0.49), median 140 for diabetics vs 107 for non-diabetics, and its class boxes barely overlap.

**Q21. Why did you plot densities instead of counts in the Pima histograms?**
There are fewer diabetics (268 vs 500), so their count bars are always lower. Scaling each class to the same area (`stat="density"`, `common_norm=False`) lets us compare the **shapes** of the two distributions.

**Q22. What does the IQR rule do, and why did you keep the outliers?**
It flags values below $Q_1 - 1.5\,IQR$ or above $Q_3 + 1.5\,IQR$. The flagged values look like real extreme patients, not errors, and the data is small, so removing them would lose real information.

**Q23. What does a correlation of 0.49 mean? Does it mean glucose causes diabetes?**
It is a moderate positive linear relationship: higher glucose tends to go with diabetes. Correlation does not prove causation (and here diabetes is partly *defined* by glucose).

**Q24. What is the problem with s1 and s2 being correlated at 0.90?**
They carry almost the same information. In plain Linear Regression this makes their coefficients unstable (they can trade off against each other). Ridge shrinks them; Lasso may drop one.

**Q25. Why is HDL (s3) negatively correlated with progression (−0.39)?**
HDL is "good" cholesterol; higher HDL is generally protective, so it goes with less progression. This matches medical knowledge.

**Q26. Why do you need feature scaling?**
The features have very different ranges (Insulin up to 846, DiabetesPedigreeFunction below 2.5). Distance-based (KNN, SVM) and coefficient-based (linear, logistic) models would otherwise be dominated by the large-valued features.

---

## Rubric 3: Preprocessing (Section C)

**Q27. What is data leakage? Give an example from this project.**
Information from the test set influencing training. Example: computing the Insulin median or the scaler's mean/std on all 768 patients before splitting; the test patients would then have shaped the preprocessing.

**Q28. How does a Pipeline prevent leakage?**
`pipeline.fit(X_train)` learns the medians, means and stds from the training data only; `predict(X_test)` only *applies* them. In cross-validation the pipeline is refitted inside each fold, so even the validation fold is never seen during fitting.

**Q29. How can you show the pipeline really used only training data?**
In C.3 the training medians differ slightly from the full-data medians (BMI 32.4 vs 32.3), and after scaling the test-set means are not exactly 0 (Insulin 0.188), while the training means are 0.

**Q30. Why use `stratify=y`?**
To keep the same ~35% diabetics in train and test (we got 0.349 vs 0.351). Otherwise, by chance, the small test set could have too few or too many diabetics.

**Q31. Why don't tree models need scaling?**
Trees split on one feature at a time ("Glucose > 127?"). Rescaling a feature moves the threshold but selects exactly the same patients.

**Q32. Why 5 folds and not 10?**
With 353 training rows (Progression), 10 folds would leave only about 35 rows per validation fold, giving noisy scores. 5 folds is a common, balanced choice.

**Q33. Why does the Progression pipeline have an imputer if there are no missing values?**
It does nothing there; we reuse the same helper functions for both tasks to keep the code simple and consistent.

---

## Rubric 3: Regression (Section D)

**Q34. Which regression model won, and why?**
The linear models (Ridge 0.481, Lasso 0.481, Linear 0.480 CV R²) beat the trees and KNN. The relationship between the features and progression is mostly linear, and with only 353 training patients the flexible models overfit (Random Forest train R² 0.920 vs CV 0.398).

**Q35. What does R² = 0.456 actually mean?**
The model explains about 46% of the variation in progression for unseen patients. The other 54% depends on things not in the data (or is noise). Its typical error (RMSE) is about 54 points, against 73 for the mean baseline.

**Q36. Did you meet the success criterion?**
Yes: test R² = 0.456 (target ≥ 0.40), clearly above the baseline (−0.012).

**Q37. Why did the Decision Tree get a negative R²?**
An unpruned tree grows until it memorises the training data (train R² 1.000), so it fits noise and does worse than the mean on new patients (CV R² −0.106).

**Q38. Why is s1 (total cholesterol) negative, when cholesterol is bad?**
s1 and s2 are 0.90 correlated. The model gives one a big negative weight and the other a big positive one, and they partly cancel out. The coefficients of correlated features should not be read one at a time. Ridge and Lasso reduce this effect (s1 goes from −44 to −29).

**Q39. What does a coefficient of +26 for bmi mean?**
Because the features are standardised, a BMI 1 standard deviation higher (about 4.4 kg/m²) predicts about 26 points more progression, keeping all other features fixed.

**Q40. Ridge vs Lasso: what is the difference?**
Both add a penalty on coefficient size. Ridge ($\alpha\sum\beta^2$) shrinks all coefficients but never to exactly 0. Lasso ($\alpha\sum|\beta|$) can set weak ones to exactly 0, so it also selects features.

**Q41. Which features did Lasso drop?**
None at the tuned α = 0.1. With stronger penalties it drops s2 first (α = 1, redundant with s1), then age and s4 (α = 3).

**Q42. Why does the predicted-vs-actual plot show high values under-predicted?**
With R² ≈ 0.46 the model cannot explain much of the extreme cases, so it pulls its predictions towards the middle.

**Q43. Is Lasso really better than Ridge if the difference is only 0.0004?**
No. We picked it by a fixed rule (highest CV score), but the gap is far inside the CV std (about 0.04).

---

## Rubric 3: Classification (Section E)

**Q44. Which classifier won, and is the win meaningful?**
Tuned SVM (CV ROC-AUC 0.8493), just ahead of tuned Logistic Regression (0.8450). The gap is inside the CV std (about 0.02–0.03), so they are practically equal.

**Q45. Why tune on ROC-AUC if recall is the main metric?**
Recall alone can be maximised by calling everyone diabetic. ROC-AUC measures how well the model ranks patients; we then get the recall we need by lowering the threshold.

**Q46. Why is accuracy misleading here?**
The top five models all have CV accuracy of about 0.76, yet their recall ranges from 0.598 to 0.762. Accuracy hides how many diabetics are missed.

**Q47. Test recall at 0.5 was only 0.50. Why?**
The SVM's probabilities are calibrated to the real 35% diabetic rate, so few patients reach 0.5. The ranking is good (ROC-AUC 0.810), but 0.5 is the wrong cut-off for screening.

**Q48. How did you choose the threshold, and why not on the test set?**
From out-of-fold training probabilities (`cross_val_predict`): the highest t with CV recall ≥ 0.75, which gave 0.30. Choosing it on the test set would be leakage, because the test score would no longer be an honest estimate.

**Q49. What did lowering the threshold cost?**
Test recall went from 0.50 to 0.796 (16 more diabetics caught), but false alarms went from 18 to 33 and precision fell to 0.566.

**Q50. Did you meet the classification success criteria?**
Yes: test recall 0.796 (≥ 0.75) at t = 0.30 and test ROC-AUC 0.810 (≥ 0.80).

**Q51. What do C and gamma do in the SVM?**
C: how much training errors are punished (large C = fits the training data more tightly). Gamma: how far each patient's influence reaches (large gamma = a very curvy boundary). The best gamma = 0.01 was small, so a smooth boundary works best.

**Q52. Why do Random Forest and the Decision Tree have train ROC-AUC = 1.000?**
Fully grown trees memorise the training data. The gap to CV (0.820 and 0.640) shows overfitting; averaging many trees (Random Forest) reduces it but does not remove it.

**Q53. Why did KNN, Naive Bayes and Gradient Boosting have low recall?**
They have no `class_weight` option, so they are not pushed to care more about the minority diabetic class (recall 0.565–0.598).

**Q54. Why use `predict_proba >= t` instead of `predict()`?**
For SVM, `predict()` uses the decision score, which can disagree with the probability model. Using probabilities everywhere keeps the 0.5 results and the threshold analysis consistent.

---

## Rubric 4: Comparative Performance Analysis (Section F)

**Q55. Which models overfit, and how can you tell?**
The tree models: Decision Tree train R² 1.000 vs CV −0.106; Random Forest 0.920 vs 0.398; in classification both trees reach train ROC-AUC 1.000. A big train-vs-CV gap means the model memorised the training data. The linear models' train, CV and test scores are close.

**Q56. Is SVM really better than Logistic Regression?**
No, not meaningfully. Tuned CV ROC-AUC is 0.849 vs 0.845, a gap far below the CV std (0.02–0.03). On the test set they are 0.810 vs 0.810.

**Q57. Tuning made the test score slightly worse. Was tuning a mistake?**
No. The changes are tiny (SVM 0.8139 → 0.8096) and within the noise of a 154-patient test set. On small data, the defaults were already good, and tuning mainly fits noise in the CV folds. That is a valid finding.

**Q58. Gradient Boosting had the best test ROC-AUC (0.831). Why not choose it?**
We chose models by CV before looking at the test set. Picking the test winner would be leakage, and its CV ROC-AUC (0.819) was lower than SVM's and Logistic Regression's.

**Q59. Is Random Forest better *enough* than Logistic Regression to justify being harder to interpret?**
No. It is actually lower in CV ROC-AUC (0.820 vs 0.844) and overfits (train 1.000). Logistic Regression is both as accurate and far easier to explain.

**Q60. Why do the linear models win in regression?**
Progression rises roughly linearly with bmi, s5 and bp, and with only 353 training patients the flexible models fit noise. Simple models generalise better on small, mostly linear data.

**Q61. How do you read the error bars in the bar chart?**
Each bar is the CV mean; the line is ± 1 std across the 5 folds. If two models' error bars overlap a lot, we cannot say one is truly better.

**Q62. Why does the Decision Tree's ROC curve look like a few straight lines?**
A fully grown tree outputs almost only 0 or 1 as its probability, so there are very few distinct thresholds to draw the curve with.

---

## Rubric 5: Explainability & Interpretation (Section G)

**Q63. What are the top drivers of diabetes risk according to the models?**
Glucose, then BMI, Pregnancies, DiabetesPedigreeFunction (family history) and Age. All four methods (LR coefficients, permutation importance for SVM and LR, SHAP for SVM) give exactly this top five, in this order.

**Q64. Interpret the Glucose odds ratio of 2.78.**
For every 1-standard-deviation increase in glucose, the odds of diabetes multiply by about 2.78, keeping the other features fixed.

**Q65. Does a high SHAP value mean glucose *causes* diabetes?**
No. SHAP explains what the *model* uses to make its prediction; it shows association in this data, not causation. Here the link is even partly circular, because diabetes is diagnosed with glucose tests.

**Q66. What is a Shapley value, in simple words?**
Add the features one by one in every possible order and record how much each one changes the prediction when it joins. A feature's Shapley value is its average contribution. The values for one patient add up exactly to (prediction − average prediction).

**Q67. Why KernelExplainer and not TreeExplainer?**
TreeExplainer only works for tree-based models; our best classifier is an SVM. KernelExplainer works for any model by asking it for predictions with features "switched off" (replaced by background values).

**Q68. Why are the SVM's SHAP values in probability units but LR's in log-odds?**
We explained the SVM's `predict_proba` output directly. LinearExplainer explains LR's linear part, which is the log-odds $z$; the sigmoid turns it into a probability afterwards.

**Q69. Why was the missed diabetic missed?**
Her measurements look healthy: glucose 78, family-history score 0.248, age 26. Almost every feature pushed her risk down (probability 0.036). No model using only these eight features could reasonably flag her. Her diabetes may show in information we do not have.

**Q70. What surprised you in the explanations?**
(1) Blood pressure and insulin barely matter (odds ratios 1.018 and 1.016). (2) For two patients with very high insulin, the SVM *lowers* the risk, which makes no clinical sense. It comes from the curved RBF boundary in a region with very few training patients. LR shows no such reversal, so here explainability caught a weakness that accuracy alone would hide.

**Q71. Why does insulin matter so little if it was correlated (0.30) with Outcome in the EDA?**
48.7% of insulin values were missing and filled with the same median, which weakens its signal, and it is correlated with glucose (0.58), which already carries most of that information.

**Q72. Do permutation importance and SHAP measure the same thing?**
Not exactly. Permutation importance measures how much the *score* drops without a feature; SHAP measures how much a feature moves each *prediction*. Agreement between them (same top five here) increases our confidence.

**Q73. What drives diabetes progression in the regression model?**
s5 (triglycerides, +29.6 per SD), bmi (+25.8) and bp (+16.6). s1 and s2 must be read together because they are 0.90 correlated.
