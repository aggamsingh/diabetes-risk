# Viva Preparation

Likely examiner questions with short answers, grouped by rubric item. Numbers quoted here come from notebook outputs.

The **Kaggle notebook** section (K1-K14) covers our main notebook. The **Theory questions** section (T1-T47) at the end covers definitions, how each model works, the metrics, explainability and the reasoning behind our choices.

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

---

## Kaggle notebook (`diabetes_risk_kaggle.ipynb`)

**K1. Why did you add a second dataset?**
Pima has only 768 patients, so scores varied a lot between folds (std 0.02–0.04). The Kaggle data has about 100,000 patients, so scores are stable (std ≤ 0.01) and test scores match CV scores.

**K2. What data-quality problems did you find?**
3,854 duplicate rows (removed), 35,816 "No Info" smoking entries (kept as a category), and BMI = 27.32 used as a filled-in value for 22.5% of patients (treated as missing).

**K3. Why is the "all features" model almost perfect (ROC-AUC 0.974)?**
Because the label is largely defined by the inputs: every patient with HbA1c above 6.6 or glucose above 200 is diabetic. The model is mostly re-learning the diagnostic cut-offs.

**K4. Then what is the useful model?**
The "no blood test" model (age, BMI, gender, hypertension, heart disease, smoking). It answers *who should get a blood test?* It catches 78.7% of diabetics (test recall) with ROC-AUC 0.825.

**K5. Its precision is only 0.211. Is that acceptable?**
For a first screening step, yes: about 1 in 5 flagged patients is diabetic, and each flag only leads to a cheap blood test. Missing a diabetic is worse than an extra test.

**K6. Why didn't you predict HbA1c, as planned?**
Ridge explained only 4.2% of its variation (CV R² 0.042). HbA1c values above 6.6 occur only in diabetics, so HbA1c follows the diagnosis, not the routine features. Adding `diabetes` as an input would be leakage.

**K7. Your BMI model has R² = 0.171. Is that a failure?**
It clearly beats the baseline (RMSE 7.04 vs 7.73) and does not overfit (train 0.181, CV 0.180, test 0.171). But it shows that age, blood pressure and lab values say little about a person's weight; only age is clearly related (correlation 0.388).

**K8. Why does "No Info" smoking lower the predicted risk (odds ratio 0.67)?**
"No Info" is common among children, who rarely have diabetes, so it acts as a stand-in for being young. It is a pattern in how the data was recorded, not a cause.

**K9. Why did tuning help the Random Forest so much here but not in Pima?**
`min_samples_leaf = 10` stops trees from memorising single patients (untuned train ROC-AUC 1.000). With about 77,000 training patients, the CV scores are precise enough to show the real gain (no blood test: 0.756 → 0.831).

**K10. Why Random Forest for all features but Logistic Regression without blood tests?**
HbA1c and glucose act through cut-offs, which trees model naturally. Without them, risk rises smoothly with age and BMI, which a linear model captures just as well (0.8313 vs 0.8307), so we pick the simpler, interpretable one.

**K11. How was the threshold chosen?**
In code, by a fixed rule on training-CV probabilities: the highest threshold with recall ≥ 0.75. That gave 0.7 for all features and 0.5 for no blood test.

**K12. Why does the missed diabetic have HbA1c 8.8 but a low predicted risk?**
The no-blood-test model cannot see HbA1c. She is 45, with a normal BMI of 23.3 and no other risk factors, so on risk factors alone she looks healthy (probability 0.237). This is why screening must be followed by a blood test.

**K13. How do you handle the text columns?**
One-hot encoding inside the pipeline (`ColumnTransformer`), so the categories are learned from the training data and new categories are ignored.

**K14. What are the main limitations of the Kaggle data?**
Its source is undocumented and some patterns suggest it may be partly synthetic; the label is circular with the lab values; 22.5% of BMIs were placeholders; and children are mixed with adults.

---

## Theory questions

Short definitions and explanations of the machine-learning ideas behind our work. Use them together with the notebook-specific questions above.

### A. Machine learning basics

**T1. What is supervised learning?**

Learning a rule that maps inputs (features) to a known target, using examples where the answer is given. Classification predicts a category and regression predicts a number. Both of our tasks are supervised.

**T2. What is the difference between classification and regression?**

Classification predicts a class or a probability of a class (diabetic or not). Regression predicts a continuous number (BMI). The target decides: two possible values means classification, a continuous scale means regression.

**T3. What are overfitting and underfitting, and how do you detect them?**

Overfitting: the model memorises noise in the training data, so it scores very well on train but worse on new data. Underfitting: the model is too simple and scores poorly everywhere. We detect them by comparing train, cross-validation and test scores. Our untuned trees had train ROC-AUC near 1.0 against much lower CV scores, which is overfitting.

**T4. What is the bias-variance trade-off?**

Simple models (linear) have high bias (they miss real patterns) but low variance (stable). Flexible models (deep trees) have low bias but high variance (they change a lot with the training sample). A good model balances the two. Random forests reduce variance by averaging many trees; regularisation reduces variance by shrinking coefficients.

**T5. What is the difference between a parameter and a hyperparameter?**

Parameters are learned from data (coefficients, split thresholds). Hyperparameters are set before training and control how the model learns (alpha, C, max_depth, min_samples_leaf). We choose hyperparameters with grid search and cross-validation.

**T6. What are the roles of the train, validation and test data?**

Train data fits the model. Validation data (cross-validation folds inside the training set) compares models and tunes hyperparameters. Test data gives the final honest estimate and is used once, at the end. If the test set influences any choice, its score is no longer honest.

**T7. What is k-fold cross-validation and why use it?**

Split the training data into k parts; train on k-1 parts and score on the remaining part, k times, then average. It uses every row for validation once, gives a more reliable estimate than one split, and the standard deviation shows how stable a model is. We used k = 5.

**T8. What is regularisation?**

Adding a penalty on large coefficients so the model cannot fit noise. Ridge (L2) adds alpha times the sum of squared coefficients; Lasso (L1) adds alpha times the sum of absolute coefficients. In Logistic Regression, C is the inverse of the penalty strength (small C means a stronger penalty).

**T9. Why do we scale features?**

Models that use coefficients or distances (Logistic Regression, Ridge, KNN, SVM) are affected by units: a feature in the hundreds (glucose) would dominate one around 5 (HbA1c). Standardising gives each feature mean 0 and standard deviation 1. Trees only compare one feature to a threshold, so they don't need it.

**T10. How do class weights handle imbalance?**

With class_weight="balanced", each class gets the weight n / (2 x n_class), so mistakes on the rare diabetic class cost about 10 times more (only 8.8% are diabetic in the Kaggle data). The model then pays attention to the minority class without creating synthetic patients (as SMOTE would).

**T11. What is one-hot encoding and why not just number the categories?**

It turns a text column with K categories into K columns of 0/1. Numbering the categories (never = 1, former = 2, current = 3) would tell the model there is an order and distances between them, which is false for smoking history.

**T12. What is imputation, and why the median?**

Imputation fills missing values. We use the median of the training data because it is not pulled by extreme values. It must be learned inside the pipeline from training data only.

### B. The models: definition and how each works

**T13. What is Linear Regression (OLS)?**

It predicts a number as a weighted sum of the features, y-hat = b0 + sum of bj xj, choosing the weights that minimise the sum of squared errors. It is simple and interpretable, but unstable when features are strongly correlated.

**T14. What is Ridge Regression, and how does it differ from OLS?**

OLS plus a penalty alpha times the sum of squared coefficients. The penalty shrinks all coefficients towards zero, which makes them stable when features are correlated. It never sets a coefficient exactly to zero. We used it to predict BMI.

**T15. What is Lasso, and why can it set coefficients to exactly zero?**

OLS plus alpha times the sum of absolute coefficients. The absolute-value penalty has a constant slope, so it keeps pushing weak coefficients until they land exactly on zero, which acts as feature selection. Ridge's squared penalty has a slope that fades near zero, so it only shrinks.

**T16. What is Logistic Regression and how does it work?**

A linear classifier: it computes z = b0 + sum of bj xj, turns it into a probability with the sigmoid p = 1 / (1 + e^(-z)), and is trained by minimising log-loss. Its decision boundary is a straight line (hyperplane), and exp(b) gives an odds ratio, which makes it very interpretable.

**T17. What is a Decision Tree?**

A series of yes/no questions on single features ("HbA1c > 6.5?") that split patients into purer groups. At each step it picks the split with the largest drop in impurity, measured by Gini = 1 - sum of p_c squared. A fully grown tree memorises the training data, so we limit depth and leaf size.

**T18. What is a Random Forest and why is it better than one tree?**

Many decision trees, each trained on a random bootstrap sample of the patients and allowed to choose from a random subset of features at each split. Their predictions are combined (majority vote, probabilities averaged). Because the trees make different errors, averaging reduces variance and overfitting.

**T19. What is KNN?**

To classify a patient, find the k most similar patients (smallest Euclidean distance) and take the majority class. There is no training step, it needs scaled features and it is slow on large data. We used it in the Pima notebook.

**T20. What is Gaussian Naive Bayes?**

It uses Bayes' theorem, P(class | features) is proportional to P(class) times the product of P(feature | class), assuming the features are independent within each class and normally distributed. The independence assumption is often false (BMI and skin thickness are correlated), but the model is fast and works surprisingly well. Pima notebook only.

**T21. What is an SVM and what does the RBF kernel do?**

It finds the boundary with the widest margin between the classes. C controls how much training errors are punished. The RBF kernel K(a,b) = exp(-gamma x distance squared) lets it draw curved boundaries; small gamma gives a smoother boundary. Pima notebook only.

**T22. What is Gradient Boosting?**

Trees are added one at a time, each fitted to the errors (residuals) of the combined model so far, then added with a small learning rate: F_m = F_(m-1) + eta x h_m. It is accurate but easy to overfit and harder to interpret. Pima notebook only.

**T23. What is a baseline model and why include one?**

A model that ignores the features (for regression, always predict the training mean, which has R-squared 0). Any real model must beat it, otherwise it adds nothing. It also shows that a negative R-squared means worse than guessing the average.

**T24. Why did we use both simple and complex models?**

Simple models (Logistic Regression, Ridge) are easy to explain. Complex ones (Random Forest) test whether extra flexibility helps. Our result: it only helped when the lab values were available, so the simple model was enough for the screening task.

### C. Metrics

**T25. What is a confusion matrix?**

A table of predictions against the truth: TP (diabetics caught), FN (diabetics missed), FP (false alarms), TN (healthy patients correctly cleared). All classification metrics are computed from these four numbers.

**T26. Define accuracy, precision, recall and F1.**

Accuracy = (TP + TN) / all. Precision = TP / (TP + FP): of those flagged, how many really are diabetic. Recall = TP / (TP + FN): of all diabetics, how many we caught. F1 = 2PR / (P + R), the harmonic mean of the two.

**T27. Why is recall our main metric?**

Missing a diabetic (false negative) is worse than a false alarm, which only leads to a cheap follow-up test. Recall measures how many diabetics we catch.

**T28. What is the trade-off between precision and recall?**

Lowering the decision threshold flags more patients: recall rises, precision falls. In the Kaggle no-blood-test model, recall is 0.787 but precision only 0.211.

**T29. What are the ROC curve and AUC?**

The ROC curve plots recall (true positive rate) against the false positive rate as the threshold moves from 1 to 0. AUC, the area under it, is the probability that a random diabetic gets a higher score than a random non-diabetic: 0.5 is guessing, 1 is perfect. It does not depend on any one threshold.

**T30. What is a decision threshold and how did we choose it?**

A patient is flagged diabetic if the predicted probability is at least t. We chose t with a fixed rule on out-of-fold training probabilities: the highest t whose recall is at least 0.75. We never used the test set.

**T31. What are MAE, MSE, RMSE and R-squared?**

MAE = average absolute error. MSE = average squared error, which punishes big misses more. RMSE = square root of MSE, back in the target's units. R-squared = 1 - SS_res / SS_tot, the share of variation explained: 0 means no better than the mean, 1 is perfect, and it can be negative.

**T32. What is a residual?**

The error of one prediction, y - y-hat. Plotted against the predictions, a good model has residuals scattered around zero with no pattern.

### D. Explainability

**T33. What is the difference between global and local explanations?**

Global explanations describe which features matter overall (odds ratios, permutation importance, the SHAP beeswarm). Local explanations describe why one patient got their prediction (the SHAP waterfall).

**T34. What are odds and an odds ratio?**

Odds = p / (1 - p). In Logistic Regression, a 1-unit increase in a feature multiplies the odds by exp(b); here 1 unit is 1 standard deviation because features are scaled. An odds ratio above 1 raises the risk and below 1 lowers it.

**T35. What is permutation importance?**

Shuffle one feature's values on the test set and measure how much the score drops. A big drop means the model relies on that feature. It works for any model, but correlated features can share their importance.

**T36. What are SHAP values?**

Each feature's fair share of one prediction, based on Shapley values from game theory: its average contribution over all orders in which features could be added. For one patient the SHAP values add up exactly to prediction minus the average prediction.

**T37. Which SHAP explainer did we use, and why?**

TreeExplainer for the Random Forest (it reads the trees directly, so it is exact), LinearExplainer for Logistic Regression (exact for linear models), and KernelExplainer for the Pima SVM (works for any model but is slower and needs background data).

**T38. Why can explanations not prove cause and effect?**

They describe what the model learned from the data, which shows association. A feature can matter because it is correlated with the true cause, or because of how the data was recorded ("No Info" smoking stands in for being a child).

### E. The theory behind our choices

**T39. What is data leakage and how does our pipeline stop it?**

Test information influencing training, for example computing a median or a scaler's mean on all the data. A scikit-learn Pipeline learns every preprocessing step (imputation, scaling, encoding) from the training data only and merely applies it to new data, even inside each cross-validation fold.

**T40. Why do we stratify the split?**

It keeps the same share of diabetics (8.8%) in train and test, so recall and precision on the test set are not distorted by chance.

**T41. Why did we tune on ROC-AUC and not on recall?**

Recall alone can be maximised by flagging everyone. ROC-AUC rewards ranking patients correctly regardless of threshold; we then reach the recall we need by choosing the threshold.

**T42. Why do we compare train, CV and test scores?**

To check generalisation. Train much higher than CV means overfitting. CV close to test means the estimate is trustworthy, as in the Kaggle data with its 19,230 test patients.

**T43. What is a circular (or leaky) label, and how did we find one?**

A target that is largely defined by some of the inputs. In the Kaggle data every patient with HbA1c above 6.6 or glucose above 200 is diabetic, so a model given those features is mostly re-learning the diagnosis. That is why we also built a no-blood-test model.

**T44. What is a proxy variable? Give our example.**

A feature that stands in for something else. "No Info" smoking is common among children, so it lowers the predicted diabetes risk and BMI because it signals youth, not because missing information protects anyone.

**T45. Why did we repeat the work on a second dataset?**

The Pima data is small, so cross-validation scores varied by 0.02-0.04 and the test set held only 154 patients, which made differences between models hard to trust. The Kaggle data has about 100,000 patients, so scores vary by 0.01 or less and test scores match CV scores. What improved is how much we can trust the numbers, not the scores themselves.

**T46. Why does a bigger dataset give more stable results?**

Each fold and the test set contain many more patients, so random luck about who lands in which set averages out. That is why the Kaggle standard deviations are much smaller and why tuning showed real gains there.

**T47. Why is this a decision-support prototype and not a diagnostic tool?**

The data source is undocumented, the label is circular with the lab values, precision without blood tests is low (about 1 in 5), and the models show association, not causation. A positive screen must lead to a blood test.
