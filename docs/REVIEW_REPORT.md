# PBL 4 Review Report: Edge Cases, Critique and Fixes

Self-review of `pbl4_boosting_bma.ipynb` and the PBL 4 deck before the presentation. Each claim was attacked the way an examiner would, re-tested, and either kept, corrected or withdrawn. All numbers come from the notebook outputs.

## Summary

| # | Claim we made | Verdict | What changed |
|---|---|---|---|
| 1 | "The first stumps rediscover the clinical cut-offs" | **Withdrawn** | The data has a hard rule; reworded (B.3) |
| 2 | "The textbook writes α = ½ ln((1 − ε)/ε)" | **Wrong, corrected** | Bishop uses the same α as scikit-learn |
| 3 | "AdaBoost is more robust to noise than a random forest" | **Too strong, corrected** | Depends on forest leaf size (C.8) |
| 4 | "Boosting beats bagging" | **Holds** | Confirmed in 5 of 5 CV folds (C.10) |
| 5 | "Without blood tests, logistic regression is as good as the ensembles" | **Holds, with a number** | Gradient boosting leads by 0.002 in CV |
| 6 | "Recall is only 0.69", with no answer for screening | **Gap, filled** | Threshold table added (E.3) |
| 7 | Results from a single 80 / 20 split | **Gap, filled** | 5-fold CV for both feature sets (C.10) |
| 8 | Noise result from one random draw | **Gap, filled** | Three more draws agree within 0.002 |
| 9 | BMI placeholder (22.6% filled in) not discussed | **Gap, filled** | Added as a limitation (B26) |
| 10 | Hand-written AdaBoost matches the library | **Holds** | Nothing to change |
| 11 | BMA collapses to one model; helps with small data | **Holds** | Nothing to change |
| 12 | Vote → probability with σ(vote) | **Holds** | Derivation re-checked against Bishop |

## 1. "Rediscovering clinical cut-offs" (withdrawn)

**Attack:** the stumps split HbA1c at 6.70 and glucose at 210, but the clinical lines are 6.5 and 200. Why claim the model found them?

**Check (notebook B.3):**

| | HbA1c | Glucose |
|---|---|---|
| Values where every patient is diabetic | 6.8, 7.0, 7.5, 8.2, 8.8, 9.0 | 220, 240, 260, 280, 300 |
| Highest value in a non-diabetic patient | 6.6 | 200 |

The rule "HbA1c ≥ 6.8 or glucose ≥ 220" covers 5,671 patients, is **100% diabetic**, and catches **66.9%** of all diabetic patients.

**Verdict:** the stumps sit exactly in the gaps of a rule that was built into the data. That rule is near the clinical lines, but the model found how the dataset was made, not medicine. The deck and notebook now say this. It also strengthens the "partly circular" argument for Section E.

## 2. The α formula (corrected)

**Attack:** "Your α has no ½. Isn't the formula α = ½ ln((1 − ε)/ε)?"

**Check:** Bishop (the course textbook) writes α = ln((1 − ε)/ε) and the combined function as f = ½ Σ αy. scikit-learn uses the same α. Some other texts move the ½ into α. Either way the sign of the vote, and so every prediction, is the same. The exponential-error bound and σ(vote) were re-derived with Bishop's form and match the outputs (4.4e-16).

**Fix:** notebook A.2, deck slide 6, FORMULAS, DECISIONS P4-04 and viva B4 all corrected. Previously they said "the textbook uses ½", which was wrong for Bishop.

## 3. Noise robustness (corrected)

**Attack:** "Your random forest is untuned, so of course it looks fragile."

**Check (C.8):** test ROC-AUC with 10% of training labels flipped:

| Model | Clean | 10% flipped | Loss |
|---|---|---|---|
| AdaBoost | 0.977 | 0.968 | 0.009 |
| Random forest, leaves ≥ 5 | 0.970 | 0.947 | 0.023 |
| Random forest, leaves ≥ 50 | 0.969 | 0.965 | 0.004 |
| Gradient boosting | 0.977 | 0.972 | 0.005 |

**Verdict:** the examiner is right. With bigger leaves the forest is *more* robust than AdaBoost. What we can defend: the exponential error does make AdaBoost chase wrong labels (36.2% of the weight lands on 5% flipped patients), but stumps cannot memorise single patients, so the test damage is small. Robustness here is about **model capacity**, not bagging versus boosting. Three more random flips gave the same numbers to within 0.002.

## 4. "Boosting beats bagging" (holds)

**Attack:** "0.977 vs 0.970 is one split. Could be luck."

**Check (C.10), stratified 5-fold CV on all 96,146 patients:**

| Model | All features | No blood tests |
|---|---|---|
| Gradient boosting | 0.978 ± 0.003 | 0.832 ± 0.005 |
| AdaBoost | 0.978 ± 0.002 | 0.827 ± 0.005 |
| Random forest | 0.972 ± 0.003 | 0.810 ± 0.005 |
| CART tree, depth 6 | 0.963 ± 0.004 | 0.820 ± 0.005 |
| Logistic regression | 0.961 ± 0.005 | 0.830 ± 0.006 |

**Verdict:** boosting beats the forest in **5 of 5 folds** (by 0.006). AdaBoost and gradient boosting are a tie (AdaBoost ahead in 3 of 5). The single-split numbers agree with the CV means to within 0.002.

## 5. Screening without blood tests (holds, made precise)

**Attack:** "So is logistic regression really as good as gradient boosting?"

**Check:** in CV, gradient boosting beats logistic regression in every fold, but only by **0.002**, which is smaller than the fold-to-fold spread. Logistic regression beats AdaBoost in every fold by 0.003.

**Verdict:** say "practically equal", not "identical". The choice of logistic regression for screening is about explainability (odds ratios) for a 0.002 cost.

## 6. Recall of 0.69 (gap filled)

**Attack:** "Your models miss 31% of diabetic patients. Useless for screening."

**Check (E.3):**

| Model | Threshold | Recall | Precision | Share flagged |
|---|---|---|---|---|
| No blood tests, logistic regression | 0.50 | 0.054 | 0.487 | 1.0% |
| | 0.08 | **0.808** | 0.198 | 36.0% |
| All features, AdaBoost | 0.50 | 0.689 | 0.978 | 6.2% |
| | 0.20 | **0.795** | 0.718 | 9.8% |

**Verdict:** 0.5 is the wrong threshold with 8.8% positives. Without blood tests, a threshold of 0.08 finds four in five diabetic patients and sends about a third of people for a blood test, which is a sensible screen.

## 7. Claims that held without change

- **Hand-written AdaBoost:** 500 of 500 stumps identical, α within 4.4e-15, identical predictions on all 19,230 test patients.
- **Exponential error = ∏ 2√(ε(1 − ε))** to 4.4e-16, always above the training error.
- **σ(vote) calibration:** log loss 0.436 → 0.083, same ROC-AUC. Follows from Bishop: the minimiser of the exponential error is ½ log-odds, and the vote is 2f.
- **BMA collapse:** on 76,916 patients the next-best model is 32 BIC units worse (posterior 1.1e-7). On 200 patients 8 models share the weight and BMA's log loss beats the single best model (0.159 vs 0.234) on average over 10 subsets.
- **No overfitting in 500 rounds:** test error 2.88% vs training 2.93%.

## 8. Remaining limitations we state openly

- One dataset (Kaggle), whose label is nearly a rule on two inputs.
- Ensembles at default settings (no tuning for the forest or gradient boosting).
- 22.6% of training BMI values are a placeholder, filled with the median, so BMI's importance is a lower bound.
- BIC is a large-sample approximation with an equal prior over models.
- Associations, not diagnoses.

## 9. Changes made

| Where | Change |
|---|---|
| Notebook | New B.3 (label rule), C.8 leaf-50 forest and repeated draws, C.10 5-fold CV, E.3 thresholds, corrected A.2, rewritten observations and limitations |
| Deck | 22 → 16 slides; slides 7, 11, 12, 13 rewritten with the corrected claims; α footnote fixed |
| Docs | DECISIONS P4-04 fixed, P4-10 to P4-13 added; FORMULAS α note fixed; VIVA B4 and B9 fixed, B21 to B26 added; README updated |

## 10. Numbering note for the final report

The course report template numbers the PBLs differently from our team sheet:

| Template | Our team sheet |
|---|---|
| PBL 1 = Units 1 and 2 | PBL 1 |
| PBL 2 = Units 3 and 4 | PBL 2 (Unit 3) + PBL 3 (Unit 4) |
| PBL 3 = Unit 5 | PBL 4 (this work) |

The report should follow the template's numbering. The deck uses the team sheet's.
