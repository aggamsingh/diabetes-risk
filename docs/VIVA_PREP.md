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
