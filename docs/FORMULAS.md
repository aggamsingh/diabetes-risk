# Formulas

Every formula used in `diabetes_risk.ipynb`, grouped by topic. Each entry gives the formula, what the symbols mean, the intuition, and where it is used.

---

## 1. Problem framing

### Classification output
$$\hat{p} = P(y = 1 \mid \mathbf{x}), \qquad \hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge t \\ 0 & \text{otherwise} \end{cases}$$
- $\mathbf{x}$: a patient's feature vector; $y$: true class (1 = diabetic); $\hat{p}$: predicted probability of diabetes; $t$: decision threshold (0.5 by default); $\hat{y}$: predicted class.
- **Intuition:** the model gives a probability; the threshold turns it into a yes/no decision. Lowering $t$ flags more patients.
- **Used in:** Section A.5 (and the threshold analysis in Section E).

### Regression output
$$\hat{y} = f(\mathbf{x}), \qquad \hat{y} \in \mathbb{R}$$
- $f$: the function learned from data; $\hat{y}$: the predicted progression score (a real number).
- **Intuition:** regression predicts a quantity, not a category.
- **Used in:** Section A.5.

---

## 2. Statistics and EDA

### Mean
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
- $n$: number of values; $x_i$: the $i$-th value.
- **Intuition:** the "balance point" of the data; sensitive to extreme values.
- **Used in:** Section A.9 (`.describe()`).

### Median
Sort the values; the median is the middle one (or the average of the two middle ones if $n$ is even).
- **Intuition:** half the values lie below it and half above; unlike the mean, it is **robust** to outliers and skew.
- **Used in:** Section A.9 (the `50%` row of `.describe()`); later for median imputation (Section C).

### Standard deviation (sample)
$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
- $\bar{x}$: the mean; dividing by $n-1$ (not $n$) corrects for estimating the mean from the same sample (pandas' default).
- **Intuition:** the typical distance of a value from the mean.
- **Used in:** Section A.9 (`.describe()`); later for CV mean ± std.

### Quartiles
- $Q_1$ (25th percentile): 25% of values are below it. $Q_3$ (75th percentile): 75% are below it.
- **Used in:** Section A.9 (`.describe()`); later for the IQR outlier rule (Section B).

---

## 3. Metrics used in the success criteria

### Recall (sensitivity, true positive rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- $TP$: diabetic patients correctly flagged; $FN$: diabetic patients missed.
- **Intuition:** "Of all the diabetic patients, what fraction did we catch?"
- **Used in:** Section A.7 (success criteria); fully covered in Section E.

### ROC-AUC (probabilistic meaning)
$$\text{AUC} = P\big(\hat{p}(\text{random diabetic}) > \hat{p}(\text{random non-diabetic})\big)$$
- **Intuition:** the chance that the model gives a randomly chosen diabetic patient a higher risk score than a randomly chosen non-diabetic one. 0.5 = random guessing, 1.0 = perfect ranking.
- **Used in:** Section A.7; fully covered in Section E.

### Coefficient of determination
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$
- $y_i$: true value; $\hat{y}_i$: prediction; $\bar{y}$: mean of the true values; $SS_{res}$: the model's squared errors; $SS_{tot}$: the squared errors of always predicting the mean.
- **Intuition:** the fraction of the target's variance explained by the model. $R^2 = 0$ means no better than predicting the mean; $R^2 = 1$ is perfect; it can be negative for a model worse than the mean.
- **Used in:** Section A.7; fully covered in Section D.

---

*Sections still to come: preprocessing, regression models, classification models, the remaining metrics, validation, and explainability.*
