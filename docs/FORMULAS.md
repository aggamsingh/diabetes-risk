# Formulas

Every formula used in `diabetes_risk.ipynb`, grouped by topic. Each entry gives the formula, what the symbols mean, the intuition, and where it is used.

---

## 1. Problem framing

### Classification output
$$\hat{p} = P(y = 1 \mid \mathbf{x}), \qquad \hat{y} = \begin{cases} 1 & \text{if } \hat{p} \ge t \\ 0 & \text{otherwise} \end{cases}$$
- $\mathbf{x}$: a patient's feature vector; $y$: true class (1 = diabetic); $\hat{p}$: predicted probability of diabetes; $t$: decision threshold (0.5 by default); $\hat{y}$: predicted class.
- **Intuition:** the model gives a probability; the threshold turns it into a yes/no decision. Lowering $t$ flags more patients.
- **Used in:** Section A.2 (and the threshold analysis in Section E).

### Regression output
$$\hat{y} = f(\mathbf{x}), \qquad \hat{y} \in \mathbb{R}$$
- $f$: the function learned from data; $\hat{y}$: the predicted progression score (a real number).
- **Intuition:** regression predicts a quantity, not a category.
- **Used in:** Section A.2.

---

## 2. Statistics and EDA

### Mean
$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$
- $n$: number of values; $x_i$: the $i$-th value.
- **Intuition:** the "balance point" of the data; sensitive to extreme values.
- **Used in:** Section A.6 (`.describe()`).

### Median
Sort the values; the median is the middle one (or the average of the two middle ones if $n$ is even).
- **Intuition:** half the values lie below it and half above; unlike the mean, it is **robust** to outliers and skew.
- **Used in:** Section A.6 (the `50%` row of `.describe()`); later for median imputation (Section C).

### Standard deviation (sample)
$$s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$
- $\bar{x}$: the mean; dividing by $n-1$ (not $n$) corrects for estimating the mean from the same sample (pandas' default).
- **Intuition:** the typical distance of a value from the mean.
- **Used in:** Section A.6 (`.describe()`); later for CV mean ± std.

### Quartiles
- $Q_1$ (25th percentile): 25% of values are below it. $Q_3$ (75th percentile): 75% are below it.
- **Used in:** Section A.6 (`.describe()`); the IQR outlier rule (Section B).

### IQR and the outlier rule
$$IQR = Q_3 - Q_1 \qquad \text{outlier if } x < Q_1 - 1.5\,IQR \ \text{ or } \ x > Q_3 + 1.5\,IQR$$
- $Q_1$, $Q_3$: first and third quartiles; $IQR$: the width of the middle 50% of the data.
- **Intuition:** a value far outside the "typical" middle range is unusual. Using quartiles (not mean/std) means the rule is not itself distorted by the outliers.
- **Used in:** Sections B.4 and B.9 (`count_outliers` function, box-plot whiskers).

### Pearson correlation
$$r = \frac{\text{cov}(X, Y)}{\sigma_X \, \sigma_Y} \qquad \text{cov}(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$
- $\text{cov}$: covariance (do $X$ and $Y$ move together?); $\sigma_X, \sigma_Y$: standard deviations, which rescale the result to lie between −1 and +1.
- **Intuition:** +1 = perfect rising straight line, −1 = perfect falling line, 0 = no *linear* relation (there could still be a curved one). Correlation is not causation.
- **Used in:** Sections B.5 and B.10 (heatmaps, correlation with the target).

---

## 3. Metrics used in the success criteria

### Recall (sensitivity, true positive rate)
$$\text{Recall} = \frac{TP}{TP + FN}$$
- $TP$: diabetic patients correctly flagged; $FN$: diabetic patients missed.
- **Intuition:** "Of all the diabetic patients, what fraction did we catch?"
- **Used in:** Section A.4 (success criteria); fully covered in Section E.

### ROC-AUC (probabilistic meaning)
$$\text{AUC} = P\big(\hat{p}(\text{random diabetic}) > \hat{p}(\text{random non-diabetic})\big)$$
- **Intuition:** the chance that the model gives a randomly chosen diabetic patient a higher risk score than a randomly chosen non-diabetic one. 0.5 = random guessing, 1.0 = perfect ranking.
- **Used in:** Section A.4; fully covered in Section E.

### Coefficient of determination
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$
- $y_i$: true value; $\hat{y}_i$: prediction; $\bar{y}$: mean of the true values; $SS_{res}$: the model's squared errors; $SS_{tot}$: the squared errors of always predicting the mean.
- **Intuition:** the fraction of the target's variance explained by the model. $R^2 = 0$ means no better than predicting the mean; $R^2 = 1$ is perfect; it can be negative for a model worse than the mean.
- **Used in:** Section A.4; fully covered in Section D.

---

## 4. Preprocessing and validation

### Median imputation
Each missing value in column $j$ is replaced with $\text{median}(x_j)$ computed on the **training set only**.
- **Intuition:** fills gaps with a typical value that is not pulled by skew or outliers.
- **Used in:** Section C.2 (`SimpleImputer(strategy="median")`).

### Standardisation (z-score)
$$z = \frac{x - \mu}{\sigma}$$
- $x$: original value; $\mu$, $\sigma$: mean and standard deviation of that feature in the **training set**.
- **Intuition:** puts every feature on the same scale (mean 0, std 1), so no feature dominates because of its units.
- **Used in:** Section C.2 (`StandardScaler`), all scaled pipelines.

### Stratified split
Each class keeps the same proportion in train and test: $\frac{n_{1,\text{train}}}{n_\text{train}} \approx \frac{n_{1,\text{test}}}{n_\text{test}} \approx \frac{n_1}{n}$.
- **Used in:** Section C.1 (0.349 vs 0.351 diabetic).

### k-fold cross-validation
$$\text{CV score} = \frac{1}{k}\sum_{i=1}^{k} \text{score}_i, \qquad \text{CV std} = \sqrt{\frac{1}{k}\sum_{i=1}^{k}(\text{score}_i - \text{CV score})^2}$$
- $k$: number of folds (5); $\text{score}_i$: score when fold $i$ is held out and the model is trained on the other $k-1$ folds.
- **Intuition:** every training row is used for validation exactly once; the mean is a fairer estimate than one split, and the std shows how stable the model is.
- **Used in:** Section C.4; all model comparisons in D, E, F. (NumPy's `.std()` divides by $k$, not $k-1$.)

---

## 5. Regression models (Section D.1)

| Model | Formula | Symbols | Intuition |
|---|---|---|---|
| Baseline | $\hat{y} = \bar{y}$ | $\bar{y}$ = training mean | The "do nothing" model; $R^2 \approx 0$ |
| Linear Regression (OLS) | $\hat{y} = \beta_0 + \sum_j \beta_j x_j$, minimise $\sum_i (y_i - \hat{y}_i)^2$ | $\beta_0$ intercept, $\beta_j$ coefficient of feature $j$ | Best straight-line (hyperplane) fit |
| Ridge | minimise $\sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j \beta_j^2$ | $\alpha$ = penalty strength | Shrinks all coefficients; stabilises correlated features |
| Lasso | minimise $\sum_i (y_i - \hat{y}_i)^2 + \alpha \sum_j \lvert\beta_j\rvert$ | same | The absolute-value penalty can set coefficients to exactly 0 (feature selection) |
| KNN regression | $\hat{y} = \frac{1}{k}\sum_{i \in N_k(x)} y_i$ | $N_k(x)$ = the $k$ nearest training patients (Euclidean distance) | Similar patients have similar outcomes |
| Decision tree (regression) | choose the split minimising $\frac{n_L}{n}\text{MSE}_L + \frac{n_R}{n}\text{MSE}_R$ | $n_L, n_R$ = patients in left/right child | Each split makes the groups more uniform (lower variance); a leaf predicts its mean |
| Random forest | $\hat{y} = \frac{1}{B}\sum_{b=1}^{B} T_b(x)$ | $B$ trees, each on a bootstrap sample with random feature subsets | Averaging many noisy trees reduces variance (overfitting) |
| Gradient boosting | $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$ | $h_m$ = small tree fitted to the residuals $y - F_{m-1}(x)$, $\eta$ = learning rate | Each new tree corrects the previous errors a little |

**Why Lasso can reach exactly 0 but Ridge cannot:** the Ridge penalty $\beta^2$ has slope $2\beta$, which becomes tiny near 0, so it only pushes coefficients *towards* 0. The Lasso penalty $\lvert\beta\rvert$ has a constant slope ($\pm 1$), so it keeps pushing until a weak coefficient lands exactly on 0.

## 6. Regression metrics (Section D.2)

$$\text{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i| \qquad \text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2 \qquad \text{RMSE} = \sqrt{\text{MSE}} \qquad R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$
- MAE: the average absolute error, in target units; not sensitive to a few big errors.
- MSE: squares the errors, so big errors count much more; its units are squared.
- RMSE: back in target units; the "typical" error size, and it punishes big errors more than MAE.
- $R^2$: the share of variance explained, compared with always predicting the mean (see Section 3).
- **Residual:** $e_i = y_i - \hat{y}_i$, plotted against $\hat{y}_i$ to check for patterns (Section D.4).

## 7. Grid search (Section D.3)
For every hyperparameter value in the grid, run k-fold CV and record the mean score. Pick the value with the highest mean, then refit with it on the whole training set.
- **Intuition:** a systematic trial-and-error search that uses only training data (via CV), so the test set stays untouched.

*Sections still to come: classification models and metrics, and explainability.*
