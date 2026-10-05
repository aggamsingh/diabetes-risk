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

---

## 8. Classification models (Section E.1)

| Model | Formula | Symbols | Intuition |
|---|---|---|---|
| Logistic Regression | $\hat{p} = \sigma(z) = \frac{1}{1+e^{-z}}$, $z = \beta_0 + \sum_j \beta_j x_j$ | $\sigma$ = sigmoid, squashes any number into (0, 1) | A linear score turned into a probability |
| Log-loss (binary cross-entropy) | $L = -\frac{1}{n}\sum_i [y_i \log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$ | $y_i$ = 0/1, $\hat{p}_i$ = predicted probability | Punishes confident wrong answers heavily; LR minimises it |
| Odds ratio | $\text{OR}_j = e^{\beta_j}$ | $\beta_j$ = coefficient | A 1-unit (here 1-SD) increase in $x_j$ multiplies the odds by $e^{\beta_j}$ (used in Section G) |
| KNN | $d(a,b) = \sqrt{\sum_j (a_j - b_j)^2}$; predict the majority class of the $k$ nearest | $k = 5$ | Similar patients share outcomes |
| Gaussian Naive Bayes | $P(y \mid \mathbf{x}) \propto P(y)\prod_j P(x_j \mid y)$, each $P(x_j \mid y)$ a normal curve | $P(y)$ = class prior | Bayes' theorem + "features are independent within a class" |
| Gini impurity | $G = 1 - \sum_c p_c^2$ | $p_c$ = share of class $c$ in a node | 0 = pure node; a tree picks the split with the largest drop in $G$ |
| Entropy / information gain | $H = -\sum_c p_c \log_2 p_c$; gain = $H_\text{parent} - \sum \frac{n_\text{child}}{n} H_\text{child}$ | | Alternative split criterion (sklearn default is Gini) |
| Random Forest | $\hat{y} = \text{mode}\{T_1(x), \dots, T_B(x)\}$ | $B$ trees on bootstrap samples | Voting reduces the overfitting of single trees |
| SVM | maximise margin $\frac{2}{\lVert w\rVert}$, with errors penalised by $C$ | $w$ = weight vector, $C$ = error penalty | The widest "street" between the classes |
| RBF kernel | $K(a,b) = e^{-\gamma\lVert a-b\rVert^2}$ | $\gamma$ = reach of each point | Lets the SVM draw curved boundaries; small $\gamma$ = smoother |
| Balanced class weight | $w_c = \frac{n}{k\, n_c}$ | $n$ = samples, $k$ = 2 classes, $n_c$ = samples in class $c$ | Rare class errors cost more |

## 9. Classification metrics (Section E.2)

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN | FP (false alarm) |
| **Actual 1** | FN (missed diabetic) | TP |

$$\text{Accuracy} = \frac{TP+TN}{\text{all}} \quad \text{Precision} = \frac{TP}{TP+FP} \quad \text{Recall} = \frac{TP}{TP+FN} \quad \text{Specificity} = \frac{TN}{TN+FP} \quad F_1 = \frac{2PR}{P+R}$$
- **ROC curve:** TPR (= recall) against FPR $= \frac{FP}{FP+TN}$ (= 1 − specificity) as the threshold moves from 1 to 0. **AUC** = the area under it = the probability that a random diabetic gets a higher score than a random non-diabetic.
- **Threshold effect:** predict 1 if $\hat{p} \ge t$. Lowering $t$ → more TP and FP → recall ↑, precision ↓ (Section E.5).

---

## 10. Explainability (Section G)

### Odds ratio (Logistic Regression)
$$\log\frac{p}{1-p} = \beta_0 + \sum_j \beta_j x_j \quad\Rightarrow\quad \text{OR}_j = e^{\beta_j}$$
- $\frac{p}{1-p}$ = odds of diabetes. Increasing $x_j$ by 1 (here 1 SD) multiplies the odds by $e^{\beta_j}$. OR > 1 raises risk, OR < 1 lowers it.
- **Used in:** Section G.1 (Glucose OR = 2.78).

### Permutation importance
$$\text{importance}_j = \text{score}(X) - \frac{1}{R}\sum_{r=1}^{R}\text{score}(X \text{ with column } j \text{ shuffled, repeat } r)$$
- $R$ = 20 repeats; score = test ROC-AUC.
- **Intuition:** if breaking the link between a feature and the target hurts the score, the model was using that feature. Works for any model.
- **Used in:** Section G.2.

### Shapley value
$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\big[f(S \cup \{j\}) - f(S)\big]$$
- $F$ = all features; $S$ = a subset not containing $j$; $f(S)$ = the prediction when only the features in $S$ are known (the others are filled from the background data).
- **Intuition:** imagine adding the features one at a time in every possible order; $\phi_j$ is feature $j$'s average extra contribution when it joins. This is the only "fair" way to split the credit (it satisfies symmetry, efficiency and the "dummy" property).
- **Used in:** Section G.3–G.5.

### SHAP additivity
$$f(x) = E[f(X)] + \sum_j \phi_j(x)$$
- $E[f(X)]$ = base value (the average prediction over the background); the SHAP values add up exactly to this patient's prediction.
- **Used in:** Section G.5 waterfall plots (e.g. 0.287 + contributions = 0.932 for the caught diabetic).

---

## 11. Kaggle notebook additions

### One-hot encoding
A text column with categories $c_1, \dots, c_K$ becomes $K$ columns: $x_{c_k} = 1$ if the patient is in category $c_k$, else 0.
- **Intuition:** models need numbers, but "never = 1, former = 2, current = 3" would invent an order. One-hot gives each category its own yes/no column.
- **Coefficient meaning:** in a linear model, a one-hot coefficient is the effect of being in that category, compared with the average of the others.
- **Used in:** Kaggle notebook, Section C.3.

All other formulas (Ridge, logistic regression, Gini, random forest, metrics, CV, odds ratios, permutation importance, SHAP) are the same as above.
