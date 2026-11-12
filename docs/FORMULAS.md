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

---

## 12. PBL 2: Gaussian Mixture Model and EM (`pbl2_alternate_em.ipynb`)

### Mixture density
$$p(x) = \sum_k \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
$\pi_k$ = share of group $k$, $\mu_k$ = group centre, $\Sigma_k$ = group spread. **Used in:** Section A.2.

### E-step (responsibilities)
$$\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
- **Intuition:** the probability that patient $i$ belongs to group $k$, given the current parameters.
- **Used in:** Section A.3, checked by hand in C.4.

### M-step
$$N_k = \sum_i \gamma_{ik} \qquad \pi_k = \frac{N_k}{N} \qquad \mu_k = \frac{\sum_i \gamma_{ik} x_i}{N_k} \qquad \Sigma_k = \frac{\sum_i \gamma_{ik}(x_i - \mu_k)(x_i - \mu_k)^\top}{N_k}$$
- **Intuition:** ordinary averages, weighted by the responsibilities.

### Log-likelihood
$$\log L = \sum_i \log \sum_k \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)$$
- **Intuition:** an E-step plus an M-step never decreases it, and EM stops at a local maximum.
- **Used in:** Section C.2.

### Hard EM = K-means
$$z_i = \arg\min_k \lVert x_i - \mu_k \rVert^2 \qquad \mu_k = \text{mean of the points with } z_i = k$$
**Used in:** Section C.5.

### BIC and AIC
$$\text{BIC} = -2\log L + p \log N \qquad \text{AIC} = -2\log L + 2p$$
Lower is better; $p$ = number of free parameters. **Used in:** Section C.1.

### Log transform
$x' = \log(1 + x)$: shortens a long right tail so the feature looks more Gaussian. **Used in:** Section B.3.

### Silhouette score
$$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$$
$a_i$ = mean distance to the patient's own group, $b_i$ = mean distance to the nearest other group. **Used in:** Section D.1.

### Adjusted Rand Index
Agreement between two groupings, corrected for chance: 1 = identical, 0 = random. **Used in:** Sections D.1 and D.5.

### PCA (for plotting only)
Projects the data onto the directions of largest variance (the eigenvectors of the covariance matrix). **Used in:** Sections C.2 and D.3.

---

## 13. PBL 3: Hidden Markov Model (`pbl3_hmm.ipynb`)

### Markov property and transition matrix
$$P(z_t \mid z_{1:t-1}) = P(z_t \mid z_{t-1}) \qquad a_{ij} = P(z_t = j \mid z_{t-1} = i)$$
**Used in:** Section A.1.

### HMM parameters
$\lambda = (\pi, A, B)$, with Gaussian emissions $b_j(x) = \mathcal{N}(x \mid \mu_j, \sigma_j^2)$. **Used in:** Section A.2.

### Forward algorithm
$$\alpha_1(j) = \pi_j b_j(x_1) \qquad \alpha_t(j) = \Big[\sum_i \alpha_{t-1}(i) a_{ij}\Big] b_j(x_t) \qquad P(x_{1:T}) = \sum_j \alpha_T(j)$$
**Used in:** Section A.3, hand-coded in C.4, reused in D.3.

### Backward algorithm
$$\beta_T(i) = 1 \qquad \beta_t(i) = \sum_j a_{ij}\, b_j(x_{t+1})\, \beta_{t+1}(j)$$

### Forward-backward posterior
$$\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{\sum_j \alpha_t(j)\beta_t(j)}$$
**Used in:** Section D.2 (`predict_proba`).

### Viterbi algorithm
$$\delta_t(j) = \max_i\big[\delta_{t-1}(i)\, a_{ij}\big]\, b_j(x_t)$$
Plus back-pointers to recover the best path. **Used in:** Sections C.4 and D.2.

### Baum-Welch (EM for HMMs)
- **E-step:** compute $\gamma_t(i)$ and $\xi_t(i,j) = P(z_t = i, z_{t+1} = j \mid x)$.
- **M-step:** $\pi_i = \gamma_1(i)$, $a_{ij} = \sum_t \xi_t(i,j) / \sum_t \gamma_t(i)$, and $\mu_j$, $\sigma_j^2$ = the $\gamma$-weighted mean and variance.
- **Used in:** Sections A.3 and C.2.

### Log-sum-exp trick
$$\log \sum_i e^{a_i} = m + \log \sum_i e^{a_i - m}, \qquad m = \max_i a_i$$
Prevents underflow on long sequences (`logsumexp`). **Used in:** Section C.4.

### Expected time in a state
$E[\text{duration}_i] = 1/(1 - a_{ii})$ (mean of a geometric distribution). **Used in:** Section C.3.

### Lag-1 autocorrelation
$r_1 = \text{corr}(x_t, x_{t-1})$, computed within each patient. **Used in:** Section B.3.

### One-step forecast
$$P(z_{t+1} = j \mid x_{1:t}) = \sum_i P(z_t = i \mid x_{1:t})\, a_{ij} \qquad \hat{x}_{t+1} = \sum_j P(z_{t+1} = j \mid x_{1:t})\, \mu_j$$
**Used in:** Section D.3, scored with MAE $= \frac{1}{n}\sum |y - \hat{y}|$.

### k-step transitions
$$P(z_{t+k} = j \mid z_t = i) = (A^k)_{ij}$$
**Used in:** Section C.3b.

### Stationary distribution
$$\pi^* = \pi^* A \qquad \sum_i \pi^*_i = 1$$
- **Intuition:** the long-run share of time in each state. It is the left eigenvector of $A$ with eigenvalue 1.
- **Used in:** Section C.3b.

## 14. PBL 4: Boosting and Bayesian Model Averaging (`pbl4_boosting_bma.ipynb`)

### AdaBoost
Labels $t_n \in \{-1, +1\}$, weak classifier $y_m(x) \in \{-1, +1\}$, start with $w_n^{(1)} = 1/N$.
$$\epsilon_m = \frac{\sum_n w_n^{(m)} I(y_m(x_n) \ne t_n)}{\sum_n w_n^{(m)}} \qquad \alpha_m = \ln\frac{1-\epsilon_m}{\epsilon_m} \qquad w_n^{(m+1)} = w_n^{(m)} \exp\big(\alpha_m I(y_m(x_n) \ne t_n)\big)$$
$$Y_M(x) = \operatorname{sign}\Big(\sum_m \alpha_m y_m(x)\Big)$$
- **Intuition:** a smaller error gives a bigger say; misclassified points get heavier, so the next stump looks at them.
- The textbook uses $\tfrac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}$; scikit-learn drops the $\tfrac{1}{2}$, which doubles the vote but not its sign.
- **Used in:** Sections A.2, C.2, C.3.

### Exponential error and the training bound
$$E = \frac{1}{N}\sum_n \exp\big(-t_n f_m(x_n)\big), \qquad f_m = \tfrac{1}{2}\sum_{l \le m} \alpha_l y_l \qquad E = \prod_{l \le m} 2\sqrt{\epsilon_l(1-\epsilon_l)} \;\ge\; \text{training error rate}$$
- **Intuition:** a misclassified point ($t f < 0$) costs exponentially more, which is why boosting chases hard points and is sensitive to wrong labels.
- **Used in:** Sections A.3, C.3, C.8.

### Boosted vote as a probability
$$\arg\min_f \mathbb{E}[e^{-t f(x)}] = \tfrac{1}{2}\ln\frac{p(t=1 \mid x)}{p(t=-1 \mid x)} \quad\Rightarrow\quad p(t = 1 \mid x) = \sigma\Big(\sum_m \alpha_m y_m(x)\Big)$$
**Used in:** Section C.6 (log loss 0.436 to 0.083).

### Other error functions (for comparison)
Cross-entropy $\ln(1 + e^{-ty})$, hinge $\max(0, 1 - ty)$, misclassification $I(ty < 0)$. **Used in:** Section C.6 figure.

### Average precision
$\text{AP} = \sum_k (R_k - R_{k-1}) P_k$: area under the precision-recall curve. Better than ROC-AUC when positives are rare (8.8% here). **Used in:** Section C.

### Brier score
$\frac{1}{N}\sum_n (p_n - y_n)^2$: mean squared error of the predicted probabilities. **Used in:** Section C.6.

### Bayesian model averaging
$$p(t \mid x, D) = \sum_k p(t \mid x, M_k, D)\, p(M_k \mid D) \qquad p(M_k \mid D) \approx \frac{\exp(-\tfrac{1}{2}\mathrm{BIC}_k)}{\sum_j \exp(-\tfrac{1}{2}\mathrm{BIC}_j)}$$
with $\mathrm{BIC}_k = -2\ln L_k + p_k \ln N$ and an equal prior on every model.
- **Intuition:** each model votes in proportion to how well the data supports it, after paying for its size.
- **Used in:** Sections A.4, D.1 to D.3.

### Posterior inclusion probability
$$\text{PIP}(\text{feature}) = \sum_{k:\ \text{feature} \in M_k} p(M_k \mid D)$$
**Used in:** Sections D.2 and D.4.
