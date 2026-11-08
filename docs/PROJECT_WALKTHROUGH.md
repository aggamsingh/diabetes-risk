# PBL 2 & 3 Presentation: Full Walkthrough

**Explainable ML for Early Prediction of Diabetes Risk Using Clinical Data**
Team 3: Jai, Aggam, Alphin · 21CSC305P Machine Learning · Faculty: Dr. Anousouya Devi

---

## The big picture (read this first)

The project asks one question from three angles:

- **PBL 1 (Review 1, already done): supervised learning.** Given a patient's measurements AND the diagnosis label, predict diabetes (Logistic Regression, Decision Tree, Random Forest) and predict BMI (Ridge regression). Kaggle dataset, 100,000 patients.
- **PBL 2 (Unit 3, "Alternate EM" = "An alternative view of EM"): unsupervised learning.** Remove the label. Do patients naturally fall into risk groups? We fit a Gaussian Mixture Model (GMM) with the EM algorithm on the Pima dataset (768 patients).
- **PBL 3 (Unit 4, "HMM"): sequential learning.** A new dataset where each patient is a SEQUENCE of glucose readings over weeks (AIM-94, UCI, 70 patients). A Hidden Markov Model finds hidden glucose states (Low / Target / High) and how patients move between them.

The thread that ties PBL 2 and 3 together: both models are trained with EM. A GMM is "groups without time". An HMM is "groups with time": the same Gaussian groups plus a transition matrix that says how the group of one reading depends on the group of the previous reading. Baum-Welch (HMM training) is literally EM applied to sequences.

**Where the code lives** (GitHub repo `aggamsingh/diabetes-risk`):

| Path | Contents |
|---|---|
| `pbl2_alternate_em.ipynb` | PBL 2 notebook (EM / GMM on Pima) |
| `pbl3_hmm.ipynb` | PBL 3 notebook (HMM on AIM-94 glucose) |
| `data/pima.csv` | Pima data (same file as the PBL 1 Pima notebook) |
| `data/uci_diabetes/` | AIM-94 raw files `data-01` ... `data-70` |
| `figures/pbl2/`, `figures/pbl3/` | every plot used in the slides |
| `docs/DECISIONS.md`, `docs/FORMULAS.md`, `docs/VIVA_PREP.md` | reasons, formulas, viva Q&A |

**Libraries:** pandas, numpy, scikit-learn (GaussianMixture, KMeans, PCA, metrics), scipy (multivariate_normal, norm, logsumexp), hmmlearn (GaussianHMM), matplotlib/seaborn. Python 3.11. `RANDOM_STATE = 42` everywhere, so every number is reproducible. Both notebooks run top to bottom with no errors.

**Rule we followed:** every number on the slides comes from a notebook output. Exceptions: the "Kaggle HbA1c and glucose have only 18 distinct values" fact on slides 3 and 5, and the other dataset counts on slide 3, came from a direct count of the files in `data/`.

---

## Slide 1: Cover

**"Unsupervised and Sequential Modelling of Diabetes Data"**

**What it says:** the course, team, faculty, and the two methods: a Gaussian Mixture Model fitted with EM on Pima, and a Hidden Markov Model on glucose sequences from AIM-94.

**Where it fits:** sets up that this is one combined presentation for PBL 2 (Unit 3) and PBL 3 (Unit 4) on the same diabetes project as Review 1.

No code.

## Slide 2: Project overview

**What it says:**
- PBL 1 (completed): supervised prediction. LR, DT, RF and Ridge regression on 100,000 Kaggle patients.
- PBL 2 & 3 (this review): unsupervised and sequential modelling.
  - Unit 3, alternative view of EM: a Gaussian mixture fitted by EM groups 768 Pima patients into risk groups without using the diagnosis label.
  - Unit 4, Hidden Markov Model: glucose states over time from 13,515 readings of 70 patients.
- Bottom line: both models are fitted with Expectation-Maximisation; the HMM adds a transition matrix between states.

**Where it fits:** the story slide. PBL 1 used labels, PBL 2 drops labels, PBL 3 adds time. That progression is the whole presentation.

No code.

---

# Part 1: PBL 2, alternative view of EM

Notebook: `pbl2_alternate_em.ipynb`

## Slide 3: Three datasets, one per task

**What it says:** one comparison table of the three datasets used across the project.

| | Kaggle Diabetes Prediction | Pima Indians Diabetes | UCI Diabetes (AIM-94) |
|---|---|---|---|
| Used in | PBL 1 · prediction | PBL 2 · EM clustering | PBL 3 · HMM |
| Raw size | 100,000 × 9 | 768 × 9 | 29,330 × 4 · 70 patient files |
| One row is | a patient | a patient | a timestamped event (20 documented codes) |
| Attributes | 8: 4 numeric, 4 binary or categorical | 8, all numeric | date, time, code, value |
| Label | diabetes: 8.5% | Outcome: 34.9% (held out) | none: states are hidden |
| Cleaning | 3,854 duplicates removed | Insulin 48.7%, SkinThickness 29.6% missing: dropped | glucose codes only, 16 bad rows removed |
| Modelled | 96,146 × 8 | 768 × 5 | 13,515 readings · 68 patients · 54 / 14 split |
| Why chosen | large and labelled | continuous values (Glucose: 136 distinct) | time-ordered, median 120 readings per patient |
| Why not reused | HbA1c and glucose: 18 distinct values each | one row per patient, no time order | |

**Code behind it:** none. The numbers were counted directly from the files in `data/` (`data/kaggle_dataset/diabetes_prediction_dataset.csv`, `data/pima.csv`, `data/uci_diabetes/data-01` ... `data-70`). The 13,531 raw glucose rows (codes 48 and 57-64) become 13,515 after removing 13 bad dates or values and 3 readings below 20 mg/dL.

**Where it fits:** the dataset story before any model. Each project asked a different question, so each needed a different kind of data: labelled patients for prediction, continuous values for Gaussian clustering, time-ordered readings for the HMM.

## Slide 4: An alternative view of EM

**What it says** (this is the Unit 3 chapter, in our words):
- Each patient xᵢ has a hidden (latent) cluster label zᵢ.
- **Complete data (X, Z):** if we knew every label, fitting each Gaussian would be easy (just the mean and covariance of its patients).
- **Incomplete data X:** we only see X. The log-likelihood is ln p(X | θ) = ln Σ_Z p(X, Z | θ), a log of a sum, which has no closed-form maximum.
- EM instead maximises the **Q-function**, Q(θ, θᵒˡᵈ) = E_{Z | X, θᵒˡᵈ}[ln p(X, Z | θ)], the expected complete-data log-likelihood.
  - **E-step:** compute the posterior responsibilities γᵢₖ = P(zᵢ = k | xᵢ, θᵒˡᵈ) = πₖ N(xᵢ | μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ | μⱼ, Σⱼ).
  - **M-step:** θⁿᵉʷ = argmax Q, which gives closed-form updates: πₖ = Nₖ / N, μₖ = Σ γᵢₖ xᵢ / Nₖ, Σₖ = γ-weighted covariance.
- Three properties:
  - The observed-data log-likelihood never decreases.
  - EM stops at a local optimum (|Lⁿᵉʷ − Lᵒˡᵈ| < ε), so we use 10 random restarts.
  - Hard assignment (γ = 0 or 1) is K-means; soft assignment is GMM.

**Code behind it:** notebook Section A is the markdown theory with the same formulas in LaTeX. The library doing this is `sklearn.mixture.GaussianMixture`. We do not just trust it: slide 7 shows we re-derived the E-step and M-step by hand and matched the library.

**Where it fits:** the theory our assigned topic is about. Every later PBL 2 slide is this algorithm running on real data.

## Slide 5: Data preparation and feature selection

**What it says:** a table of every Pima column and what we did with it.

| Feature | Decision | Reason |
|---|---|---|
| Insulin | dropped | 48.7% missing |
| SkinThickness | dropped | 29.6% missing |
| Pregnancies | dropped | discrete count: a Gaussian collapses on repeated values |
| Outcome | held out | never used for fitting, only to check the groups afterwards |
| DiabetesPedigreeFunction | log(1 + x) | skew 1.92 → 1.12 |
| Age | log(1 + x) | skew 1.13 → 0.61 |
| Glucose, BloodPressure, BMI | kept | few missing values, median-imputed |

- Result: 768 patients, 5 features, standardised so Euclidean dissimilarity treats all features equally. Max correlation 0.34, so no redundant features.
- Last line: why Pima and not the Review 1 Kaggle data. Kaggle's HbA1c and glucose take only 18 distinct values each, and 4 of its 8 columns are binary or categorical, which breaks Gaussian components.

**Code behind it** (notebook Section B):
- **B.1:** `pd.read_csv("data/pima.csv")`. Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are replaced with NaN (a 0 is physically impossible, so it means "missing"). Then the missing % per column.
- **B.2:** the label is kept aside. A 25% missing rule is applied in code (not a hand-picked list), and Pregnancies is dropped. `SimpleImputer(strategy="median")` fills the few gaps. `.skew()` measures asymmetry.
- **B.3:** `np.log1p` on columns with skew > 1, then `StandardScaler` (z = (x − μ) / σ), then a correlation heatmap (`figures/pbl2/fig01_feature_correlation.png`).

**Why it matters for the viva:** on a first try with Pregnancies included, one Gaussian collapsed onto the patients with exactly 0 pregnancies and BIC went wrong. That is why rule 3 exists. Median-filling half of Insulin would cause the same problem.

## Slide 6: Choosing the number of components with BIC

**What it says:** a BIC table for K = 1 to 6 with diagonal and full covariance.
- The lowest BIC is **10422.4**, at diag covariance with K = 3.
- K = 4 is practically tied (10423.5), so BIC keeps the simpler model.
- Full covariance only wins at K = 2, because each extra group adds a whole covariance matrix of parameters.
- Unlike the elbow method from the K-means notes, BIC uses the likelihood and penalises model size: BIC = −2 log L + p log N.

**Code behind it** (Section C.1): a double loop over `cov in ["full", "diag"]` and `k in 1..8`. Each step fits `GaussianMixture(n_components=k, covariance_type=cov, n_init=10)` and records `gmm.bic(X)` and `gmm.aic(X)`. The lowest BIC is picked in code (`idxmin`), not by eye. Plot: `figures/pbl2/fig02_bic.png`. K = 7 and 8 are left off the slide (higher BIC).

**Where it fits:** the Unit 3 topic "choosing the number of latent dimensions". This fixes K = 3 and diagonal covariance for everything after.

## Slide 7: EM convergence: monotonic log-likelihood

**What it says:** a plot of EM running one step at a time, plus four numbers.
- The average log-likelihood climbs from **−7.094** (step 1) to **−6.646** (step 60).
- **0 decreases** in 60 steps, which proves the monotonic property.
- **1.3e-15:** the gap between our hand-written E-step and the library's `predict_proba`, which is pure rounding error.
- **25 steps** for the final model to converge (|Lⁿᵉʷ − Lᵒˡᵈ| < 10⁻³).

The small scatter plots show the groups after steps 1, 3 and 60, going from mixed to clearly separated (plotted on 2 PCA components).

**Code behind it:**
- **C.2:** `GaussianMixture(max_iter=1, warm_start=True, init_params="random")`. With `max_iter=1`, each `.fit()` call runs exactly ONE E-step and ONE M-step; `warm_start` makes it continue from the last call. Looping 60 times and saving `lower_bound_` after each call gives the curve. `ConvergenceWarning` is silenced on purpose (one step "does not converge" by design). `PCA(n_components=2)` is used only to draw the groups. Figure: `figures/pbl2/fig03_em_steps.png`.
- **C.3:** the final model, `GaussianMixture(..., n_init=10, max_iter=500)`. Converged after 25 steps, weights πₖ = 0.374, 0.296, 0.330.
- **C.4:** the "we understand the library" check.
  - E-step by hand: `scipy.stats.multivariate_normal` gives N(xᵢ | μₖ, Σₖ). Multiply by πₖ, divide by the row sum to get γᵢₖ, and compare with `gmm.predict_proba`: max difference 1.3e-15.
  - M-step by hand: Nₖ = Σ γᵢₖ, πₖ = Nₖ / N, μₖ = (γᵀX) / Nₖ, compared with the fitted values. They differ by at most 0.0034 (weights) and 0.0073 (means), because EM stops at a tolerance, not at the exact maximum.

**Where it fits:** proof that the algorithm on slide 4 is what actually ran.

## Slide 8: Three risk profiles from a 3-component GMM

**What it says:** three cards, one per cluster, each with its diabetes rate. The label was never used in fitting; it is only checked afterwards.

| Cluster | Patients | Diabetic | Profile |
|---|---|---|---|
| Low risk | 269 | 10.8% | age 23, glucose 102, BMI 29.7, BP 65 |
| Older | 237 | 38.0% | age 38, glucose 125, lowest family history (−0.68 SD) |
| High risk | 262 | 56.9% | glucose 130, BMI 34.6, highest family history (+0.73 SD) |

Bottom line: the diabetes rate rises 5× across clusters without using the label, and the drivers (glucose, BMI, family history) match what PBL 1 found.

**Code behind it** (Section D.2): `clusters = gmm.predict(X)` gives each patient the group with the highest responsibility. The table groups patients by cluster and takes medians in ORIGINAL units (before log/scaling, so they are readable), plus group size and the mean of Outcome (diabetes rate). A heatmap of the standardised cluster means (`gmm.means_`) gives the ±SD numbers. Figure: `figures/pbl2/fig04_cluster_profiles.png`.

**Where it fits:** the main PBL 2 result. Unsupervised EM rediscovers clinically sensible risk groups, which is the "explainable" part of the project title.

## Slide 9: Evaluating clustering output: soft vs hard

**What it says:**

| Metric | GMM (soft EM) | K-means (hard) |
|---|---|---|
| Silhouette (internal) | 0.143 | 0.216 |
| ARI vs diagnosis (external) | 0.101 | 0.185 |
| Gives probabilities | yes | no |

- Low ARI is expected: diabetes risk is continuous, so clusters describe risk profiles, not diagnoses. GMM vs K-means agreement: ARI 0.385.
- **30.6%** of patients (235) are borderline, with no group above 0.8 responsibility. Only soft EM can show this.
- **9 of 10** single random starts end at a worse local maximum (−6.670 vs −6.647), which is why we use `n_init = 10`.

**Code behind it:**
- **C.5:** `KMeans(n_clusters=3, n_init=10)` as the hard-EM comparison.
- **D.1:** `silhouette_score(X, labels)` measures internal compactness and separation; `adjusted_rand_score(Outcome, labels)` measures external agreement with the real label (1 = identical, 0 = random).
- **D.3:** PCA scatter of GMM groups, K-means groups and the true label (`figures/pbl2/fig05_clusters_pca.png`, 56.1% variance in 2 components).
- **D.4:** `max_resp = gmm.predict_proba(X).max(axis=1)`. Count patients below 0.8 and plot a histogram (`figures/pbl2/fig06_responsibility.png`).
- **D.5:** refit with `n_init=1` for seeds 0 to 9 and compare each with the final model by ARI and log-likelihood. Only seed 1 reached −6.647.

**Why K-means "wins" on silhouette:** silhouette uses plain distances, which favours K-means's round, equal-sized clusters. The GMM is still more useful here because of the probabilities. Hard vs soft is exactly the comparison the alternative-view-of-EM chapter draws.

End of PBL 2.

---

# Part 2: PBL 3, Hidden Markov Model

Notebook: `pbl3_hmm.ipynb`

## Slide 10: Sequential data: AIM-94 glucose records (UCI)

**What it says:** the new sequential dataset (the Unit 4 rubric asks for one).
- **13,515** clean glucose readings, **70** insulin-dependent patients, a median of **120** readings per patient, a median gap of **6 h** (mostly before meals).
- A plot of one patient's first 4 weeks, swinging in and out of the 70-180 mg/dL target band.
- Temporal dependence: lag-1 autocorrelation ρ(1) = **0.186**, which drops to 0.0 when the order is shuffled. So the order carries information.
- Kept 68 patients with ≥ 50 readings, sorted by time.
- Leakage-free split by patient (order never shuffled): 54 train / 14 test.

**Code behind it** (Section B):
- **B.1:** a loop over `data/uci_diabetes/data-01 ... data-70`, using `pd.read_csv(sep="\t", names=[date, time, code, value])`, plus a patient id.
- **B.2:** keep codes 48 and 57-64 (the glucose measurements; the others are insulin doses, meals, exercise). `pd.to_numeric(errors="coerce")` and `pd.to_datetime(errors="coerce")` turn bad entries into NaN/NaT (for example the date 06-31-1991). Drop those (13) and readings below 20 mg/dL (3), then sort by patient and datetime.
- **B.3:** `groupby("patient").size()` for sequence lengths. A datetime diff for hours between readings. `groupby("patient").shift(1)` gives the previous reading WITHIN each patient, then `.corr()` gives lag-1 autocorrelation; the shuffled version is the control. Figures `fig01_temporal_structure.png` and `fig02_example_sequence.png`.
- **B.4:** keep patients with ≥ 50 readings. `train_test_split` on PATIENT IDS (not readings). Stack all readings into one column `X` plus a `lengths` array, so hmmlearn knows where each patient ends and never learns a fake transition between two different patients.

**Where it fits:** the Unit 4 "sequential data" material: lag features, autocorrelation, and leakage-free splitting.

## Slide 11: Hidden Markov Model: structure and algorithms

**What it says:**
- **λ = (π, A, B):** start probabilities, transition matrix and emissions. Hidden states follow a first-order Markov chain (low, target, high). Gaussian emissions: xₜ | zₜ = j ~ N(μⱼ, σⱼ²). Readings are conditionally independent given the state.
- The three canonical HMM problems and our check of each:

| Task | Algorithm | Our code vs hmmlearn |
|---|---|---|
| Evaluation, p(x₁:T \| λ) | Forward | −600.2392 = −600.2392 |
| Decoding, the MAP path | Viterbi (max-product) | identical path for all 104 readings |
| Learning, estimate λ | Baum-Welch (EM) | converged in 136 rounds, 5 restarts |

- BIC prefers 5 states, but they overlap (means 74, 117, 180, 191, 211). We use 3 because they match the clinical ranges: 75 low, 126 target, 206 high mg/dL.
- Footer: all recursions run in the log domain with log-sum-exp to avoid underflow (the "scaling factors" topic).

**Code behind it:**
- **C.1:** a helper `fit_best_hmm(k, X, lengths)` fits `GaussianHMM(n_components=k, covariance_type="diag", n_iter=200, tol=1e-4)` from 5 random seeds and keeps the one with the highest training log-likelihood (Baum-Welch only finds local optima). A BIC table for K = 2..6 uses `model.bic(X_train, len_train)`.
- **C.2:** `N_STATES = 3`, the final fit. `hmm.monitor_.history` gives the Baum-Welch convergence curve (`figures/pbl3/fig03_baum_welch.png`).
- **C.4:** two hand-written functions, which are the core "implementation" proof:
  - `forward(x, model)`: log b = `norm.logpdf(x, means, stds)`, then log α₁ = log π + log b₁ and log αₜ = `logsumexp`(log αₜ₋₁ + log A) + log bₜ. It returns log p(x) and the filtered state probabilities. Compared with `hmm.score(x)`: identical.
  - `viterbi(x, model)`: the same recursion with `max` instead of `logsumexp`, storing back-pointers (`argmax`) and then backtracking. Compared with `hmm.decode(x, algorithm="viterbi")`: identical path and score.

**Where it fits:** the Unit 4 HMM, forward-backward, Viterbi, scaling and EM topics in one slide. The K = 3 choice is a deliberate fit-vs-interpretability trade-off, and we say so openly.

## Slide 12: Learned parameters: transitions and state durations

**What it says:**

Transition matrix A (aᵢⱼ = P(zₜ = j | zₜ₋₁ = i), rows sum to 1):

| From \ To | Low | Target | High |
|---|---|---|---|
| Low | 0.43 | 0.01 | 0.56 |
| Target | 0.00 | 0.97 | 0.03 |
| High | 0.20 | 0.01 | 0.79 |

States, with expected duration 1 / (1 − aᵢᵢ):

| State | Mean (mg/dL) | Stay | Lasts (readings) |
|---|---|---|---|
| Low | 74.8 | 0.43 | 1.75 |
| Target | 126.1 | 0.97 | 33.7 |
| High | 205.5 | 0.79 | 4.7 |

- **Low → High = 0.56:** a low is usually followed by a high (the post-hypoglycaemia rebound).
- **Low/High → Target ≤ 0.01:** Target behaves as a separate, stable regime.
- **Stationary distribution π\* = π\*A = (0.188, 0.280, 0.532)**, close to the Viterbi share of training readings (0.208, 0.304, 0.487). From Low, Aᵏ reaches π\* only after about 100 steps.

**Code behind it:**
- **C.3:** EM state numbers are arbitrary, so we sort states by mean glucose: `order = np.argsort(hmm.means_)`, `rank[order] = 0, 1, 2`, and A is reordered as `hmm.transmat_[order][:, order]`. A params table holds the means, std (sqrt of `covars_`), start prob, stay prob (diagonal of A) and 1 / (1 − diag(A)). Heatmap: `figures/pbl3/fig04_transition_matrix.png`.
- **C.3b** (Markov models chapter):
  - `np.linalg.matrix_power(A, k)` for k = 1, 2, 5, 20, 100 gives k-step transitions.
  - The stationary distribution is the left eigenvector of A for eigenvalue 1: `np.linalg.eig(A.T)`, take the eigenvector closest to eigenvalue 1, normalise to sum 1.
  - It is checked against the actual share of training readings Viterbi assigns to each state (`hmm.predict` on all training data, then `np.bincount`).

**Where it fits:** interpretation of λ, and the Unit 4 Markov-model topics (transition matrix, Aᵏ, stationary distribution).

## Slide 13: Viterbi decoding: test patient 48

**What it says:** one unseen test patient decoded by the model.
- **Top plot:** each reading coloured by its Viterbi state (the MAP path).
- **Bottom plot:** the smoothed posterior γₜ(i) from forward-backward, which shows how sure the model is about each state at each reading.
- Range ≈ 40-330 mg/dL over 104 readings, **0** readings decoded as Target, and the peak smoothed posterior γₜ(Low) ≈ 0.85 at the dips.
- Interpretation: unstable control, mostly High with repeated hypo dips. One average value would hide this.

**Code behind it** (Section D.2): `pid = test_ids[0]` (patient 48). `hmm.predict(x)` gives the Viterbi path; `hmm.predict_proba(x)` gives the forward-backward posteriors. Both are mapped to sorted state order with `rank`/`order`. A scatter coloured by state sits on top and a stackplot of the posteriors below. Figure: `figures/pbl3/fig05_viterbi_posterior.png`.

**Note on the syllabus:** Viterbi gives the best JOINT path; forward-backward gives the best state per time step. They usually agree but need not. That is the "Viterbi vs posterior decoding" point in the notes.

## Slide 14: Evaluation on 14 held-out patients

**What it says:** two tables, both computed on patients the model never saw.

| Test log-likelihood per reading (higher is better) | |
|---|---|
| HMM, 3 states | −5.6847 |
| GMM, no time order (same 3 Gaussians, no transitions) | −5.7313 |
| Single Gaussian | −5.7861 |

| Next-reading forecast MAE, mg/dL (lower is better) | |
|---|---|
| HMM, using the filtered P(zₜ \| x₁:ₜ) times A | 60.1 |
| Training mean | 62.6 |
| Last reading | 74.0 |

Bottom line: both models share the same Gaussian emissions; the gain comes from the transition matrix.

**Code behind it:**
- **D.1:** `hmm.score(X_test, len_test) / len(X_test)` against `GaussianMixture(3).fit(X_train).score(X_test)` and `GaussianMixture(1).score(X_test)`. The GMM is exactly the PBL 2 model without time, so this is the fairest test of "does order help".
- **D.3:** for each test patient, our own `forward()` gives the FILTERED probabilities P(zₜ | x₁:ₜ), using only the past (no peeking at the future). Multiplying by `transmat_` gives P(zₜ₊₁ | x₁:ₜ); multiplying by the state means gives the expected next reading. Its MAE is compared with two baselines.

**Where it fits:** proves the sequential model is better on new patients. Filtering vs smoothing (the forward-backward chapter) shows up here: slide 13 uses smoothing (all readings), this slide uses filtering (past only), because a forecast cannot use the future.

**Why the forecast gain is small:** readings are about 6 h apart and only weakly related (ρ(1) = 0.186), so the exact next value is hard to predict. Copying the last reading is worst because it copies that reading's noise.

## Slide 15: Clinical interpretation of decoded states

**What it says:**
- **63.0%** of readings are High before breakfast (vs 49.4% before lunch), consistent with the dawn phenomenon.
- **29.9%** are Low before the evening snack (24.6% after supper), consistent with evening hypoglycaemia.
- **8 of 14** test patients never reach Target; 4 spend 26-58% there.
- Time in each state ranks patients by control: the best is patient 49 (5.1% High), the worst is patient 6 (96.9% High).

**Code behind it:**
- **D.4:** Viterbi-decode every test patient, then `pd.crosstab(test["slot"], test["state"], normalize="index")` gives the % of each state per meal slot. Slots with < 30 readings are skipped in the plot (`figures/pbl3/fig06_state_by_slot.png`). The model never saw the meal slot, so these patterns are discoveries, not inputs.
- **D.5:** `pd.crosstab(patient, state, normalize="index")` gives the % of readings per state for each test patient (`figures/pbl3/fig07_patient_profiles.png`).

**Where it fits:** the "explainable" payoff of PBL 3. Hidden states turn weeks of noisy readings into something a clinician understands, similar to the clinical "time in range" measure.

## Slide 16: Conclusions

**What it says:**
- **Gaussian Mixture Model (EM):** three risk clusters found without the label, 10.8% / 38.0% / 56.9% diabetic. The drivers agree with PBL 1, and 30.6% of patients are borderline.
- **Hidden Markov Model:** three states at 75 / 126 / 206 mg/dL, with a higher test likelihood than the no-time model (−5.685 vs −5.731). It captures rebounds, morning highs and evening lows.
- **Limitations:**
  - Pima covers one population of women, and EM assumes Gaussian clusters.
  - The HMM treats each reading as one step even though the gaps are irregular, it uses glucose only (no insulin or meals), and the data is old (70 Type 1 patients).
  - The results are associations, not causes.

No code. It wraps the two parts back into the one project story from slide 2.

---

## How the notebooks map to the slides (quick reference)

**`pbl2_alternate_em.ipynb`**

| Notebook section | Slide |
|---|---|
| A (theory) | 4 |
| B.1-B.3 (data, features) | 5 |
| C.1 (BIC) | 6 |
| C.2-C.4 (EM steps, hand check) | 7 |
| D.2 (profiles) | 8 |
| C.5, D.1, D.3-D.5 (evaluation) | 9 |

**`pbl3_hmm.ipynb`**

| Notebook section | Slide |
|---|---|
| B.1-B.4 (data, sequence) | 10 |
| A (theory), C.1, C.2, C.4 (fit, hand check) | 11 |
| C.3, C.3b (parameters, π\*) | 12 |
| D.2 (patient 48) | 13 |
| D.1, D.3 (evaluation) | 14 |
| D.4, D.5 (interpretation) | 15 |

## Syllabus terms used (for the viva)

**Unit 3:** K-means clustering, mixtures of Gaussians, an alternative view of EM (latent variables, complete vs incomplete data, Q-function, responsibilities, E-step, M-step, convergence, local optimum, hard vs soft assignment), PCA, choosing the number of latent dimensions (BIC vs elbow), measuring dissimilarity (standardised Euclidean), evaluating clustering output (silhouette = internal, ARI = external).

**Unit 4:** sequential data (lag, autocorrelation ρ(k), leakage-free split), Markov models (first-order Markov property, transition matrix, Aᵏ, stationary distribution π\* = π\*A), HMM (λ = (π, A, B), Gaussian emissions, conditional independence, three canonical problems), maximum likelihood / EM (Baum-Welch, multiple restarts), forward-backward (α, β, γ, filtering vs smoothing), scaling factors (log-sum-exp against underflow), Viterbi (δ, back-pointers, MAP path, max-product).

**Not used (not assigned):** factor analysis and hierarchical clustering (Unit 3), sum-product and linear dynamical systems (Unit 4). If asked: forward-backward is the sum-product algorithm on the HMM chain, and Viterbi is its max-product version.
