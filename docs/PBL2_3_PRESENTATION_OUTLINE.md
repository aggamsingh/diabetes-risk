# Presentation Outline: PBL 2 (Alternate EM) + PBL 3 (HMM)

16 slides, about 15 minutes, split between Jai, Aggam and Alphin. Every number comes from the notebook outputs, except the dataset counts on slide 3, which were counted directly from the files in `data/`. Full per-slide detail: `docs/PROJECT_WALKTHROUGH.md`.

---

### Slide 1: Cover
**Unsupervised and Sequential Modelling of Diabetes Data**. Team 3: Jai, Aggam, Alphin.
- PBL 2: Gaussian Mixture Model fitted with EM on the Pima data.
- PBL 3: Hidden Markov Model on glucose sequences from AIM-94.

### Slide 2: Project overview
Figure: none.
- PBL 1 (completed): LR, DT, RF and Ridge on 100,000 Kaggle patients, **with** the label.
- PBL 2: risk groups for 768 Pima patients **without** the label (EM).
- PBL 3: glucose states **over time**, 13,515 readings from 70 patients (HMM).
- Common thread: both PBL 2 and 3 are trained with EM; the HMM adds a transition matrix.

### Slide 3: Three datasets, one per task
Figure: none (comparison table).
- Kaggle: 100,000 × 9 raw, 96,146 × 8 modelled (3,854 duplicates removed), diabetes 8.5%. HbA1c and glucose have only 18 distinct values each.
- Pima: 768 × 9 raw, 768 × 5 modelled, Outcome 34.9% (held out). Glucose has 136 distinct values.
- UCI AIM-94: 29,330 × 4 events in 70 patient files, 13,515 glucose readings, 68 patients, 54 / 14 split.
- Kaggle suits prediction, Pima suits Gaussian clustering, AIM-94 suits sequences.

## PBL 2: Alternate EM

### Slide 4: An alternative view of EM
Figure: formulas from `pbl2_alternate_em.ipynb` Section A.
- Complete data (X, Z) vs incomplete data X: ln Σ_Z p(X, Z | θ) has no closed form, so EM maximises the Q-function.
- E-step: responsibilities γ. M-step: closed-form π, μ, Σ.
- Log-likelihood never decreases; local optimum, so 10 restarts. Hard assignment = K-means.

### Slide 5: Data preparation and feature selection
Figure: `figures/pbl2/fig01_feature_correlation.png`
- Dropped Insulin (48.7% missing), SkinThickness (29.6%) and Pregnancies (discrete count: a Gaussian collapses on repeats).
- log(1 + x) on DiabetesPedigreeFunction (skew 1.92 → 1.12) and Age (1.13 → 0.61), then standardised.
- 768 patients, 5 features, max correlation 0.34.

### Slide 6: Choosing the number of components with BIC
Figure: `figures/pbl2/fig02_bic.png`
- Lowest BIC: **diag covariance, K = 3** (10422.4). K = 4 nearly tied (10423.5), so the simpler model is kept.
- BIC = −2 log L + p log N, unlike the elbow method.

### Slide 7: EM convergence: monotonic log-likelihood
Figure: `figures/pbl2/fig03_em_steps.png`
- Log-likelihood −7.094 → −6.646 over 60 steps, **0 decreases**.
- Hand-written E-step = `predict_proba` (difference 1.3e-15). Final model converges in 25 steps.

### Slide 8: Three risk profiles
Figure: `figures/pbl2/fig04_cluster_profiles.png`
- Low risk: 269 patients, age 23, glucose 102, **10.8% diabetic**.
- Older: 237 patients, age 38, lowest family history, **38.0%**.
- High risk: 262 patients, glucose 130, BMI 34.6, highest family history, **56.9%**.
- About 5× rise in diabetes rate without ever using the label.

### Slide 9: Evaluating clustering output: soft vs hard
Figures: `figures/pbl2/fig05_clusters_pca.png`, `figures/pbl2/fig06_responsibility.png`
- K-means: silhouette 0.216, ARI 0.185. GMM: 0.143, 0.101.
- Only the GMM gives probabilities: **30.6%** (235) borderline (max responsibility < 0.8).
- 9 of 10 single random starts end at a worse local maximum, hence n_init = 10.

## PBL 3: Hidden Markov Model

### Slide 10: Sequential data: AIM-94 glucose records
Figures: `figures/pbl3/fig01_temporal_structure.png`, `figures/pbl3/fig02_example_sequence.png`
- **13,515** clean glucose readings, **70** insulin-dependent patients, median 120 readings, 6 h apart.
- Lag-1 autocorrelation 0.186 (0.0 when shuffled): order carries information.
- Kept 68 patients with ≥ 50 readings, **split by patient** (54 train / 14 test).

### Slide 11: Hidden Markov Model: structure and algorithms
Figure: `figures/pbl3/fig03_baum_welch.png`
- λ = (π, A, B), Gaussian emissions.
- Forward: −600.2392 = hmmlearn. Viterbi: identical 104-step path. Baum-Welch (EM): 136 rounds, 5 restarts.
- BIC prefers 5 overlapping states; 3 states (75 / 126 / 206 mg/dL) match the clinical bands.

### Slide 12: Learned parameters: transitions and state durations
Figure: `figures/pbl3/fig04_transition_matrix.png`
- **Low → High = 0.56** (rebound after a hypo). Target stay 0.97, lasts about 33.7 readings.
- Stationary π\* = (0.188, 0.280, 0.532) vs Viterbi share (0.208, 0.304, 0.487).

### Slide 13: Viterbi decoding: test patient 48
Figure: `figures/pbl3/fig05_viterbi_posterior.png`
- 104 readings, about 40-330 mg/dL, **0** decoded as Target.
- Smoothed posterior γ(Low) about 0.85 at the dips: unstable control.

### Slide 14: Evaluation on 14 held-out patients
Figure: none (tables).
- Test log-likelihood per reading: **HMM −5.6847** vs GMM without time −5.7313 vs one Gaussian −5.7861.
- Next-reading MAE: **HMM 60.1** vs training mean 62.6 vs last reading 74.0 mg/dL.

### Slide 15: Clinical interpretation of decoded states
Figures: `figures/pbl3/fig06_state_by_slot.png`, `figures/pbl3/fig07_patient_profiles.png`
- High before breakfast 63.0% (vs 49.4% before lunch): dawn phenomenon.
- Low before the evening snack 29.9%: evening hypoglycaemia.
- 8 of 14 test patients never reach Target; best patient 49 (5.1% High), worst patient 6 (96.9% High).

### Slide 16: Conclusions
Figure: none.
- EM found label-free risk profiles that match PBL 1's top features (glucose, BMI, family history); 30.6% borderline.
- The HMM beats the no-time model on new patients and captures rebounds, morning highs and evening lows.
- Limitations: one population (Pima), Gaussian assumption, irregular gaps treated as steps, glucose only, old and small data, associations not causes.

---

## 2-minute verbal summary
In PBL 1 we predicted diabetes from a single snapshot. We used three datasets, each chosen for what its model needs: Kaggle for prediction, Pima for Gaussian clustering, and AIM-94 for sequences. In PBL 2 we removed the label and asked whether patients form natural groups. We fitted a Gaussian Mixture Model with EM, which alternates an E-step (the probability of each group per patient) and an M-step (refitting the Gaussians). After choosing five suitable features and using BIC, EM found three profiles with diabetes rates of 10.8%, 38.0% and 56.9%, driven by glucose, BMI and family history, the same drivers as in PBL 1. Unlike K-means, it also shows which 30.6% of patients are borderline.

In PBL 3 we moved to a new sequential dataset of 13,515 glucose readings from 70 patients. We modelled the readings as a Hidden Markov Model with three hidden regimes: low, target and high. It was trained with Baum-Welch, the same EM idea applied to sequences, and we checked our own forward and Viterbi code against the library. On new patients the HMM explained the data better than the same model without time and gave the best next-reading forecast. It also revealed clinically known patterns: quick rebounds after lows, high glucose before breakfast and lows in the evening. Time spent in each regime gives a simple, explainable ranking of how well each patient's glucose is controlled.
