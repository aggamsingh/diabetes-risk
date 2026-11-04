# Presentation Outline: PBL 2 (Alternate EM) + PBL 3 (HMM)

12 slides, about 10-12 minutes, split between Jai, Aggam and Alphin. Every number comes from the notebook outputs.

---

### Slide 1: Title
**Explainable ML for Early Prediction of Diabetes Risk: PBL 2 and PBL 3**. Team 3: Jai, Aggam, Alphin.
- PBL 2: Alternate EM (Gaussian Mixture Model) on the Pima data.
- PBL 3: Hidden Markov Model on blood-glucose sequences.

### Slide 2: Where this fits
Figure: none.
- PBL 1: predict diabetes from one snapshot, **with** the label.
- PBL 2: find patient risk profiles **without** the label (EM).
- PBL 3: follow patients **over time** and find hidden glucose regimes (HMM).
- Common thread: both PBL 2 and 3 are trained with EM.

## PBL 2: Alternate EM

### Slide 3: The EM idea
Figure: formulas from `pbl2_alternate_em.ipynb` Section A.3.
- GMM: each patient comes from one of K hidden Gaussian groups.
- E-step: responsibilities γ (the probability of each group per patient). M-step: refit π, μ, Σ as γ-weighted averages.
- Each round never lowers the log-likelihood; it reaches a local maximum, so we use 10 random starts. K-means = hard EM.

### Slide 4: Data preparation and feature selection
Figure: `figures/pbl2/fig01_feature_correlation.png`
- Pima, 768 patients; label kept aside.
- Dropped Insulin (48.7% missing), SkinThickness (29.6%) and Pregnancies (discrete count): repeated values make a Gaussian collapse.
- Log of DiabetesPedigreeFunction and Age (skew > 1), then standardised. Five weakly correlated features (max r = 0.34).

### Slide 5: Choosing K and watching EM alternate
Figures: `figures/pbl2/fig02_bic.png`, `figures/pbl2/fig03_em_steps.png`
- BIC picks **diag covariance, K = 3** (10422.4).
- Step by step, the log-likelihood rises from -7.094 to -6.646 and never decreases.
- Hand-written E-step = `predict_proba` (difference 1e-15).

### Slide 6: The three risk profiles
Figure: `figures/pbl2/fig04_cluster_profiles.png`
- Low risk: young (23), low glucose (102), **10.8% diabetic**.
- Older: age 38, low family history, **38.0%**.
- High risk: high glucose (130), BMI (34.6) and family history, **56.9%**.
- A five-fold risk difference, without ever using the label.

### Slide 7: Evaluation (GMM vs K-means)
Figures: `figures/pbl2/fig05_clusters_pca.png`, `figures/pbl2/fig06_responsibility.png`
- K-means: silhouette 0.216, ARI 0.185. GMM: 0.143, 0.101. Diabetes is a gradual risk, not a separate cluster.
- Only the GMM gives probabilities: **30.6%** of patients are borderline (max responsibility < 0.8).
- Single random starts: 9 of 10 stuck at a worse local maximum, which is why we use n_init = 10.

## PBL 3: Hidden Markov Model

### Slide 8: The new sequential dataset
Figures: `figures/pbl3/fig01_temporal_structure.png`, `figures/pbl3/fig02_example_sequence.png`
- AIM-94 / UCI Diabetes: **13,515 glucose readings from 70 insulin-dependent patients**, weeks to months each.
- Median 120 readings per patient, 6 h apart, mostly before meals.
- Lag-1 autocorrelation 0.186 (0.0 when shuffled): order matters, but readings are noisy.
- Sorted by time, kept patients with ≥ 50 readings, **split by patient** (54 train / 14 test).

### Slide 9: HMM and its algorithms
Figure: `figures/pbl3/fig03_baum_welch.png`
- λ = (π, A, B), Gaussian emissions. Forward = P(x), Viterbi = best path, Baum-Welch = EM.
- BIC prefers 5 states, but 3 states (75 / 126 / 206 mg/dL) match the clinical low / target / high bands.
- Our hand-written forward and Viterbi match hmmlearn exactly (-600.2392, identical 104-step path).

### Slide 10: What the model learned
Figure: `figures/pbl3/fig04_transition_matrix.png`
- Low lasts about 1.75 readings; **Low → High = 0.56** (rebound after a hypo).
- Target is a stable regime (stay 0.97, about 33.7 readings) that Low/High almost never enter.
- High peaks before breakfast (63.0%), and lows are most common in the evening (pre-snack 29.9%). See `fig06_state_by_slot.png`.

### Slide 11: Evaluation on new patients
Figures: `figures/pbl3/fig05_viterbi_posterior.png`, `figures/pbl3/fig07_patient_profiles.png`
- Test log-likelihood per reading: **HMM -5.685** vs GMM without time -5.731 vs one Gaussian -5.786.
- Next-reading MAE: **HMM 60.1** vs mean 62.6 vs last reading 74.0 mg/dL.
- Time in each regime ranks patients: 4 of 14 spend 26-58% in Target; 8 never reach it.

### Slide 12: Conclusion and limitations
Figure: none.
- EM found label-free risk profiles that match PBL 1's top features (glucose, BMI, family history).
- The HMM turns weeks of noisy readings into an explainable summary of glucose control.
- Limitations: one step = one reading (irregular gaps), glucose only, old and small data, associations not diagnoses.

---

## 2-minute verbal summary
In PBL 1 we predicted diabetes from a single snapshot. In PBL 2 we removed the label and asked whether patients form natural groups. We fitted a Gaussian Mixture Model with EM, which alternates an E-step (the probability of each group per patient) and an M-step (refitting the Gaussians). After choosing five suitable features and using BIC, EM found three profiles with diabetes rates of 10.8%, 38.0% and 56.9%, driven by glucose, BMI and family history, the same drivers as in PBL 1. Unlike K-means, it also shows which 30.6% of patients are borderline.

In PBL 3 we moved to a new sequential dataset of 13,515 glucose readings from 70 patients. We modelled the readings as a Hidden Markov Model with three hidden regimes: low, target and high. It was trained with Baum-Welch, the same EM idea applied to sequences, and we checked our own forward and Viterbi code against the library. On new patients the HMM explained the data better than the same model without time and gave the best next-reading forecast. It also revealed clinically known patterns: quick rebounds after lows, high glucose before breakfast and lows in the evening. Time spent in each regime gives a simple, explainable ranking of how well each patient's glucose is controlled.
