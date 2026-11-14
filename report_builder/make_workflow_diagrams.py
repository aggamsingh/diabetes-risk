import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "C:/Programming/Projects/diabetes-risk-repo/figures/report"
os.makedirs(OUT, exist_ok=True)
INK, ACC, FILL, OUTF = "#17212B", "#2457A6", "#EEF3FB", "#FFF4E5"

def flow(name, title, steps, outputs=None):
    # steps: list of (heading, detail) in a left-to-right flow wrapping into rows of 4
    per_row = 4
    rows = (len(steps) + per_row - 1) // per_row
    fig_h = 1.0 + rows * 2.0 + (1.6 if outputs else 0)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 13); ax.set_ylim(0, fig_h); ax.axis("off")
    ax.text(0.2, fig_h - 0.45, title, fontsize=15, fontweight="bold", color=INK, va="center")
    w, h, gap = 2.75, 1.45, 0.55
    centres = []
    for i, (head, detail) in enumerate(steps):
        r, c = divmod(i, per_row)
        if r % 2 == 1:
            c = per_row - 1 - c            # snake layout so arrows stay short
        x = 0.25 + c * (w + gap); y = fig_h - 1.1 - (r + 1) * 2.0 + 0.3
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=FILL, ec=ACC, lw=1.6))
        ax.text(x + w / 2, y + h - 0.32, f"{i + 1}. {head}", ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 0.55, detail, ha="center", va="center", fontsize=9.5, color="#33404D", wrap=True)
        centres.append((x, y))
    for (x1, y1), (x2, y2) in zip(centres, centres[1:]):
        if abs(y1 - y2) < 1e-6:
            sx, ex = (x1 + w, x2) if x2 > x1 else (x1, x2 + w)
            ax.annotate("", xy=(ex, y2 + h / 2), xytext=(sx, y1 + h / 2), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
        else:
            ax.annotate("", xy=(x2 + w / 2, y2 + h), xytext=(x1 + w / 2, y1), arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    if outputs:
        ax.add_patch(FancyBboxPatch((0.25, 0.25), 12.5, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12", fc=OUTF, ec="#C07A1A", lw=1.4))
        ax.text(6.5, 0.75, "Outputs: " + outputs, ha="center", va="center", fontsize=10.5, color=INK)
    plt.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()

flow("workflow_ch2", "PBL 1 workflow: data analysis, regression and classification",
     [("Load data", "Pima 768 x 9\nProgression 442 x 11\nKaggle 100,000 x 9"),
      ("Clean", "impossible zeros -> NaN\nduplicates removed\nBMI 27.32 -> missing"),
      ("Analyse (EDA)", "class balance, histograms,\nboxplots, correlations"),
      ("Split", "80 / 20, stratified\nrandom_state = 42"),
      ("Pipeline", "median imputer ->\nStandardScaler (if needed)\n-> model"),
      ("Train and compare", "5-fold CV: 7 classifiers,\n7 regressors"),
      ("Tune and threshold", "GridSearchCV on ROC-AUC,\nthreshold for recall"),
      ("Explain", "odds ratios, permutation\nimportance, SHAP")],
     "test recall / precision / ROC-AUC, R2 / RMSE / MAE, risk drivers")

flow("workflow_ch3_unit3", "Unit 3 workflow: Gaussian mixture fitted by EM",
     [("Pima data", "768 patients,\nlabel held out"),
      ("Select features", "drop Insulin, SkinThickness\n(>25% missing), Pregnancies"),
      ("Transform", "median impute, log1p on\nskewed, standardise"),
      ("Choose K", "BIC over K = 1..8,\nfull vs diagonal covariance"),
      ("Run EM", "E-step responsibilities,\nM-step updates, 10 restarts"),
      ("Check by hand", "own E-step and M-step\nvs scikit-learn"),
      ("Compare", "K-means (hard EM),\nsilhouette, ARI, PCA"),
      ("Interpret", "risk profiles, borderline\npatients, local optima")],
     "3 risk profiles (10.8 / 38.0 / 56.9% diabetic), 30.6% borderline")

flow("workflow_ch3_unit4", "Unit 4 workflow: Hidden Markov Model on glucose sequences",
     [("AIM-94 files", "70 patients,\n29,330 timestamped events"),
      ("Extract glucose", "codes 48, 57-64,\n16 bad rows removed"),
      ("Order and split", "sort by time per patient,\n>= 50 readings, 54 / 14 by patient"),
      ("Choose K", "BIC for K = 2..6,\nK = 3 for interpretation"),
      ("Baum-Welch", "EM for HMMs,\n5 random starts"),
      ("Check by hand", "own forward and Viterbi\nin log space vs hmmlearn"),
      ("Evaluate", "test log-likelihood vs GMM,\none-step forecast MAE"),
      ("Interpret", "transitions, durations,\nstationary distribution, slots")],
     "3 glucose states (75 / 126 / 206 mg/dL), better fit than the no-time model")

flow("workflow_ch4", "Unit 5 workflow: boosting and Bayesian model averaging",
     [("Kaggle data", "96,146 patients after\nduplicates, PBL 1 split"),
      ("Encode", "one-hot gender and smoking,\nmedian BMI, 15 columns"),
      ("Baselines", "stump, CART tree,\nrandom forest"),
      ("AdaBoost", "500 stumps; own loop\nchecked against library"),
      ("Analyse boosting", "weights, sigma(vote)\ncalibration, label noise"),
      ("BMA", "256 logistic models,\nBIC posterior weights"),
      ("Validate", "5-fold CV, small-data\nsubsets, no-blood-test rerun"),
      ("Decide", "screening threshold,\nmodel choice")],
     "ROC-AUC 0.977 with blood tests, 0.825 without; BMA helps only with little data")
print(sorted(os.listdir(OUT)))
