// Builds the integrated ML project report (docx) following the course template.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, PageBreak, Footer, PageNumber, NumberFormat, TabStopType,
  TableOfContents, LevelFormat, BorderStyle, ExternalHyperlink,
} = require("docx");

const REPO = "C:/Programming/Projects/diabetes-risk-repo";
const FIG = (p) => path.join(REPO, "figures", p);
const FONT = "Times New Roman";
const TEXT_W = 11906 - 2160 - 1440; // A4 width minus 1.5" left and 1" right margins (DXA)
const COLAB = (nb) => `https://colab.research.google.com/github/aggamsingh/diabetes-risk/blob/main/${nb}`;

// ---------- helpers ----------
function runs(text, base = {}) {
  // **bold** markup
  return text.split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) =>
    s.startsWith("**") ? new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...base })
                       : new TextRun({ text: s, font: FONT, ...base }));
}
const P = (t, o = {}) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, children: runs(t), ...o });
const PC = (t, o = {}) => new Paragraph({ alignment: AlignmentType.CENTER, children: runs(t), ...o });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
const BL = (items) => items.map(B);

let chap = 0, figN = 0, tabN = 0, eqN = 0;
const figList = [], tabList = [];

function chapter(title) {
  chap += 1; figN = 0; tabN = 0; eqN = 0;
  return [
    new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 60 },
      children: [new TextRun({ text: `CHAPTER ${chap}`, bold: true, size: 32, font: FONT })] }),
    new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: `${chap}. ${title}`, font: FONT })] }),
  ];
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  if (b.slice(1, 4).toString() === "PNG") return [b.readUInt32BE(16), b.readUInt32BE(20)];
  // JPEG: scan for SOF0/SOF2
  let i = 2;
  while (i < b.length) {
    const marker = b.readUInt16BE(i);
    const len = b.readUInt16BE(i + 2);
    if (marker === 0xffc0 || marker === 0xffc2) return [b.readUInt16BE(i + 7), b.readUInt16BE(i + 5)];
    i += 2 + len;
  }
  return [600, 400];
}

function figure(file, caption, widthIn = 6.0) {
  figN += 1;
  const id = `${chap}.${figN}`;
  figList.push([id, caption]);
  const [w, h] = pngSize(file);
  const width = Math.round(widthIn * 96);
  const height = Math.round((width * h) / w);
  const type = file.toLowerCase().endsWith(".png") ? "png" : "jpg";
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type, data: fs.readFileSync(file), transformation: { width, height },
        altText: { title: caption, description: caption, name: `Figure ${id}` } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Figure ${id}: ${caption}`, bold: true, size: 22, font: FONT })] }),
  ];
}

function table(caption, header, rows, widths) {
  tabN += 1;
  const id = `${chap}.${tabN}`;
  tabList.push([id, caption]);
  const total = widths.reduce((a, b) => a + b, 0);
  const cw = widths.map((w) => Math.round((w / total) * TEXT_W));
  cw[cw.length - 1] += TEXT_W - cw.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, hdr) => new TableCell({
    width: { size: cw[i], type: WidthType.DXA }, borders,
    shading: hdr ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { line: 240, after: 0 }, alignment: AlignmentType.LEFT,
      children: runs(String(t), { size: 22, bold: hdr || undefined }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: `Table ${id}: ${caption}`, bold: true, size: 22, font: FONT })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: cw,
      rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }),
             ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function eq(text) {
  eqN += 1;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text: `\t${text}\t(${chap}.${eqN})`, font: "Cambria Math", size: 24 })],
  });
}

function code(lines) {
  return lines.map((l) => new Paragraph({ spacing: { line: 240, before: 0, after: 0 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
    children: [new TextRun({ text: l || " ", font: "Consolas", size: 20 })] }));
}

// ---------- cover ----------
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 240 },
    children: [new TextRun({ text: "Explainable Machine Learning for Early Prediction of Diabetes Risk Using Clinical Data", bold: true, size: 36, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: "21CSC305P - MACHINE LEARNING PROJECT", bold: true, size: 28, font: FONT })] }),
  PC("**Submitted by**"),
  PC("Jai [RA2411026010131]"),
  PC("Aggam [RA2411026010139]"),
  PC("Alphin [RA2411026010177]", { spacing: { after: 240 } }),
  PC("**Under the Guidance of**"),
  PC("Dr. Anousouya Devi"),
  PC("([Designation], Department of Computational Intelligence)", { spacing: { after: 240 } }),
  PC("in partial fulfillment of the requirements for the degree of"),
  PC("**BACHELOR OF TECHNOLOGY**"),
  PC("in"),
  PC("**COMPUTER SCIENCE ENGINEERING**"),
  PC("**with specialization in AIML**", { spacing: { after: 200 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "jpg", data: fs.readFileSync(path.join(__dirname, "logo.jpeg")),
    transformation: (() => { const [w, h] = pngSize(path.join(__dirname, "logo.jpeg")); const W = 200; return { width: W, height: Math.round(W * h / w) }; })(),
    altText: { title: "SRM logo", description: "SRM Institute of Science and Technology logo", name: "logo" } })] }),
  PC("**DEPARTMENT OF COMPUTATIONAL INTELLIGENCE**", { spacing: { before: 200 } }),
  PC("**COLLEGE OF ENGINEERING AND TECHNOLOGY**"),
  PC("**SRM INSTITUTE OF SCIENCE AND TECHNOLOGY**"),
  PC("**KATTANKULATHUR - 603 203**"),
  PC("**NOV 2026**", { spacing: { before: 200 } }),
];

// ---------- chapters ----------
const body = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? body.push(...x) : body.push(x)));

// CHAPTER 1
add(chapter("INTRODUCTION"));
add(H2("1.1 Background, Motivation and Scope"));
add(P("Diabetes mellitus is a chronic condition in which blood glucose stays too high, either because the body produces too little insulin (Type 1) or responds poorly to it (Type 2). Many people live with undiagnosed diabetes or prediabetes for years, and early detection allows lifestyle changes and treatment before complications such as heart, kidney and eye disease develop. Routine clinical measurements (age, body mass index, blood pressure, family history and, where available, blood glucose and HbA1c) already carry much of the information needed to flag people at risk."));
add(P("This project builds an **explainable** machine learning pipeline for diabetes risk. The stakeholders are clinicians and screening programmes, who need to know not only who is at risk but **why**, and patients, who benefit from early, low-cost screening. The scope is a decision-support prototype built on public datasets: it ranks and groups patients and explains its decisions; it is not a diagnostic device."));
add(H2("1.2 Integrated Problem Statement"));
add(P("**Overall problem:** given routine clinical data, identify people at high risk of diabetes early, and explain which factors drive that risk. The same problem is studied from five angles, one per unit:"));
add(BL([
  "**Units 1 and 2 (original datasets):** analyse the data, then predict diabetes (classification) and disease progression or BMI (regression) with supervised models.",
  "**Unit 3 (original dataset, Pima):** without using the diagnosis, find natural risk groups with a Gaussian mixture fitted by the EM algorithm.",
  "**Unit 4 (separate sequential dataset, AIM-94):** model how a patient's blood glucose moves between hidden low, target and high states over time with a Hidden Markov Model.",
  "**Unit 5 (original dataset, Kaggle):** combine models with boosting and Bayesian model averaging, and test whether combining helps, including in a realistic screening setting without blood tests.",
]));
add(H2("1.3 Project Objectives"));
add(BL([
  "**Objective 1 (Unit 1, data analysis):** profile the Pima, Diabetes Progression and Kaggle datasets, detect hidden missing values and data issues, and identify the main risk factors before modelling.",
  "**Objective 2 (Unit 2, regression and classification):** reach test recall of at least 0.75 and ROC-AUC of at least 0.80 for diabetes classification, and beat the mean baseline in regression (target R² about 0.40).",
  "**Objective 3 (Unit 3, Alternate EM):** fit a Gaussian mixture by EM without labels, choose the number of clusters with BIC, verify the E and M steps by hand, and check whether the clusters separate diabetes risk.",
  "**Objective 4 (Unit 4, Hidden Markov Model):** learn glucose states and transitions from patient sequences, verify forward and Viterbi by hand, and show that time order improves the fit on new patients.",
  "**Objective 5 (Unit 5, Combining Models):** build AdaBoost from decision stumps and verify it by hand, compare boosting with bagging and single models, and test when Bayesian model averaging improves on choosing one model.",
]));
add(H2("1.4 Unit-wise Topic and Dataset Allocation"));
add(table("Unit-wise topic and dataset allocation",
  ["PBL", "Unit", "Allotted topic / algorithm", "Dataset"],
  [["PBL 1", "1", "Data analysis: EDA, data quality, preprocessing", "Original: Pima, Diabetes Progression, Kaggle"],
   ["PBL 1", "2", "Regression (Linear, Ridge, Lasso) and classification (Logistic regression, SVM, trees and others)", "Original: Pima, Diabetes Progression, Kaggle"],
   ["PBL 2", "3", "Alternate EM: Gaussian mixture model, K-means as hard EM", "Original: Pima"],
   ["PBL 2", "4", "Hidden Markov Model (Baum-Welch, forward-backward, Viterbi)", "Approved sequential dataset: UCI Diabetes (AIM-94)"],
   ["PBL 3", "5", "Combining models: boosting (AdaBoost) and Bayesian model averaging", "Original: Kaggle"]],
  [10, 8, 52, 30]));
add(P("Note on numbering: the team presentations were held as PBL 1 (Units 1 and 2), a combined PBL 2 and 3 review (Units 3 and 4) and a PBL 4 review (Unit 5). This report follows the template's numbering above."));

// CHAPTER 2
add(chapter("DATA ANALYSIS AND FUNDAMENTAL MACHINE LEARNING TECHNIQUES (UNITS 1 AND 2)"));
add(H2("2.1 Overview of Allotted Topics"));
add(P("**Unit 1, data analysis.** Before any model, the data is profiled: data types, value ranges, class balance, distributions, correlations and, most importantly, data quality. In clinical data a value can look valid but be impossible (a glucose or BMI of 0), so these checks decide what the models can be trusted to learn."));
add(P("**Unit 2, regression and classification.** Regression predicts a number (disease progression after one year, or BMI); classification predicts a class (diabetic or not). Linear models (linear, Ridge, Lasso and logistic regression) are interpretable through their coefficients; Ridge and Lasso add a penalty that shrinks coefficients to reduce overfitting, and Lasso can set some to exactly zero. Non-linear models (k-nearest neighbours, decision tree, random forest, gradient boosting, Gaussian naive Bayes and the support vector machine with an RBF kernel) can capture interactions and thresholds. Comparing both families on the same folds shows whether the extra complexity pays off."));
add(H2("2.2 Dataset Description"));
add(table("Datasets used in Units 1 and 2",
  ["", "Pima Indians Diabetes", "Diabetes Progression", "Kaggle Diabetes Prediction"],
  [["Source", "NIDDK, via OpenML (fetch_openml, name diabetes, version 1)", "Efron et al. (2004), scikit-learn load_diabetes(scaled=False)", "Kaggle, diabetes_prediction_dataset.csv"],
   ["Size", "768 rows x 9 columns", "442 rows x 11 columns", "100,000 rows x 9 columns (96,146 after removing 3,854 duplicates)"],
   ["Features", "8 numeric clinical attributes", "10 baseline attributes (age, sex, BMI, BP, 6 serum measures s1 to s6)", "8 attributes: 4 numeric, 4 binary or categorical"],
   ["Target", "Outcome (1 = diabetic), 34.9% positive", "progression after one year, mean 152.1, SD 77.1", "diabetes (1 = diabetic), 8.8% positive after cleaning"],
   ["Data issues", "impossible zeros in 5 columns (hidden missing values)", "none; small sample", "duplicates; BMI 27.32 used as a placeholder for 22.5%; HbA1c and glucose rounded to 18 values each"],
   ["Limitations", "women of Pima heritage aged 21+ only", "442 patients only", "label nearly follows HbA1c and glucose cut-offs"]],
  [16, 28, 28, 28]));
add(table("Pima attributes and hidden missing values (zeros)",
  ["Attribute", "Meaning", "Range", "Zeros (missing)"],
  [["Pregnancies", "number of pregnancies", "0 to 17", "111 (real zeros)"],
   ["Glucose", "plasma glucose 2 h after an oral glucose tolerance test (mg/dL)", "0 to 199", "5"],
   ["BloodPressure", "diastolic blood pressure (mm Hg)", "0 to 122", "35"],
   ["SkinThickness", "triceps skin fold (mm)", "0 to 99", "227 (29.6%)"],
   ["Insulin", "2-hour serum insulin (mu U/mL)", "0 to 846", "374 (48.7%)"],
   ["BMI", "body mass index (kg/m²)", "0 to 67.1", "11"],
   ["DiabetesPedigreeFunction", "family history score", "0.078 to 2.42", "0"],
   ["Age", "years", "21 to 81", "0"]],
  [26, 40, 16, 18]));
add(P("The Kaggle attributes are gender (Female 58,552, Male 41,430, Other 18), age (0.08 to 80), hypertension (7.5% yes), heart_disease (3.9% yes), smoking_history (6 categories including \"No Info\"), bmi (10.0 to 95.7), HbA1c_level (3.5 to 9.0) and blood_glucose_level (80 to 300), with the target diabetes."));
add(H2("2.3 Data Preprocessing"));
add(BL([
  "**Hidden missing values (Pima):** a 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI is physiologically impossible, so it is converted to NaN. Dropping those rows would lose about half the data.",
  "**Imputation:** missing values are filled with the **median**, learned on the training data only, because Insulin and SkinThickness are right-skewed (Insulin mean 79.8 vs median 30.5).",
  "**Outliers:** kept, because the flagged values are plausible real patients, not errors.",
  "**Split:** 80 / 20 with random_state = 42, stratified for the classification targets (Pima diabetic share 0.349 train vs 0.351 test).",
  "**Pipelines:** every preprocessing step (imputer, then StandardScaler where needed) lives inside a scikit-learn Pipeline, so it is refitted inside each cross-validation fold and never sees validation or test data.",
  "**Scaling:** only for distance- and coefficient-based models (linear, Ridge, Lasso, logistic regression, KNN, SVM); trees and naive Bayes do not need it.",
  "**Kaggle:** 3,854 exact duplicate rows removed; BMI = 27.32 treated as missing; gender and smoking_history one-hot encoded with \"No Info\" kept as its own category.",
  "**Validation:** 5-fold cross-validation with shuffling (stratified for classification) for model selection; the test set is used once at the end.",
]));
add(code([
  "pima[cols_with_zeros] = pima[cols_with_zeros].replace(0, np.nan)",
  "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,",
  "                                                    stratify=y, random_state=42)",
  "pipe = Pipeline([(\"imputer\", SimpleImputer(strategy=\"median\")),",
  "                 (\"scaler\", StandardScaler()), (\"model\", SVC(probability=True))])",
]));
add(H2("2.4 Exploratory Data Analysis (EDA)"));
add(figure(FIG("fig01_pima_missing_zeros.png"), "Impossible zero values per Pima column (hidden missing values)", 5.2));
add(P("Insulin (48.7%) and SkinThickness (29.6%) have large shares of impossible zeros. Left as zeros they would distort every mean and correlation, which is why they are treated as missing."));
add(table("Correlation of each feature with the target",
  ["Pima feature", "r with Outcome", "Kaggle feature", "r with diabetes"],
  [["Glucose", "0.495", "blood_glucose_level", "0.424"], ["BMI", "0.314", "HbA1c_level", "0.406"], ["Insulin", "0.303", "age", "0.265"],
   ["SkinThickness", "0.259", "bmi", "0.215"], ["Age", "0.238", "hypertension", "0.196"], ["Pregnancies", "0.222", "heart_disease", "0.171"]],
  [28, 22, 28, 22]));
add(P("In both datasets blood sugar is by far the strongest single signal, followed by BMI and age. Diabetic Pima patients have a median glucose of 140 against 107, a median BMI of 34.3 against 30.1, and a median age of 36 against 27. For disease progression, BMI (r = 0.586) and s5, the log triglyceride level (r = 0.566), are the strongest predictors."));
add(figure(FIG("fig05_pima_correlation.png"), "Correlation matrix of the Pima features", 4.8));
add(P("No pair of Pima features is so strongly correlated that one is redundant, so all features are kept for the supervised models in Unit 2."));
add(H2("2.5 Proposed System Architecture / Workflow"));
add(figure(FIG("report/workflow_ch2.png"), "Workflow for Units 1 and 2", 6.2));
add(P("Data is loaded and cleaned, analysed, then split before any value is learned from it. All preprocessing sits inside pipelines, models are compared with 5-fold cross-validation, the best are tuned on ROC-AUC, the decision threshold is then lowered to reach the recall target, and the final model is explained with odds ratios, permutation importance and SHAP."));
add(H2("2.6 Algorithm Theory and Mathematical Formulation"));
add(P("Linear regression minimises the squared error; Ridge and Lasso add a penalty with strength λ (alpha):"));
add(eq("Ridge:  min_β  Σᵢ (yᵢ − xᵢᵀβ)² + λ Σⱼ βⱼ²"));
add(eq("Lasso:  min_β  Σᵢ (yᵢ − xᵢᵀβ)² + λ Σⱼ |βⱼ|"));
add(P("Logistic regression models the log-odds of diabetes as a linear function, so each coefficient becomes an odds ratio eᵝ per unit (or per standard deviation after scaling):"));
add(eq("P(y = 1 | x) = σ(β₀ + xᵀβ) = 1 / (1 + e^−(β₀ + xᵀβ))"));
add(P("The SVM with an RBF kernel finds the boundary with the largest margin in a transformed feature space, K(x, x′) = exp(−γ‖x − x′‖²). Decision trees split on single-feature thresholds chosen to reduce Gini impurity; random forests average many deep trees on bootstrap samples; gradient boosting adds small trees one at a time to reduce the remaining error."));
add(P("Metrics: recall = TP / (TP + FN), the share of diabetic patients found (the primary metric, because a missed diabetic costs more than a false alarm); precision = TP / (TP + FP); ROC-AUC, the probability that a random diabetic is ranked above a random non-diabetic; and for regression R², RMSE and MAE."));
add(eq("R² = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²"));
add(H2("2.7 Results and Performance Evaluation"));
add(table("Pima classifiers: 5-fold CV and test ROC-AUC (before tuning)",
  ["Model", "CV recall", "CV ROC-AUC", "Train ROC-AUC", "Test ROC-AUC"],
  [["SVM (RBF)", "0.762", "0.837", "0.909", "0.814"], ["Logistic regression", "0.701", "0.844", "0.852", "0.813"],
   ["Random forest", "0.701", "0.820", "1.000", "0.821"], ["Gradient boosting", "0.598", "0.819", "0.989", "0.831"],
   ["Gaussian naive Bayes", "0.598", "0.828", "0.833", "0.765"], ["KNN", "0.565", "0.784", "0.906", "0.790"],
   ["Decision tree", "0.533", "0.640", "1.000", "0.661"]],
  [32, 17, 17, 17, 17]));
add(P("The SVM had the best cross-validated recall, so it was tuned (C = 1, γ = 0.01, CV ROC-AUC 0.849) and its threshold lowered to 0.30. The single decision tree scores 1.000 on its training data but 0.640 in cross-validation: a clear case of overfitting."));
add(figure(FIG("fig14_clf_threshold.png"), "Recall and precision of the tuned SVM as the decision threshold is lowered", 5.2));
add(P("Lowering the threshold from 0.50 to 0.30 raises cross-validated recall from 0.589 to 0.780 at precision 0.592, which meets the recall target while flagging under half of all patients."));
add(table("Final PBL 1 results on the held-out test sets",
  ["Task", "Final model", "Test result", "Target met?"],
  [["Pima classification", "SVM (RBF), threshold 0.30", "recall 0.796, precision 0.566, ROC-AUC 0.810", "yes (recall ≥ 0.75, ROC-AUC ≥ 0.80)"],
   ["Progression regression", "Lasso (alpha = 0.1)", "R² 0.456, RMSE 53.71, MAE 42.81 (mean baseline R² −0.012)", "yes (R² ≥ 0.40)"],
   ["Kaggle classification, all features", "Random forest (tuned), threshold 0.7", "recall 0.784, precision 0.729, ROC-AUC 0.974", "yes"],
   ["Kaggle classification, no blood tests", "Logistic regression, threshold 0.5", "recall 0.787, precision 0.211, ROC-AUC 0.825", "yes"],
   ["Kaggle regression (BMI)", "Ridge (alpha = 100)", "R² 0.171, RMSE 7.04 (baseline 7.73)", "beats baseline"]],
  [24, 24, 34, 18]));
add(figure(FIG("fig13_clf_test_cm_roc.png"), "Confusion matrix and ROC curve of the final Pima classifier on the test set", 5.6));
add(H2("2.8 Discussion and Findings"));
add(BL([
  "**Why the SVM and not the others:** it had the highest cross-validated recall (0.762) and, after tuning, the highest ROC-AUC (0.849). Logistic regression was close and remains the explainable reference.",
  "**Why the decision tree and KNN were dropped:** the tree overfits (train 1.000 vs CV 0.640) and KNN had the lowest recall among the stable models (0.565).",
  "**Why Lasso for regression:** Ridge, Lasso and linear regression were tied in CV (R² about 0.48); Lasso was kept because it also selects features, and tree models overfitted (random forest train R² 0.920 vs CV 0.398).",
  "**Why HbA1c was dropped as the Kaggle regression target:** it cannot be predicted from the other features (CV R² 0.042), so BMI was predicted instead.",
  "**Why the no-blood-test model matters:** in the Kaggle data the label is largely determined by HbA1c and glucose, so the all-features ROC-AUC (0.974) is partly circular; without them the realistic result is 0.825.",
  "**Risk drivers:** glucose has the largest effect (odds ratio 2.78 per standard deviation), followed by BMI, pregnancies, family history and age. Odds ratios, permutation importance and SHAP agree on this order.",
]));
add(figure(FIG("fig18_permutation_importance.png"), "Permutation importance of the Pima features for the final model", 5.0));
add(H2("2.9 Chapter Conclusion"));
add(P("PBL 1 met all its targets: the SVM finds about four in five diabetic Pima patients (recall 0.796, ROC-AUC 0.810), and Lasso explains 46% of the variation in disease progression. On the larger Kaggle data the realistic screening baseline without blood tests is ROC-AUC 0.825. These numbers, and the risk factors glucose, BMI, age and family history, are the baseline for the later units."));

// CHAPTER 3
add(chapter("UNSUPERVISED LEARNING AND SEQUENTIAL DATA MODELLING (UNITS 3 AND 4)"));
add(H2("3.1 Overview of Allotted Unit 3 and Unit 4 Topics"));
add(P("**Unit 3, Alternate EM:** the alternative view of the Expectation-Maximisation algorithm, applied to a Gaussian mixture model on the existing Pima dataset, with K-means as the hard-assignment version of EM. **Unit 4, Hidden Markov Model:** a Gaussian HMM on a new sequential dataset of blood glucose readings, trained with Baum-Welch and decoded with forward-backward and Viterbi."));
add(H2("3.2 Unit 3 - Dataset Description and Preprocessing"));
add(P("The Pima dataset is described in Section 2.2. Unit 3 adds clustering-specific preprocessing, because a Gaussian mixture is sensitive to missing values, repeated values and scale:"));
add(table("Unit 3 feature selection and transformation",
  ["Feature", "Decision", "Reason"],
  [["Insulin", "dropped", "48.7% missing; filling half a column with one median creates a spike a Gaussian would fit"],
   ["SkinThickness", "dropped", "29.6% missing (rule: drop above 25% missing)"],
   ["Pregnancies", "dropped", "discrete count; on a first try one Gaussian collapsed onto the patients with 0 pregnancies and BIC chose K = 8"],
   ["Outcome", "held out", "unsupervised: used only to check the clusters afterwards"],
   ["DiabetesPedigreeFunction", "log(1 + x)", "skew 1.92 reduced to 1.12"], ["Age", "log(1 + x)", "skew 1.13 reduced to 0.61"],
   ["Glucose, BloodPressure, BMI", "kept, median-imputed", "at most 4.6% missing"]],
  [26, 20, 54]));
add(P("Result: 768 patients by 5 features, standardised so the Euclidean distance weighs them equally. The largest correlation between the five features is 0.34, so none is redundant."));
add(H2("3.3 Unit 3 - Theory, EDA, Implementation and Results"));
add(P("EDA from Chapter 2 is not repeated. The additional check for clustering is the correlation of the five selected features, shown below, and the BIC search in Section 3.6."));
add(figure(FIG("pbl2/fig01_feature_correlation.png"), "Correlation of the five features used for clustering", 4.4));
add(H2("3.4 Unit 3 - Algorithm Theory and Mathematical Formulation"));
add(P("A Gaussian mixture with K components models each patient as coming from one of K groups, with mixing weights πₖ, means μₖ and covariances Σₖ:"));
add(eq("p(x) = Σₖ πₖ N(x | μₖ, Σₖ)"));
add(P("The group label z of each patient is hidden. With complete data (X, Z) fitting would be easy; with only X, the log-likelihood is a log of a sum with no closed-form maximum. EM instead maximises the expected complete-data log-likelihood Q(θ, θᵒˡᵈ), alternating two steps:"));
add(eq("E-step:  γₙₖ = πₖ N(xₙ | μₖ, Σₖ) / Σⱼ πⱼ N(xₙ | μⱼ, Σⱼ)"));
add(eq("M-step:  Nₖ = Σₙ γₙₖ,  πₖ = Nₖ / N,  μₖ = Σₙ γₙₖ xₙ / Nₖ"));
add(P("Each iteration never decreases the likelihood, and EM stops at a local maximum, so it is restarted from 10 random starts. Forcing γ to 0 or 1 gives K-means, the hard version of EM. The number of components is chosen with the Bayesian Information Criterion:"));
add(eq("BIC = −2 ln L + p ln N"));
add(H2("3.5 Unit 3 Proposed System Architecture / Workflow"));
add(figure(FIG("report/workflow_ch3_unit3.png"), "Workflow for Unit 3", 6.2));
add(H2("3.6 Unit 3 - Results and Performance Evaluation"));
add(P("BIC is lowest for diagonal covariance with K = 3 (10422.4); K = 4 is practically tied (10423.5), so BIC keeps the simpler model. Full covariance wins only at K = 2."));
add(figure(FIG("pbl2/fig03_em_steps.png"), "EM run one step at a time: log-likelihood and the clusters after steps 1, 3 and 60", 6.0));
add(P("The average log-likelihood climbs from −7.094 to −6.646 over 60 steps with zero decreases, which confirms the monotonic property. Our hand-written E-step matches scikit-learn's predict_proba to 1.3 × 10⁻¹⁵, and the final model converged after 25 steps."));
add(table("Three risk profiles found without the diagnosis",
  ["Cluster", "Patients", "Diabetic", "Profile (medians)"],
  [["Low risk", "269", "10.8%", "age 23, glucose 102, BMI 29.7, BP 65"],
   ["Older", "237", "38.0%", "age 38, glucose 125, lowest family history (−0.68 SD)"],
   ["High risk", "262", "56.9%", "glucose 130, BMI 34.6, highest family history (+0.73 SD)"]],
  [18, 14, 14, 54]));
add(table("Soft (GMM) versus hard (K-means) clustering",
  ["Metric", "GMM (soft EM)", "K-means (hard EM)"],
  [["Silhouette (internal)", "0.143", "0.216"], ["Adjusted Rand Index vs diagnosis (external)", "0.101", "0.185"],
   ["Gives probabilities", "yes", "no"], ["Borderline patients (no group above 0.8)", "30.6% (235)", "not available"]],
  [50, 25, 25]));
add(figure(FIG("pbl2/fig05_clusters_pca.png"), "GMM clusters, K-means clusters and the real diagnosis on two PCA components (56.1% of variance)", 6.0));
add(P("K-means scores higher on silhouette because silhouette favours round, equal-sized clusters, which is exactly what K-means produces. Only the GMM can show that 30.6% of patients sit between groups. In 9 of 10 single random starts EM stopped at a worse local maximum (−6.670 vs −6.647), which is why 10 restarts are used."));
add(H2("3.7 Unit 4 - Sequential Dataset Description"));
add(table("UCI Diabetes (AIM-94) sequential dataset",
  ["Property", "Value"],
  [["Source", "UCI Machine Learning Repository; Michael Kahn, Washington University, St. Louis (AIM-94)"],
   ["Patients and period", "70 insulin-dependent (Type 1) patients, March 1988 to September 1991"],
   ["Size", "70 files, 29,330 rows by 4 fields: date, time, code, value"],
   ["Sequence identifier", "patient file (data-01 to data-70)"],
   ["Event codes", "insulin doses (33 to 35): 14,401; glucose readings (48, 57 to 64): 13,531; hypoglycaemic symptoms: 331; meals: 514; exercise: 305; special events: 94; undocumented codes: 154"],
   ["Sampling", "irregular; median gap 6 hours, mostly before meals; paper records use fixed times (08:00, 12:00, 18:00, 22:00)"],
   ["Observation variable", "blood glucose (mg/dL)"],
   ["Limitations", "irregular gaps, approximate paper-record times, old data from one clinic"]],
  [26, 74]));
add(H2("3.8 Unit 4 - Sequential Preprocessing and EDA"));
add(BL([
  "Kept only glucose codes; removed 13 rows with unreadable values or dates (for example 06-31-1991) and 3 readings below 20 mg/dL, leaving **13,515 readings from 70 patients**.",
  "Sorted each patient's readings by time; **temporal order is never shuffled**.",
  "Kept the **68 patients with at least 50 readings** and split **by patient**: 54 for training, 14 for testing, so no patient appears in both.",
  "All readings are stacked with a lengths array, so the model never learns a false transition between two different patients.",
  "Median of 120 readings per patient (31 to 616); median reading 149 mg/dL, a quarter above 210.",
  "**Order matters:** lag-1 autocorrelation within patients is 0.186 and drops to 0.0 when the order is shuffled.",
]));
add(figure(FIG("pbl3/fig02_example_sequence.png"), "One patient's glucose readings over four weeks, against the 70 to 180 mg/dL target band", 6.0));
add(H2("3.9 Unit 4 - Mathematical Formulation"));
add(P("A Hidden Markov Model λ = (π, A, B) has hidden states zₜ (here: low, target and high glucose regimes) that follow a first-order Markov chain, and observations xₜ (glucose readings) that depend only on the current state:"));
add(eq("aᵢⱼ = P(zₜ = j | zₜ₋₁ = i),   xₜ | zₜ = j ~ N(μⱼ, σⱼ²)"));
add(P("The forward algorithm computes the likelihood of a sequence; Viterbi finds the most likely state path by replacing the sum with a maximum and keeping back-pointers; forward-backward gives the smoothed state probabilities γₜ(i); Baum-Welch is EM for HMMs, using forward-backward in the E-step. All recursions run in log space with log-sum-exp to avoid underflow."));
add(eq("αₜ(j) = [Σᵢ αₜ₋₁(i) aᵢⱼ] bⱼ(xₜ),   P(x₁:T) = Σⱼ α_T(j)"));
add(eq("δₜ(j) = maxᵢ [δₜ₋₁(i) aᵢⱼ] bⱼ(xₜ)"));
add(P("The stationary distribution π* = π*A gives the long-run share of time in each state, and the expected time spent in state i is 1 / (1 − aᵢᵢ)."));
add(H2("3.10 Unit 4 Proposed System Architecture / Workflow"));
add(figure(FIG("report/workflow_ch3_unit4.png"), "Workflow for Unit 4", 6.2));
add(H2("3.11 Results and Performance Evaluation"));
add(P("BIC keeps falling until K = 5, but from K = 4 the extra states overlap (means 180, 191 and 211 mg/dL) and differ only in spread. K = 3 gives clearly separated states that match the clinical bands, so K = 3 was chosen for interpretability, and this trade-off is stated openly. Baum-Welch converged after 136 rounds with the best of 5 random starts."));
add(table("Learned states",
  ["State", "Mean (SD) mg/dL", "Stay probability aᵢᵢ", "Expected duration (readings)"],
  [["Low", "74.8 (24.1)", "0.43", "1.75"], ["Target", "126.1 (37.8)", "0.97", "33.7"], ["High", "205.5 (76.6)", "0.79", "4.7"]],
  [22, 26, 26, 26]));
add(table("Learned transition matrix A (rows sum to 1)",
  ["From \\ To", "Low", "Target", "High"],
  [["Low", "0.43", "0.01", "0.56"], ["Target", "0.00", "0.97", "0.03"], ["High", "0.20", "0.01", "0.79"]],
  [25, 25, 25, 25]));
add(P("A low reading is usually followed by a high one (0.56), the known rebound after hypoglycaemia. Target is a stable regime that Low and High rarely jump into. The stationary distribution is 0.188 low, 0.280 target and 0.532 high, close to the share of training readings Viterbi assigns to each state (0.208, 0.304, 0.487)."));
add(P("**Verification:** our hand-written forward algorithm gives exactly the library's log-likelihood (−600.2392) for a test patient, and our Viterbi gives the identical 104-step path."));
add(table("Unit 4 evaluation on 14 held-out patients",
  ["Measure", "Hidden Markov Model", "Gaussian mixture (no time)", "Single Gaussian / baselines"],
  [["Test log-likelihood per reading (higher is better)", "−5.6847", "−5.7313", "−5.7861"],
   ["Next-reading forecast MAE, mg/dL (lower is better)", "60.1", "", "training mean 62.6; last reading 74.0"]],
  [34, 20, 22, 24]));
add(figure(FIG("pbl3/fig05_viterbi_posterior.png"), "Viterbi path and smoothed state probabilities for test patient 48", 6.0));
add(P("Test patient 48 is never decoded as Target: mostly High with repeated drops to Low, a pattern a single average would hide. Across test patients, 63.0% of pre-breakfast readings are High against 49.4% before lunch (consistent with the dawn phenomenon), 29.9% of pre-snack readings are Low, and 8 of 14 patients never reach Target."));
add(H2("3.12 Discussion, Findings and Limitations"));
add(BL([
  "**Unit 3:** EM rediscovers clinically sensible risk groups without the label, and its soft assignments expose borderline patients that hard clustering hides. Limitations: Gaussian cluster shapes, local optima, one population of women.",
  "**Unit 4:** the states are interpretable and the time order measurably helps on new patients, because the HMM and the GMM share the same Gaussian emissions and differ only by the transition matrix. Limitations: one step equals one reading regardless of the gap, glucose only (no insulin or meals as inputs), approximate paper-record times and old data.",
  "**Validation:** Unit 3 is checked against the held-out diagnosis only after fitting; Unit 4 is validated on patients never seen in training, never on shuffled readings.",
]));
add(H2("3.13 Chapter Conclusion"));
add(P("Unit 3 showed that the existing Pima data contains three risk groups, from 10.8% to 56.9% diabetic, found without labels. Unit 4 showed, on a separate sequential dataset, that blood glucose moves between three hidden regimes with medically known patterns, and that modelling time improves the fit on new patients. The two units use different datasets and different tasks, so their performance values are not compared directly."));

// CHAPTER 4
add(chapter("ADVANCED PREDICTIVE MODELLING AND ENSEMBLE LEARNING (UNIT 5)"));
add(H2("4.1 Overview of Allotted Unit 5 Topics"));
add(P("Unit 5 combines models. **Boosting** trains weak models one after another, each focusing on the patients the previous ones got wrong, and combines them in a weighted vote; we implement AdaBoost with decision stumps (one-split CART trees). **Bayesian model averaging (BMA)** keeps many candidate models and weights each by its posterior probability. A single CART tree and a random forest (bagging) are the baselines, and gradient boosting is included as boosting with the cross-entropy error."));
add(table("Three ways of combining models",
  ["", "Bagging (random forest)", "Boosting (AdaBoost)", "Bayesian model averaging"],
  [["Training", "in parallel, bootstrap samples", "in sequence, on reweighted data", "each candidate on all data"],
   ["Combined by", "equal vote", "weighted vote, αₘ from error", "posterior weight p(M | D)"],
   ["Mainly reduces", "variance", "bias", "model uncertainty"],
   ["With more data", "keeps all members", "keeps all members", "weight moves to one model"]],
  [19, 27, 27, 27]));
add(H2("4.2 Dataset Description and Preprocessing"));
add(P("The Kaggle dataset is described in Section 2.2 and reused with the same cleaning and the same stratified 80 / 20 split (76,916 training and 19,230 test patients, 8.82% diabetic in both), so results compare directly with PBL 1. Additional Unit 5 preprocessing: BMI placeholders are filled with the training median (26.50); gender and smoking_history are one-hot encoded, giving 15 columns; trees need no scaling, while the logistic regressions used in BMA are standardised. The rounded HbA1c and glucose values that ruled this dataset out for the Gaussian models of Units 3 and 4 are no problem for trees, which only need a threshold between two values."));
add(H2("4.3 EDA and Feature Analysis"));
add(P("The additional analysis relevant to boosting is how the label depends on the two blood tests, because a stump splits on exactly one such threshold."));
add(table("The label is nearly a rule on HbA1c and glucose",
  ["", "HbA1c", "Glucose"],
  [["Values where every patient is diabetic", "6.8, 7.0, 7.5, 8.2, 8.8, 9.0", "220, 240, 260, 280, 300"],
   ["Highest value in a non-diabetic patient", "6.6", "200"],
   ["Rule: HbA1c ≥ 6.8 or glucose ≥ 220", "5,671 patients, 100% diabetic", "catches 66.9% of all diabetic patients"]],
  [36, 32, 32]));
add(P("The data appears to have been generated with hard cut-offs close to the clinical diagnosis lines (HbA1c 6.5%, glucose 200 mg/dL). Any model that sees these two features gets two thirds of the diabetic patients almost for free, and the label is partly defined by them. Section 4.8 therefore repeats the comparison without blood tests."));
add(figure(FIG("pbl4/fig06_feature_importance.png"), "Feature importance of AdaBoost and the random forest", 5.0));
add(H2("4.4 Proposed Workflow / Architecture"));
add(figure(FIG("report/workflow_ch4.png"), "Workflow for Unit 5", 6.2));
add(H2("4.5 Algorithm Theory and Mathematical Formulation"));
add(P("**AdaBoost.** Labels tₙ ∈ {−1, +1}; all weights start at wₙ = 1/N. In each round m a stump yₘ is fitted with weights w, and:"));
add(eq("εₘ = Σₙ wₙ I(yₘ(xₙ) ≠ tₙ) / Σₙ wₙ"));
add(eq("αₘ = ln((1 − εₘ) / εₘ)"));
add(eq("wₙ ← wₙ exp(αₘ I(yₘ(xₙ) ≠ tₙ)),   Y(x) = sign(Σₘ αₘ yₘ(x))"));
add(P("This is the form in Bishop and in scikit-learn. AdaBoost greedily minimises the exponential error, which also bounds the training error, and whose minimiser is half the log-odds, so the vote converts to a probability:"));
add(eq("E = Σₙ exp(−tₙ fₘ(xₙ)),   fₘ = ½ Σₗ αₗ yₗ,   E/N = Πₘ 2√(εₘ(1 − εₘ))"));
add(eq("p(t = 1 | x) = σ(Σₘ αₘ yₘ(x))"));
add(P("**Bayesian model averaging.** For candidate models M₁ … M_K, with the evidence approximated by BIC and an equal prior:"));
add(eq("p(t | x, D) = Σₖ p(t | x, Mₖ) p(Mₖ | D),   p(Mₖ | D) ∝ exp(−½ BICₖ)"));
add(P("The posterior inclusion probability of a feature is the total posterior weight of the models that contain it."));
add(table("Parameter settings and model-selection decisions",
  ["Model", "Settings", "Reason"],
  [["Decision stump", "DecisionTreeClassifier(max_depth=1)", "the textbook weak learner; each round readable"],
   ["CART tree", "max_depth = 6", "single-tree baseline"],
   ["Random forest", "300 trees, min_samples_leaf = 5 (and 50 in the noise test)", "bagging baseline"],
   ["AdaBoost", "500 stumps, learning rate 1.0", "test ROC-AUC still rising slowly at 500; no overfitting"],
   ["Gradient boosting", "HistGradientBoostingClassifier, defaults", "boosting with the cross-entropy error"],
   ["BMA candidates", "256 logistic regressions (every subset of 8 feature groups), C = 1e6", "BIC needs a likelihood and a clear parameter count; C = 1e6 gives maximum-likelihood fits"],
   ["Validation", "same test split as PBL 1; stratified 5-fold CV on all 96,146 patients", "check the ranking is not luck of one split"]],
  [20, 42, 38]));
add(H2("4.6 Results and Performance Evaluation"));
add(table("First AdaBoost rounds (all features)",
  ["Round m", "Stump splits on", "εₘ", "αₘ"],
  [["1", "HbA1c ≤ 6.70", "0.048", "2.98"], ["2", "HbA1c ≤ 5.35", "0.280", "0.94"], ["3", "glucose ≤ 210", "0.222", "1.26"],
   ["4", "age ≤ 41.5", "0.257", "1.06"], ["5", "glucose ≤ 113", "0.364", "0.56"], ["8", "glucose ≤ 113", "0.423", "0.31"]],
  [16, 40, 22, 22]));
add(P("The first stumps sit exactly in the gaps of the rule in Table 4.2, and the weighted error rises towards 0.5 as each new stump is given the patients the earlier ones got wrong."));
add(P("**Verification:** our hand-written AdaBoost (only the stump fitting is borrowed) reproduces scikit-learn on all 500 of 500 stumps, αₘ and εₘ agree to 4.4 × 10⁻¹⁵, and the test predictions are identical for all 19,230 patients. The exponential error equals Πₘ 2√(εₘ(1 − εₘ)) to 4.4 × 10⁻¹⁶, falling from 0.428 to 0.161, while the training error falls from 4.8% to 2.9% and always stays below it."));
add(figure(FIG("pbl4/fig02_weak_to_strong.png"), "From one stump to 500: test ROC-AUC and training and test error by round", 6.0));
add(P("Test ROC-AUC rises from 0.737 (one stump) to 0.952 after 5 rounds and 0.977 after 500, and the test error (2.88%) stays close to the training error (2.93%), so there is no overfitting."));
add(table("Which patients boosting focuses on (after 500 rounds)",
  ["", "Heaviest 1%", "Other diabetic", "Non-diabetic"],
  [["Patients", "774", "6,012", "70,130"], ["Median HbA1c", "6.1", "6.8", "5.8"], ["Median glucose", "155", "200", "140"],
   ["Below both clinical cut-offs", "63.6%", "15.8%", "76.8%"]],
  [34, 22, 22, 22]));
add(P("The heaviest 1% of patients hold 28.4% of all the weight and every one is diabetic: boosting concentrates on diabetic patients whose lab values look almost normal, which are exactly the hard cases for screening."));
add(figure(FIG("pbl4/fig04_error_functions_calibration.png"), "Error functions, and calibration of the library's predict_proba against σ(vote)", 6.0));
add(P("The library's predict_proba squeezes probabilities towards 0.5 (log loss 0.436, Brier 0.129). Converting the vote with σ(vote), as the theory says, gives log loss 0.083 and Brier 0.024, with the calibration curve on the diagonal and the same ROC-AUC."));
add(table("Test-set comparison, all features",
  ["Model", "ROC-AUC", "Avg precision", "Log loss", "Recall @0.5", "Precision @0.5"],
  [["One stump", "0.737", "0.521", "0.186", "0.475", "1.000"], ["CART tree, depth 6", "0.962", "0.825", "0.093", "0.678", "1.000"],
   ["Random forest", "0.970", "0.871", "0.093", "0.680", "0.997"], ["AdaBoost, 500 stumps", "0.977", "0.879", "0.083", "0.689", "0.978"],
   ["Gradient boosting", "0.977", "0.881", "0.083", "0.693", "0.978"], ["BMA, 256 logistic models", "0.960", "0.817", "0.117", "0.640", "0.871"]],
  [30, 14, 14, 14, 14, 14]));
add(table("Robustness to wrong training labels (test ROC-AUC on clean labels)",
  ["Labels flipped", "AdaBoost", "Random forest, leaf 5", "Random forest, leaf 50", "Gradient boosting"],
  [["0%", "0.977", "0.970", "0.969", "0.977"], ["5%", "0.969", "0.955", "0.966", "0.974"], ["10%", "0.968", "0.947", "0.965", "0.972"]],
  [20, 20, 20, 20, 20]));
add(P("With 5% of labels flipped the flipped patients end up holding 36.2% of AdaBoost's weight, more than 7 times their share, which confirms that the exponential error chases wrong labels. The test damage is small because a stump cannot memorise one patient. The forest's loss depends on its leaf size (0.023 with leaves of 5, 0.004 with leaves of 50), so robustness here is about model capacity rather than bagging versus boosting. Three further random flips reproduce these values to within 0.002."));
add(table("Bayesian model averaging by training size (mean of 10 subsets)",
  ["Training patients", "Top model weight", "Models for 95% of weight", "Log loss, BMA", "Log loss, best model", "Log loss, equal average"],
  [["200", "0.47", "8.1", "0.159", "0.234", "0.166"], ["500", "0.51", "6.1", "0.127", "0.131", "0.159"],
   ["1,000", "0.54", "4.7", "0.123", "0.124", "0.158"], ["5,000", "0.77", "2.4", "0.119", "0.119", "0.157"],
   ["76,916 (all)", "1.000", "1", "0.117", "0.117", "0.157"]],
  [18, 16, 18, 16, 16, 16]));
add(figure(FIG("pbl4/fig07_bma_sample_size.png"), "BMA, best single model and equal average by training size, and the posterior weight of the best model", 6.0));
add(P("On all 76,916 patients the full model gets posterior weight 1.000 (the next best, without gender, is 32 BIC units worse, weight 1.1 × 10⁻⁷), so BMA reduces to model selection. With 200 patients about 8 models share the weight and BMA gives clearly better probabilities than the single best model (0.159 vs 0.234). Equal averaging is worst at every size. With 200 patients the inclusion probabilities are glucose 1.00, HbA1c 0.99, age 0.52, BMI 0.40, heart disease 0.29, hypertension 0.18, gender 0.02 and smoking 0.00."));
add(H2("4.7 Comparison with Previous Models"));
add(P("Unit 5 uses the same task, the same test split and the same metric (ROC-AUC) as the PBL 1 Kaggle models, so the comparison is direct. 5-fold cross-validation on all 96,146 patients checks that the ranking is not luck of one split."));
add(table("Comparison with PBL 1 on the same Kaggle task",
  ["Model", "Test ROC-AUC, all features", "5-fold CV, all features", "5-fold CV, no blood tests"],
  [["PBL 1 random forest (tuned)", "0.974", "", ""], ["PBL 1 logistic regression, no blood tests", "", "", "0.825 (test)"],
   ["Logistic regression", "0.960", "0.961 ± 0.005", "0.830 ± 0.006"], ["CART tree, depth 6", "0.962", "0.963 ± 0.004", "0.820 ± 0.005"],
   ["Random forest", "0.970", "0.972 ± 0.003", "0.810 ± 0.005"], ["AdaBoost, 500 stumps", "0.977", "0.978 ± 0.002", "0.827 ± 0.005"],
   ["Gradient boosting", "0.977", "0.978 ± 0.003", "0.832 ± 0.005"]],
  [34, 22, 22, 22]));
add(P("With all features, boosting improves on the tuned PBL 1 forest (0.977 vs 0.974) and beats the forest in all 5 cross-validation folds (by 0.006 on average); AdaBoost and gradient boosting are tied. Without blood tests the gain almost disappears: gradient boosting leads logistic regression in every fold, but only by 0.002, which is smaller than the fold-to-fold spread."));
add(H2("4.8 Discussion, Findings and Limitations"));
add(figure(FIG("pbl4/fig09_no_blood_tests.png"), "Test ROC-AUC of every model with and without the blood tests", 6.0));
add(P("Without HbA1c and glucose every model loses about 0.15 of ROC-AUC (AdaBoost 0.977 to 0.820, random forest 0.970 to 0.804) and average precision falls from about 0.88 to about 0.30. Ensembles win clearly when the signal has sharp thresholds; without blood tests risk rises smoothly with age (AdaBoost importance 0.758) and BMI (0.146), which a linear model already captures. For screening before a blood test, the explainable logistic regression is the sensible choice."));
add(table("Choosing a screening threshold (test set)",
  ["Model", "Threshold", "Recall", "Precision", "Share flagged"],
  [["No blood tests: logistic regression", "0.50", "0.054", "0.487", "1.0%"], ["", "0.10", "0.754", "0.220", "30.3%"],
   ["", "0.08", "0.808", "0.198", "36.0%"], ["All features: AdaBoost, σ(vote)", "0.50", "0.689", "0.978", "6.2%"],
   ["", "0.20", "0.795", "0.718", "9.8%"]],
  [36, 14, 16, 16, 18]));
add(BL([
  "**Interpretability:** the hand-checked AdaBoost is transparent round by round, and BMA's inclusion probabilities state how sure the evidence is about each feature; the logistic regression keeps odds ratios.",
  "**Training cost:** 500 stumps fit in about 13 seconds, the random forest and gradient boosting in a few seconds, and the full 5-fold comparison in under 3 minutes on a laptop.",
  "**Limitations:** one dataset whose label is nearly a rule on two inputs; ensembles at default settings; 22.6% of training BMI values are a filled-in placeholder, which weakens BMI's measured importance; BIC is a large-sample approximation with an equal prior over models.",
]));
add(H2("4.9 Chapter Conclusion"));
add(P("Unit 5 contributed the strongest classifier of the project (ROC-AUC 0.977 with all features), a verified implementation of AdaBoost, calibrated risk probabilities from the theory of the exponential error, and the lesson that model averaging helps only when data is scarce. It also showed, honestly, that before a blood test the simple logistic regression is as good as any ensemble (0.830 vs 0.832 in cross-validation)."));

// CHAPTER 5
add(chapter("INTEGRATED DISCUSSION AND FINDINGS"));
add(P("This chapter compares the datasets and the models across all five units. Performance values are compared only where the task, data and metric are the same; otherwise the comparison is about purpose, strengths and weaknesses."));
add(H2("5.1 Dataset Comparison"));
add(table("Every dataset: why it was used, and why it was not used elsewhere",
  ["Dataset", "Size used", "Used for", "Why chosen", "Why not used elsewhere"],
  [["Pima Indians Diabetes", "768 x 9 (Unit 3: 768 x 5)", "Units 1 and 2 classification; Unit 3 clustering", "continuous clinical values; a label that can be hidden and checked afterwards", "one row per patient, so no time order for Unit 4"],
   ["Diabetes Progression", "442 x 11", "Unit 2 regression", "continuous target (progression after one year)", "no diabetes label; small"],
   ["Kaggle Diabetes Prediction", "96,146 x 8 (15 after one-hot)", "Units 1 and 2 classification and BMI regression; Unit 5 ensembles", "large and labelled; trees handle rounded values", "HbA1c and glucose rounded to 18 values each break Gaussian models (Units 3 and 4)"],
   ["UCI Diabetes (AIM-94)", "13,515 readings, 68 patients", "Unit 4 HMM", "readings in time order, about 120 per patient", "no diagnosis label; one variable"]],
  [18, 17, 21, 22, 22]));
add(H2("5.2 Model Comparison"));
add(table("Every model used: purpose, result, strengths and weaknesses",
  ["Model", "Unit", "Learning type", "Key result", "Strength", "Weakness"],
  [["Linear / Ridge / Lasso regression", "2", "supervised regression", "Lasso R² 0.456 (progression); Ridge R² 0.171 (BMI)", "coefficients, feature selection (Lasso)", "linear only"],
   ["Logistic regression", "2, 5", "supervised classification", "Pima test 0.813; Kaggle no blood tests 0.825 to 0.830", "odds ratios, calibrated", "misses sharp thresholds"],
   ["SVM (RBF)", "2", "supervised classification", "Pima final recall 0.796, ROC-AUC 0.810", "best Pima recall", "no coefficients to explain"],
   ["KNN, naive Bayes", "2", "supervised classification", "Pima test 0.790 and 0.765", "simple", "lowest recall; not chosen"],
   ["CART decision tree", "2, 5", "supervised classification", "Pima 0.661; Kaggle 0.962", "readable rules", "unstable, overfits small data"],
   ["Random forest", "2, 5", "ensemble (bagging)", "Kaggle 0.974 tuned; CV 0.972", "strong with little tuning", "small leaves memorise wrong labels"],
   ["Gradient boosting", "2, 5", "ensemble (boosting)", "Kaggle CV 0.978; no blood tests 0.832", "top CV score, most noise-robust", "least transparent"],
   ["AdaBoost (stumps)", "5", "ensemble (boosting)", "Kaggle 0.977; CV 0.978", "verified by hand; calibrated via σ(vote)", "chases wrong labels"],
   ["K-means", "3", "unsupervised, hard EM", "silhouette 0.216, ARI 0.185", "compact groups", "no probabilities"],
   ["Gaussian mixture (EM)", "3", "unsupervised, soft EM", "10.8 / 38.0 / 56.9% diabetic; 30.6% borderline", "probabilities, BIC picks K", "Gaussian shape, local optima"],
   ["Hidden Markov Model", "4", "sequential, EM (Baum-Welch)", "test log-likelihood −5.685 vs −5.731 without time", "uses time order, readable states", "equal time steps, glucose only"],
   ["Bayesian model averaging", "5", "model combination", "0.960; at 200 patients log loss 0.159 vs 0.234", "honest uncertainty", "one model wins with large data"]],
  [17, 7, 15, 23, 19, 19]));
add(H2("5.3 What Was Dropped, and Why"));
add(table("Design decisions across the project",
  ["Dropped or changed", "Where", "Why"],
  [["Insulin, SkinThickness", "Unit 3", "48.7% and 29.6% missing; imputing half a column creates an artificial spike"],
   ["Pregnancies", "Unit 3", "discrete count; a Gaussian collapsed onto zero pregnancies and BIC chose K = 8"],
   ["Full covariance", "Unit 3", "more parameters; BIC preferred diagonal at K = 3"],
   ["HbA1c as regression target", "Unit 2 (Kaggle)", "unpredictable from the other features (CV R² 0.042); BMI predicted instead"],
   ["Decision tree, KNN, naive Bayes as final models", "Unit 2", "overfitting (tree) or lowest recall"],
   ["Threshold 0.5", "Units 2 and 5", "too high with few positives; 0.30 (Pima) and 0.08 (Kaggle screening) reach the recall targets"],
   ["K = 5 HMM states", "Unit 4", "overlapping states; K = 3 matches the clinical bands"],
   ["Insulin, meals, exercise codes", "Unit 4", "a plain HMM has no inputs; listed as future work"],
   ["Duplicates and BMI 27.32", "Units 2 and 5", "duplicate rows and a placeholder value, not measurements"],
   ["HbA1c and glucose", "Unit 5 (screening)", "they nearly define the label; removed to measure realistic screening"]],
  [30, 18, 52]));
add(H2("5.4 Critical Findings"));
add(BL([
  "**The same risk factors appear in every method:** blood sugar (glucose, HbA1c), then BMI and age, with family history in Pima. Odds ratios, permutation importance, SHAP, the GMM profiles, AdaBoost importance and BMA inclusion probabilities agree.",
  "**Complexity pays only with sharp signals:** with HbA1c and glucose, boosting beats single models and bagging; without them, logistic regression is within 0.002 of the best ensemble.",
  "**Probabilities matter as much as rankings:** σ(vote) for AdaBoost and BMA at small sample sizes both improve probability quality without changing the ranking.",
  "**EM is the common thread:** K-means, the Gaussian mixture and Baum-Welch for the HMM are all forms of the EM algorithm, and each was verified against hand-written code.",
  "**Study limitations:** public, partly old or synthetic-looking data; one population in Pima; a label that nearly follows two inputs in Kaggle; associations, not causes.",
]));

// CHAPTER 6
add(chapter("OVERALL CONCLUSION AND FUTURE SCOPE"));
add(P("The project met each objective with evidence. PBL 1 reached test recall 0.796 and ROC-AUC 0.810 on Pima and R² 0.456 for disease progression. Unit 3 found three risk groups without labels (10.8% to 56.9% diabetic) and exposed 30.6% borderline patients. Unit 4 learned three glucose regimes from 13,515 readings and showed that time order improves the fit on new patients. Unit 5 built a verified AdaBoost that reaches ROC-AUC 0.977, showed when Bayesian model averaging helps, and established that for screening before a blood test a simple logistic regression (0.830 in cross-validation) is the right choice."));
add(H2("6.1 Future Scope"));
add(BL([
  "**Better data:** a recent, multi-centre dataset with real measurement values instead of rounded or placeholder ones.",
  "**Tuning:** tune the random forest and gradient boosting and use early stopping for AdaBoost under label noise.",
  "**Sequential model:** an input-output HMM that uses insulin, meals and exercise, and a continuous-time model for irregular gaps.",
  "**Deployment:** a simple screening form that returns a risk, the threshold decision and the main contributing factors.",
  "**Interpretability:** generalised additive models (Unit 5) to give one risk curve per feature.",
]));

// REFERENCES
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "REFERENCES", font: FONT })] }));
[
  "C. M. Bishop, Pattern Recognition and Machine Learning. New York, NY, USA: Springer, 2006.",
  "A. P. Dempster, N. M. Laird, and D. B. Rubin, \"Maximum likelihood from incomplete data via the EM algorithm,\" J. Roy. Statist. Soc. B, vol. 39, no. 1, pp. 1-38, 1977.",
  "L. R. Rabiner, \"A tutorial on hidden Markov models and selected applications in speech recognition,\" Proc. IEEE, vol. 77, no. 2, pp. 257-286, 1989.",
  "Y. Freund and R. E. Schapire, \"A decision-theoretic generalization of on-line learning and an application to boosting,\" J. Comput. Syst. Sci., vol. 55, no. 1, pp. 119-139, 1997.",
  "J. A. Hoeting, D. Madigan, A. E. Raftery, and C. T. Volinsky, \"Bayesian model averaging: a tutorial,\" Statistical Science, vol. 14, no. 4, pp. 382-417, 1999.",
  "G. Schwarz, \"Estimating the dimension of a model,\" The Annals of Statistics, vol. 6, no. 2, pp. 461-464, 1978.",
  "J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler, and R. S. Johannes, \"Using the ADAP learning algorithm to forecast the onset of diabetes mellitus,\" in Proc. Annu. Symp. Comput. Appl. Med. Care, 1988, pp. 261-265.",
  "B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, \"Least angle regression,\" The Annals of Statistics, vol. 32, no. 2, pp. 407-499, 2004.",
  "M. Kahn, \"Diabetes data set (AIM-94),\" UCI Machine Learning Repository, 1994.",
  "\"Diabetes prediction dataset,\" Kaggle. [Online]. Available: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset",
  "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, pp. 2825-2830, 2011.",
  "American Diabetes Association, \"Diagnosis and classification of diabetes: Standards of care in diabetes,\" Diabetes Care, 2024.",
  "R. Miotto, L. Wang, S. Wang, X. Jiang, and J. T. Dudley, \"Deep learning for healthcare: review, opportunities and challenges,\" Briefings in Bioinformatics, vol. 19, no. 6, pp. 1236-1246, 2018.",
].forEach((r, i) => body.push(P(`[${i + 1}] ${r}`)));

// APPENDIX
body.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: "APPENDIX A: CODING", font: FONT })] }));
body.push(P("The complete Python implementation is in five Jupyter notebooks in the project repository (https://github.com/aggamsingh/diabetes-risk). Each opens directly in Google Colab; run the first cell after cloning the repository so the data files in data/ are available. Key excerpts follow."));
const nbs = [["Chapter 2, PBL 1 (Pima and Progression)", "diabetes_risk.ipynb"], ["Chapter 2, PBL 1 (Kaggle)", "diabetes_risk_kaggle.ipynb"],
  ["Chapter 3, Unit 3 (Alternate EM)", "pbl2_alternate_em.ipynb"], ["Chapter 3, Unit 4 (Hidden Markov Model)", "pbl3_hmm.ipynb"],
  ["Chapter 4, Unit 5 (Boosting and BMA)", "pbl4_boosting_bma.ipynb"]];
nbs.forEach(([t, nb]) => body.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [
  new TextRun({ text: `${t}: `, bold: true, font: FONT }),
  new ExternalHyperlink({ link: COLAB(nb), children: [new TextRun({ text: COLAB(nb), style: "Hyperlink", font: FONT, size: 20 })] })] })));
body.push(H2("Chapter 2: Implementation Methodology and Algorithm Configuration"));
body.push(...code([
  "def make_scaled_pipeline(model):",
  "    return Pipeline([(\"imputer\", SimpleImputer(strategy=\"median\")),",
  "                     (\"scaler\", StandardScaler()), (\"model\", model)])",
  "",
  "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
  "grid = GridSearchCV(make_scaled_pipeline(SVC(probability=True)),",
  "                    {\"model__C\": [0.1, 1, 10], \"model__gamma\": [0.001, 0.01, 0.1]},",
  "                    scoring=\"roc_auc\", cv=cv).fit(X_train, y_train)",
  "proba = grid.predict_proba(X_test)[:, 1]",
  "y_pred = (proba >= 0.30).astype(int)          # threshold lowered for recall",
]));
body.push(H2("Chapter 3: Unit 3 Proposed Workflow and Implementation Methodology"));
body.push(...code([
  "gmm = GaussianMixture(n_components=3, covariance_type=\"diag\", n_init=10,",
  "                      max_iter=500, random_state=42).fit(X)",
  "# hand-written E-step, compared with the library",
  "dens = np.column_stack([gmm.weights_[k] * multivariate_normal.pdf(",
  "        X, gmm.means_[k], np.diag(gmm.covariances_[k])) for k in range(3)])",
  "resp = dens / dens.sum(axis=1, keepdims=True)",
  "print(np.abs(resp - gmm.predict_proba(X)).max())   # 1.3e-15",
]));
body.push(H2("Chapter 3: Unit 4 Proposed Workflow and Implementation Methodology"));
body.push(...code([
  "hmm = GaussianHMM(n_components=3, covariance_type=\"diag\", n_iter=200,",
  "                  tol=1e-4, random_state=seed).fit(X_train, lengths_train)",
  "def forward(x, model):                           # log-space forward algorithm",
  "    log_b = norm.logpdf(x, means, stds)",
  "    log_a = np.log(model.startprob_) + log_b[0]",
  "    for t in range(1, len(x)):",
  "        log_a = logsumexp(log_a[:, None] + np.log(model.transmat_), axis=0) + log_b[t]",
  "    return logsumexp(log_a)                       # equals hmm.score(x): -600.2392",
]));
body.push(H2("Chapter 4: Implementation Methodology and Hyperparameter Configuration"));
body.push(...code([
  "def adaboost_by_hand(X, y01, n_rounds, random_state):",
  "    t = np.where(y01 == 1, 1, -1); w = np.full(len(t), 1 / len(t))",
  "    rng = np.random.RandomState(random_state)",
  "    for m in range(n_rounds):",
  "        h = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(2**31 - 1))",
  "        h.fit(X, y01, sample_weight=w)",
  "        miss = np.where(h.predict(X) == 1, 1, -1) != t",
  "        eps = w[miss].sum() / w.sum()",
  "        alpha = np.log((1 - eps) / eps)",
  "        w = w * np.exp(alpha * miss); w = w / w.sum()",
  "",
  "# Bayesian model averaging over 256 logistic regressions",
  "bic = -2 * loglik + (n_coef + 1) * np.log(N)",
  "w = np.exp(-(bic - bic.min()) / 2); w = w / w.sum()",
  "p_bma = w @ probs",
]));

// ---------- preliminary pages ----------
const abstractText = "This project develops an explainable machine learning pipeline for the early prediction of diabetes risk from routine clinical data, covering all five units of the course. The objective is to identify people at risk early and to explain which factors drive that risk. The original datasets are the Pima Indians Diabetes dataset (768 patients), the Diabetes Progression dataset (442 patients) and the Kaggle Diabetes Prediction dataset (96,146 patients after cleaning); Unit 4 uses the UCI AIM-94 dataset of 13,515 blood glucose readings from 70 patients. Units 1 and 2 analysed data quality and compared regression and classification models; a tuned support vector machine reached a test recall of 0.796 and a ROC-AUC of 0.810, and Lasso regression explained 46 percent of the variance in disease progression. In Unit 3, a Gaussian mixture fitted with the Expectation-Maximisation algorithm found three risk groups without labels, with diabetes rates of 10.8, 38.0 and 56.9 percent. In Unit 4, a Hidden Markov Model learned low, target and high glucose states and fitted new patients better than a model without time order. In Unit 5, AdaBoost built from decision stumps reached a ROC-AUC of 0.977, verified against a hand-written implementation, while Bayesian model averaging improved probability estimates only when training data was scarce. Without blood tests, every model fell to about 0.83 and logistic regression matched the best ensemble. Across all methods, blood glucose, HbA1c, body mass index and age were the main risk factors, giving a consistent and explainable basis for decision support.";

const abbr = [["ADA", "American Diabetes Association"], ["AIM-94", "Artificial Intelligence in Medicine 1994 diabetes dataset"], ["ARI", "Adjusted Rand Index"],
  ["BIC", "Bayesian Information Criterion"], ["BMA", "Bayesian Model Averaging"], ["BMI", "Body Mass Index"], ["CART", "Classification and Regression Trees"],
  ["CV", "Cross-Validation"], ["EDA", "Exploratory Data Analysis"], ["EM", "Expectation-Maximisation"], ["GMM", "Gaussian Mixture Model"],
  ["HbA1c", "Glycated haemoglobin (average blood glucose over 2 to 3 months)"], ["HMM", "Hidden Markov Model"], ["KNN", "k-Nearest Neighbours"],
  ["MAE", "Mean Absolute Error"], ["PCA", "Principal Component Analysis"], ["RBF", "Radial Basis Function"], ["RMSE", "Root Mean Squared Error"],
  ["ROC-AUC", "Area Under the Receiver Operating Characteristic Curve"], ["SHAP", "SHapley Additive exPlanations"], ["SVM", "Support Vector Machine"],
  ["UCI", "University of California, Irvine"]];

function simpleTable(header, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cw = widths.map((w) => Math.round((w / total) * TEXT_W)); cw[cw.length - 1] += TEXT_W - cw.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, hdr) => new TableCell({ width: { size: cw[i], type: WidthType.DXA }, borders,
    shading: hdr ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined, margins: { top: 30, bottom: 30, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { line: 240, after: 0 }, children: [new TextRun({ text: t, font: FONT, size: 22, bold: hdr || undefined })] })] });
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: cw,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true)) }), ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, i, false)) }))] });
}
const prelimHead = (t, pb = true) => new Paragraph({ pageBreakBefore: pb, alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: t, bold: true, size: 32, font: FONT })] });

const prelims = [
  prelimHead("ABSTRACT", false),
  P(abstractText),
  prelimHead("TABLE OF CONTENTS"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  P("(Right-click the table and choose Update Field to refresh page numbers.)", { alignment: AlignmentType.CENTER }),
  prelimHead("LIST OF FIGURES"),
  simpleTable(["Figure No.", "Title"], figList.map(([id, c]) => [id, c]), [16, 84]),
  prelimHead("LIST OF TABLES"),
  simpleTable(["Table No.", "Title"], tabList.map(([id, c]) => [id, c]), [16, 84]),
  prelimHead("ABBREVIATIONS"),
  simpleTable(["Abbreviation", "Meaning"], abbr, [22, 78]),
];

// ---------- document ----------
const pageProps = { page: { size: { width: 11906, height: 16838 }, margin: { left: 2160, right: 1440, top: 1440, bottom: 1440 } } };
const footer = (fmt) => ({ default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] }) });

const doc = new Document({
  creator: "Team 3", title: "Explainable Machine Learning for Early Prediction of Diabetes Risk",
  features: { updateFields: true },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  styles: {
    default: { document: { run: { font: FONT, size: 24 }, paragraph: { spacing: { line: 360, after: 120 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, allCaps: true }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 0, after: 360 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true }, paragraph: { spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  sections: [
    { properties: { ...pageProps }, children: cover },
    { properties: { ...pageProps, page: { ...pageProps.page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } }, footers: footer(), children: prelims },
    { properties: { ...pageProps, page: { ...pageProps.page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } }, footers: footer(), children: body },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(REPO, "docs", "ML_Project_Report_Team3.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes;", figList.length, "figures,", tabList.length, "tables");
});
