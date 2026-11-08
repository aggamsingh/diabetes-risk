# Presentation script: PBL 2 and PBL 3 (16 slides)

A spoken script for the current deck, slide by slide. About 1 minute per slide, roughly 16 minutes in total.

**Suggested speaker split:** Jai slides 1 to 5, Aggam slides 6 to 11, Alphin slides 12 to 16.

For the meaning of any term, see [GLOSSARY.md](GLOSSARY.md). For the code behind each slide, see [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md).

---

## Slide 1: Title

**What's on it:** project title, team and course.

**What to say:**
"Good morning ma'am. We are Team 3: Jai, Aggam and Alphin. This is our combined review for Project 2, the alternative view of the Expectation-Maximisation algorithm from Unit 3, and Project 3, the Hidden Markov Model from Unit 4. Both continue our diabetes risk project from Review 1."

## Slide 2: Project overview

**What's on it:** PBL 1 (supervised prediction) on one side, PBL 2 and 3 (unsupervised and sequential) on the other.

**What to say:**
"In Project 1 we used the diagnosis to predict diabetes, which is supervised learning. In Project 2 we remove the diagnosis and ask whether patients naturally fall into risk groups, using a Gaussian Mixture Model fitted with the Expectation-Maximisation algorithm on 768 patients. In Project 3 we add time: we model glucose readings as hidden states with a Hidden Markov Model, on 13,515 readings from 70 patients. The connecting idea is that both models are trained with Expectation-Maximisation; the Hidden Markov Model adds a transition matrix between states."

## Slide 3: Three datasets, one per task

**What's on it:** one table comparing the three datasets.

| | Kaggle Diabetes Prediction | Pima Indians Diabetes | UCI Diabetes (AIM-94) |
|---|---|---|---|
| Used in | PBL 1, prediction | PBL 2, EM clustering | PBL 3, HMM |
| Raw size | 100,000 x 9 | 768 x 9 | 29,330 x 4, 70 patient files |
| Label | diabetes 8.5% | Outcome 34.9% (held out) | none, states are hidden |
| Modelled | 96,146 x 8 | 768 x 5 | 13,515 readings, 68 patients, 54 / 14 split |
| Why chosen | large and labelled | continuous values | time-ordered, median 120 readings per patient |
| Why not reused | HbA1c and glucose: 18 distinct values each | one row per patient, no time order | |

**What to say:**
"Each project asked a different question, so each needed a different kind of data. Project 1 predicts diabetes, so it needs many labelled patients: the Kaggle dataset, 100,000 rows, 96,146 after removing duplicates. Project 2 fits bell curves, which need continuous numbers. Kaggle's long-term blood sugar and glucose take only 18 values each, and half its attributes are categories, so we used Pima: 768 women, 8 numeric attributes, with the diagnosis hidden during fitting. Project 3 needs readings in time order for the same person, which Pima does not have. The University of California Irvine dataset has 29,330 events from 70 patients between 1988 and 1991. 13,531 of them are glucose readings, 13,515 after cleaning. We keep the 68 patients with at least 50 readings and split them by patient into 54 for training and 14 for testing."

**If asked "why not use one dataset for everything?":** "Each model needs a different shape of data: labels for prediction, continuous values for bell curves, and time order for a Hidden Markov Model. No single dataset has all three."

## Slide 4: Alternative view of the Expectation-Maximisation algorithm

**What's on it:** complete vs incomplete data, the Q-function, the two steps and three properties.

**What to say:**
"Each patient has a hidden group label we cannot see. If we knew every label, called complete data, fitting would be easy: just the average of each group. But we only have incomplete data, the measurements, and the likelihood becomes a log of a sum, which has no direct solution. So the algorithm maximises the expected complete-data log-likelihood, called the Q-function. In the expectation step we compute each patient's probability of belonging to each group, called responsibilities. In the maximisation step we update each group's weight, mean and spread using those probabilities. Three properties follow: the likelihood never decreases; it can stop at a local optimum, so we restart it 10 times; and if we force the probabilities to be 0 or 1, we get K-means, the hard version of this algorithm."

## Slide 5: Data preparation

**What's on it:** a table of each Pima column and what we did with it.

**What to say:**
"We used the Pima dataset with 768 patients. A zero in this dataset means missing, since a glucose of zero is impossible. We removed insulin, 48.7% missing, and skin thickness, 29.6% missing. We also removed pregnancies: on our first attempt, one bell curve collapsed onto the patients with exactly zero pregnancies and broke the model selection. We log-transformed the two skewed features, family history score and age, and standardised everything so no feature dominates. The diagnosis is kept aside and only used afterwards to check our groups."

## Slide 6: Choosing the number of groups

**What's on it:** Bayesian Information Criterion table for 1 to 6 groups, full and diagonal covariance.

**What to say:**
"To decide how many groups to use, we tried 1 to 8 groups with two bell-curve shapes and scored each with the Bayesian Information Criterion. It rewards good fit but penalises extra parameters; lower is better. The best was 3 groups with diagonal covariance, at 10422.4. Four groups was almost tied at 10423.5, so we kept the simpler model. This is more reliable than the elbow method because it uses the actual likelihood instead of judging a graph by eye."

## Slide 7: Watching the algorithm converge

**What's on it:** log-likelihood per step, and the groups after steps 1, 3 and 60.

**What to say:**
"Here we ran the algorithm one step at a time. The log-likelihood rose from -7.094 to -6.646 over 60 steps and never decreased once, which proves the property from slide 4. The small plots show the groups going from mixed to clearly separated. To show we understand the library, we wrote the expectation step ourselves: our probabilities match scikit-learn to within 0.0000000000000013, which is rounding error. The final model converged in 25 steps."

## Slide 8: Three risk profiles

**What's on it:** three cards, one per group, with size and diabetic rate.

**What to say:**
"These are the three groups the model found without ever seeing the diagnosis. The low-risk group has 269 young patients with normal glucose; only 10.8% are diabetic. The older group has 237 patients, 38% diabetic. The high-risk group has 262 patients with the highest glucose, body mass index and family history, and 56.9% are diabetic. So the diabetes rate rises about five times across the groups, found without labels, and the drivers are the same ones our Project 1 models found."

## Slide 9: Evaluating the clusters

**What's on it:** silhouette and Adjusted Rand Index for both methods, borderline share, restarts.

**What to say:**
"We compared our soft clustering with K-means, the hard version. K-means scores higher on the silhouette score, 0.216 against 0.143, and the Adjusted Rand Index, 0.185 against 0.101. But those scores favour round, equal-sized groups, which is exactly what K-means produces. The Adjusted Rand Index is low for both because the groups describe risk levels, not a yes or no diagnosis. The real advantage of the mixture model is the probabilities: 30.6% of patients, 235 people, are borderline, with no group above 80%. K-means hides that completely. We also found that 9 out of 10 single random starts got stuck at a worse answer, which is why we use 10 restarts."

## Slide 10: Sequential dataset

**What's on it:** 13,515 readings, 70 patients, 120 readings per patient, 6 hour gap, one patient's 4 weeks.

**What to say:**
"For Unit 4 we needed data where order matters. We used the University of California Irvine diabetes dataset: 13,515 glucose readings from 70 insulin-dependent patients, about 120 readings each, roughly 6 hours apart. To prove the order matters: each reading is correlated with the previous one at 0.186, and when we shuffle the order that drops to 0. We kept the 68 patients with at least 50 readings and split by patient, 54 for training and 14 for testing, so the model is tested on people it has never seen."

## Slide 11: Hidden Markov Model structure

**What's on it:** the three parts of the model and the three classic problems with our checks.

**What to say:**
"The model has three parts: start probabilities, a transition matrix between hidden states, and a bell curve of glucose values for each state. It solves three classic problems, and we checked each against our own code. For evaluation, our forward algorithm gives -600.2392, exactly the library's value. For decoding, our Viterbi algorithm gives the identical path for all 104 readings. For learning, we used the Baum-Welch algorithm, which is Expectation-Maximisation for Hidden Markov Models, with 5 restarts. The Bayesian Information Criterion preferred 5 states, but they overlapped, so we chose 3 because they match the medical ranges: 75 low, 126 in range and 206 high. All calculations run in logarithms so numbers never become too small for the computer."

---

## Slide 12: What the model learned (transitions and how long states last)

**What's on it:** a table of the chance of moving from one state to another. Each row adds up to 100%.

| From this state, next reading goes to | Low | In range | High |
|---|---|---|---|
| **Low** | 43% | 1% | 56% |
| **In range** | 0% | 97% | 3% |
| **High** | 20% | 1% | 79% |

**How to read it:** go along a row. "If the patient is Low now, the next reading is Low 43% of the time, In range 1%, High 56%."

**What it tells us, and why:**

1. **Low is usually followed by High (56%).**
   - Why: when sugar drops, the body releases stress hormones and the patient often eats something sugary to recover. Both push sugar up too far. Doctors call this the **rebound effect**.
   - Why it matters: the model was never told this; it found a known medical effect from the numbers alone.

2. **In range is very stable (97% stay).** How long a state lasts = 1 / (1 - chance of staying):

| State | Calculation | Lasts about |
|---|---|---|
| In range | 1 / (1 - 0.97) | 34 readings (around 8 days, since readings are about 6 hours apart) |
| High | 1 / (1 - 0.79) | 4.7 readings |
| Low | 1 / (1 - 0.43) | 1.75 readings (a low passes quickly) |

   - Why it matters: in range behaves like a separate "good control period", not something patients bounce in and out of.

3. **Almost nobody moves from Low or High into In range (1% or less).** Once control is lost it does not recover in one reading; patients swing between low and high instead.

4. **Stationary distribution (long-run time in each state).**

| | Low | In range | High |
|---|---|---|---|
| Predicted long-run share | 18.8% | 28.0% | 53.2% |
| Share we actually decoded | 20.8% | 30.4% | 48.7% |

   - How: the probabilities that stay the same after one more step through the table (an eigenvector of the transition matrix).
   - Why it matters: the two rows are close, so the table is consistent with the real data. These patients spend over half their time high.

**What to say:**
"This table is the heart of the Hidden Markov Model: it tells us how glucose moves between states. Three things stand out. First, after a low reading, the next is usually high, 56% of the time. That is the known rebound after low sugar, and the model found it without being told. Second, the in-range state is very stable: patients stay there 97% of the time, which works out to about 34 readings. Third, patients almost never jump from low or high straight back into range. The long-run time in each state works out to 18.8% low, 28% in range and 53.2% high, which closely matches what we actually saw, so the table is consistent with the data."

**If asked "why are the states sorted?":** "The algorithm numbers states randomly, so we sorted them by average glucose. That way state 1 is always low, 2 in range and 3 high."

## Slide 13: Decoding one patient (test patient 48)

**What's on it:** two graphs for one patient the model never saw during training.

**Top graph, Viterbi algorithm:**
- What: each of the 104 readings coloured by its most likely hidden state.
- How: Viterbi finds the single best **full path** of states through all 104 readings at once, keeping the best route into each state at each step and tracing back at the end.
- Why a full path: it uses the transition table. A single reading of 150 might look in range on its own, but if the readings around it are all high, the best overall path calls it high.

**Bottom graph, forward-backward algorithm:**
- What: for each reading, the probability of each state, which shows **how confident** the model is.
- How: it looks at readings before and after each point (smoothing).
- Example: at the sharp dips the model is about 85% sure the state is low.

**What it shows:** readings swing from about 40 to about 330; **not one reading is in range**; mostly high with repeated sudden drops to low. That is unstable control. An average glucose value would hide the dangerous drops; the hidden states show the pattern over time.

**What to say:**
"This is test patient 48, someone the model never saw during training. The top graph shows the Viterbi algorithm's answer: the single most likely sequence of states across all 104 readings. The bottom graph shows the forward-backward probabilities, which tell us how confident the model is at each point; at the dips it is about 85% sure the patient is low. This patient is never in range. They are mostly high, with repeated sudden drops. An average glucose value would hide this unstable pattern, but the hidden states make it obvious."

**If asked "what is the difference between the two graphs?":** Viterbi gives the best **whole path** together; forward-backward gives the most likely state **at each reading on its own**, plus a confidence. They usually agree but do not have to.

## Slide 14: Testing on new patients

**Why test on new patients:** a model can memorise training data. The real test is people it has never seen, so 14 patients were held back completely.

**Test 1: how well each model explains the new patients' readings** (log-likelihood per reading, higher is better)

| Model | What it knows | Score |
|---|---|---|
| **Hidden Markov Model** | 3 bell curves **plus** the transition table (time order) | **-5.6847** |
| Gaussian Mixture Model | the same 3 bell curves, **no** time order | -5.7313 |
| Single bell curve | one bell curve for all readings | -5.7861 |

Why this is fair: the first two use the same 3 bell curves. The only difference is the time order, so the improvement can only come from modelling time.

**Test 2: predicting the next reading** (average error in milligrams per decilitre, lower is better)

| Way of guessing the next reading | Average error |
|---|---|
| **Hidden Markov Model** | **60.1** |
| Always guess the training average | 62.6 |
| Copy the last reading | 74.0 |

How the model predicts:
1. Using only past readings, work out the probability of each state right now (**filtering**).
2. Multiply by the transition table to get the probability of each state at the next reading.
3. Turn that into an expected glucose value using each state's average.

Why only the past here, when slide 13 used past and future: a real forecast cannot see the future. Slide 13 uses **smoothing** (for understanding), slide 14 uses **filtering** (for prediction).

Why the improvement is small: readings are about 6 hours apart, meals and insulin change a lot in that time, and the link between neighbouring readings is weak (0.186). Copying the last reading is worst because it copies that reading's random noise.

**What to say:**
"To prove the model works, we tested it on 14 patients it never saw. First, how well three models explain their readings: the Hidden Markov Model scores best at -5.68. The mixture model, with the same three bell curves but no time order, scores -5.73. The only difference is the time order, and that makes it better. Second, predicting each patient's next reading from their past readings only: the Hidden Markov Model's average error is 60.1, better than always guessing the average, 62.6, or copying the last reading, 74. The gain is modest because readings are about 6 hours apart, and a lot happens in 6 hours."

## Slide 15: What the states mean medically

**Key point:** the model was **never told the time of day or meal times**. It only saw glucose numbers in order. We compared its states with meal times afterwards, so any pattern is a **discovery**, not an input.

1. **Mornings are high.** 63.0% of before-breakfast readings are high, against 49.4% before lunch. Why: early-morning hormones (cortisol, growth hormone) raise blood sugar, the **dawn phenomenon**.
2. **Evenings are often low.** 29.9% of readings before the evening snack are low, against 24.6% after supper. Why: insulin taken earlier is still working and dinner has been digested, which is why diabetic patients are told to have a bedtime snack.
3. **Many patients never reach the good state.** 8 of the 14 test patients never reach in range; only 4 spend 26% to 58% of their time there.
4. **It ranks patients by control.** Best: patient 49, only 5.1% of readings high. Worst: patient 6, 96.9% high. Doctors use **time in range** to judge diabetes control; our model produces something very similar from raw readings.

**What to say:**
"This slide is why we call the project explainable. The model was never told the time of day; it only saw glucose numbers in order. But when we compared its states with meal times, real medical patterns appeared. 63% of before-breakfast readings are high, compared with 49% before lunch, which matches the dawn phenomenon. Before the evening snack, nearly 30% of readings are low, matching the known evening drop from insulin still working. 8 of the 14 test patients never reach the in-range state, and the time in each state ranks patients from best control, patient 49 at only 5% high, to worst, patient 6 at 97% high. That is very close to the time-in-range measure doctors already use."

## Slide 16: Conclusions and limitations

**PBL 2 summary:** 3 risk groups found without the diagnosis, 10.8%, 38.0% and 56.9% diabetic; drivers (glucose, body mass index, family history) match Project 1, so two methods agree; 30.6% of patients are borderline, which only soft clustering shows.

**PBL 3 summary:** 3 glucose states (75, 126, 206) matching medical ranges; better than the same model without time on new patients; found the rebound after lows, morning highs and evening lows by itself.

**Limitations:**

| Limitation | What it means | Why it matters |
|---|---|---|
| Pima is one population | women aged 21+ of Pima heritage only | groups may differ for other people |
| Bell-curve assumption | each group is assumed bell-shaped | real groups may be lopsided |
| Uneven time gaps | every step treated as equal, though gaps vary from hours to a whole night | a long gap should allow more change |
| Glucose only | insulin, meals and exercise not used, though the dataset has them | adding them could explain why states change |
| Old, small data | 70 type 1 patients from 1988 to 1991 | modern treatment differs |
| Association, not cause | the models find patterns, not reasons | family history goes with risk, but we have not proven it causes it |

**What to say:**
"To conclude. In Project 2, the Gaussian Mixture Model found three risk groups without ever seeing the diagnosis, with the diabetic rate rising from about 11% to 57%. The drivers match Project 1, and almost a third of patients are borderline. In Project 3, the Hidden Markov Model found three glucose states matching medical ranges, beat the same model without time on new patients, and discovered real patterns like the rebound after lows, morning highs and evening lows. Our limitations: Pima covers only one population, and the mixture model assumes bell-shaped groups. The Hidden Markov Model treats uneven time gaps as equal steps and uses glucose only. The data is old and small, and our results show associations, not causes. A natural next step is adding insulin and meal information to the Hidden Markov Model. Thank you, ma'am. We are happy to take questions."
