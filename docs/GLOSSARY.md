# Glossary: the terms used in PBL 2 and PBL 3

Plain-language meanings of every technical term in the notebooks, slides and viva prep.

## Short forms

| Short form | Full name |
|---|---|
| EM | Expectation-Maximisation (algorithm) |
| GMM | Gaussian Mixture Model |
| HMM | Hidden Markov Model |
| BIC | Bayesian Information Criterion |
| PCA | Principal Component Analysis |
| ARI | Adjusted Rand Index |
| MAE | Mean Absolute Error |
| HbA1c | Glycated haemoglobin (average blood sugar over the last 2 to 3 months) |
| BMI | Body Mass Index |
| UCI | University of California Irvine (Machine Learning Repository) |
| AIM-94 | the diabetes dataset from the 1994 Artificial Intelligence in Medicine symposium |

## Part 1: Clustering and EM (PBL 2, Unit 3)

**Clustering.** Grouping similar patients together without telling the computer the answer. Nobody tells it "this patient is diabetic"; it finds the groups by itself.

**Bell curve (Gaussian or normal distribution).** The hump shape: most values sit near the middle, fewer far away. It is described by two numbers: the **mean** (where the middle is) and the **variance** or **standard deviation** (how wide it is).

**Gaussian Mixture Model.** A model that assumes the data is made of several bell curves overlapping. Each bell curve is one group of patients. It gives each patient a probability of belonging to each group, for example 70% low risk, 25% older, 5% high risk.

**Soft vs hard clustering.**
- Soft (Gaussian Mixture Model): a patient can be partly in several groups.
- Hard (K-means): each patient goes 100% into exactly one group.

**K-means.** The simplest clustering method. Place K centre points, give each patient to the nearest centre, move each centre to the middle of its patients, repeat.

**Latent (hidden) variable.** Something that exists but we cannot see. Here: which group each patient truly belongs to. We only see their measurements.

**Expectation-Maximisation algorithm.** The method used to fit a Gaussian Mixture Model. A loop of two steps:
1. **Expectation step (E-step):** using the current bell curves, work out the probability that each patient belongs to each group. These probabilities are called **responsibilities**.
2. **Maximisation step (M-step):** using those probabilities, move and reshape each bell curve to fit its patients better.
3. Repeat until nothing changes. Each loop is guaranteed to fit the data at least as well as the last; it never gets worse.

**Alternative view of the Expectation-Maximisation algorithm.** The Unit 3 chapter title. It explains why the loop works:
- **Complete data:** if we knew every patient's true group, fitting is easy (take the average of each group).
- **Incomplete data:** we do not know the groups, so the maths becomes a "log of a sum", which has no direct solution.
- **The trick:** maximise the *expected* complete-data score instead, called the **Q-function**. The expectation step builds it, the maximisation step maximises it.
- **The link:** if you force the probabilities to be only 0 or 1, you get K-means. So K-means is "hard" Expectation-Maximisation.

**Local optimum (local maximum).** The algorithm climbs uphill, but it can stop on a small hill instead of the tallest mountain. That is why we restart it 10 times from different random starting points and keep the best.

**Log-likelihood.** One number for how well the model explains the data. Higher (closer to zero) is better. It is negative because it is the logarithm of a small probability.

**Bayesian Information Criterion.** A score for choosing how many groups to use. More groups always fit better but risk over-fitting, so the score is (how badly the model fits) + (a penalty for every extra parameter). Lower is better. It picks the model that fits well without being needlessly complicated.

**Elbow method.** The K-means way of picking the number of groups: plot the error and look by eye for where the curve bends like an elbow. The Bayesian Information Criterion gives an exact number instead.

**Covariance: full vs diagonal.**
- Full: each group's bell curve can be tilted (features are allowed to move together).
- Diagonal: each feature spreads independently, so there are fewer parameters.

**Principal Component Analysis.** Squashes many features into 2 or 3 new ones while keeping as much information as possible. We used it only to draw the groups on a flat graph (2 components keep 56.1% of the variance).

**Silhouette score.** How tight and well-separated the groups are, using only the data. From -1 to 1, higher is better.

**Adjusted Rand Index.** How closely the groups match a known answer (here, the real diagnosis). 1 means identical, 0 means no better than random.

**Standardise.** Rescale each feature to an average of 0 and a spread of 1, so a feature with big numbers (glucose, about 120) does not outweigh a small one (the family history score, about 0.5).

**Log transform.** Taking the logarithm of a skewed feature (one with a long tail) to make it more bell-shaped.

## Part 2: Sequences and the HMM (PBL 3, Unit 4)

**Sequential data.** Data where the order matters, such as glucose readings over weeks. Shuffling the order destroys information.

**Autocorrelation.** How much a reading is linked to the one before it. Ours is 0.186, and it drops to 0 when the order is shuffled.

**Markov chain.** A model where the next state depends only on the current state, not the full history. Example: tomorrow's weather depends only on today's.

**Transition matrix.** A table of the probabilities of moving from each state to each other state. Each row adds up to 1.

**Hidden Markov Model.** A Markov chain where the states **cannot be seen directly**; we only see measurements the states produce.
- Hidden states: low, in range, high blood sugar.
- What we see: the noisy glucose reading.
- Three parts:
  1. **Start probabilities:** how likely each state is at the first reading.
  2. **Transition matrix:** the chance of moving between states.
  3. **Emission probabilities:** for each state, a bell curve of the glucose values it produces.

**The three classic Hidden Markov Model problems.**
1. **Evaluation:** how likely is this sequence of readings? Solved by the **forward algorithm**.
2. **Decoding:** what is the most likely sequence of hidden states? Solved by the **Viterbi algorithm**.
3. **Learning:** find the model's numbers from data. Solved by the **Baum-Welch algorithm**.

**Forward algorithm.** Goes through the readings in order and keeps a running probability of being in each state. Using only past readings like this is called **filtering**.

**Forward-backward algorithm.** Runs forward, then backward, so it uses both past and future readings to judge each state. This is called **smoothing**. It gives the probability of each state at each reading.

**Viterbi algorithm.** Finds the single best full path of states. At each step it keeps only the best route into each state and remembers where it came from (**back-pointers**), then traces back at the end.

**Baum-Welch algorithm.** The Expectation-Maximisation algorithm applied to a Hidden Markov Model. The expectation step uses forward-backward; the maximisation step updates the transition matrix and the bell curves. This is the link between PBL 2 and PBL 3.

**Log-sum-exp (scaling).** Multiplying many small probabilities gives a number so tiny the computer rounds it to 0 (**underflow**). Working with logarithms avoids this; log-sum-exp is the safe way to add numbers stored as logarithms.

**Stationary distribution.** If the chain ran forever, the long-run share of time spent in each state. Ours: 18.8% low, 28.0% in range, 53.2% high.

**Expected duration.** How many readings a state usually lasts: 1 / (1 - the chance of staying in it).

**Train/test split by patient.** The model learns from 54 patients and is tested on 14 it has never seen. Splitting by patient means no patient's readings are in both sets.
