# CXR8 Machine Learning Project Review Guide
## Complete Technical Breakdown: Data Pipeline, Unit 3 (Clustering & GMM), and Unit 4 (HMM)

This guide provides an end-to-end explanation of the entire project codebase, the underlying mathematical theory, the experimental setup, and the exact rationale behind every implementation choice. Use this document to prepare for your project review and viva presentation.

---

## Table of Contents
1. [Executive Summary & Syllabus Alignment](#1-executive-summary--syllabus-alignment)
2. [Dataset & Data Engineering Pipeline (`cxr8_data_helper.py`)](#2-dataset--data-engineering-pipeline-cxr8_data_helperpy)
3. [Unit 3: K-Means Clustering (`unit3_clustering_mlflow.py`)](#3-unit-3-k-means-clustering-unit3_clustering_mlflowpy)
4. [Unit 3: Gaussian Mixture Models (`unit3_clustering_mlflow.py`)](#4-unit-3-gaussian-mixture-models-unit3_clustering_mlflowpy)
5. [Unit 4: Hidden Markov Models with Causal Forward Filtering (`unit4_hmm_mlflow.py`)](#5-unit-4-hidden-markov-models-with-causal-forward-filtering-unit4_hmm_mlflowpy)
6. [MLflow Experiment Architecture](#6-mlflow-experiment-architecture)
7. [Review & Viva Q&A Cheat Sheet](#7-review--viva-qa-cheat-sheet)

---

## 1. Executive Summary & Syllabus Alignment

This project develops an MLflow-tracked unsupervised and probabilistic learning suite evaluated on the **NIH Chest X-Ray 14 (CXR8) longitudinal patient cohort**.

### Official Course Syllabus Mapping

| Syllabus Topic | Our Implementation | Script / Module | Status |
| :--- | :--- | :--- | :--- |
| **K-Means Clustering** | Multi-init K-Means hyperparameter optimization | `unit3_clustering_mlflow.py` | **Included (Unit 3)** |
| **Mixture of Gaussians (GMM)** | Gaussian Mixture Models across 4 covariance structures | `unit3_clustering_mlflow.py` | **Included (Unit 3)** |
| **Alternative View of EM** | EM lower bound (ELBO) and convergence tracking | `unit3_clustering_mlflow.py` | **Included (Unit 3)** |
| **Clustering Evaluation** | Silhouette Score, Davies–Bouldin, Calinski–Harabasz, Inertia | `unit3_clustering_mlflow.py` | **Included (Unit 3)** |
| **PCA & Dimensionality** | Leak-free PCA fit strictly on training cohort ($D \in \{8, 12, 16\}$) | `cxr8_data_helper.py` | **Included (Preprocessing)** |
| **Markov Models** | Transition probability matrix estimation ($A_{ij}$) | `unit4_hmm_mlflow.py` | **Included (Unit 4)** |
| **Hidden Markov Models** | `GaussianHMM` continuous multi-dimensional emissions | `unit4_hmm_mlflow.py` | **Included (Unit 4)** |
| **Forward Algorithm** | Log-space causal forward recursion with patient boundary resets | `unit4_hmm_mlflow.py` | **Included (Unit 4)** |
| **Scaling & Sum-Product** | Numerically stable log-sum-exp normalization | `unit4_hmm_mlflow.py` | **Included (Unit 4)** |
| **Hierarchical Clustering** | Explicitly excluded per professor/syllabus guidelines | N/A | **Excluded** |
| **Viterbi Algorithm** | Excluded per syllabus guidelines | N/A | **Excluded** |
| **Factor Analysis** | Excluded per syllabus guidelines | N/A | **Excluded** |

---

## 2. Dataset & Data Engineering Pipeline (`cxr8_data_helper.py`)

### 2.1 The Clinical Challenge & Longitudinal Motivation
Standard computer vision models treat chest X-rays as independent, static images. However, pulmonary pathologies (such as effusion, infiltration, pneumonia, and cardiomegaly) are **dynamic diseases** that evolve over multiple hospital visits. Modeling sequences of scans from the same patient provides clinical insights into disease onset, progression, and remission.

### 2.2 Sourcing Real Visual & Clinical Features
The project uses authentic raw NIH CXR8 data:
- **`Data_Entry_2017_v2020.csv`**: Contains 112,120 rows covering 30,805 unique patients.
- **Three Real Image Archives**:
  - `images_001.tar.gz` (5,000 images, Patient IDs 1 – 1,335)
  - `images_002.tar.gz` (10,001 images, Patient IDs 1,336 – 3,923)
  - `images_003.tar.gz` (10,001 images, Patient IDs 3,923 – 6,585)

### 2.3 Cohort Selection (No Leakage)
1. **Filtering**: We select only patients who have **at least 2 hospital visits** ($\ge 2$ scans) and whose images are **100% complete** across all visits in the 3 tar archives. There are 3,275 such patients (21,683 scans).
2. **Cohort Sampling**: To ensure fast, reproducible execution, we sample a fixed cohort of **1,500 patients** (10,055 total X-ray scans) using a fixed random seed (`seed=42`).
3. **Patient-Wise Partitioning**:
   - **Training Set (70%)**: 1,050 patients $\to$ 6,946 scans (mean sequence length: 6.6 visits)
   - **Validation Set (15%)**: 225 patients $\to$ 1,579 scans (mean sequence length: 7.0 visits)
   - **Test Set (15%)**: 225 patients $\to$ 1,530 scans (mean sequence length: 6.8 visits)
   - **Zero Patient Leakage**: All scans belonging to a patient reside entirely within one split. A patient's early scans never appear in training while their later scans appear in validation/test.

### 2.4 Multimodal Feature Representation (1,587 Dimensions)
Each scan is represented by 1,587 features combining visual gradients and clinical metadata:
1. **Visual HOG Features (1,568 dimensions)**:
   - Extracted directly from each raw PNG inside the tar archives using `skimage.feature.hog`.
   - The image is converted to grayscale and resized to $128 \times 128$.
   - Parameters: 8 gradient orientations, $16 \times 16$ pixels per cell, $2 \times 2$ cells per block.
   - Captures anatomical boundaries: cardiac borders, diaphragmatic domes, pleural margins, and focal parenchymal densities.
2. **Clinical & Acquisition Metadata (19 dimensions)**:
   - 14 binary thoracic pathology indicators: `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`.
   - `Finding_Count`: Total number of active concurrent findings.
   - Demographics: `Patient Age` (normalized float) and `Patient Sex` (binary: 1=Male, 0=Female).
   - Radiographic Geometry: `View Position` (binary: 1=PA [posteroanterior], 0=AP [anteroposterior]) and `Follow-up #` (integer visit count).

### 2.5 Preprocessing: Standard Scaling & PCA
- **Fitting**: `StandardScaler` and `PCA` are fitted **strictly on training patients only**.
- **Transforming**: The fitted scaler and PCA models then transform the validation and test patient scans. This prevents any statistical distribution leakage from validation/test data into the models.
- **Dimensionality Reduction**: We evaluate $D \in \{8, 12, 16\}$ principal components to capture the variance while preventing covariance matrix singularity in downstream models.

---

## 3. Unit 3: K-Means Clustering (`unit3_clustering_mlflow.py`)

### 3.1 Mathematical Objective
K-Means partitions the $N$ observations into $K$ non-overlapping clusters by minimizing within-cluster inertia (sum of squared Euclidean distances to cluster centroids $\mu_k$):
$$J = \sum_{k=1}^K \sum_{i \in C_k} \| x_i - \mu_k \|^2$$

### 3.2 Hyperparameter Grid
- **`n_clusters` ($K$)**: `[2, 3, 4, 5, 6]` — investigates the natural grouping of chest X-ray features.
- **`init`**: `["k-means++", "random"]` — compares standard random centroid seeding against `k-means++` (seeding proportional to squared distance, which guarantees $O(\log K)$ competitive clustering).
- **`n_init`**: `[10, 20]` — number of distinct centroid seeds to avoid local minima.
- **Total Trials**: $5 \times 2 \times 2 = 20$ nested child runs under parent run `KMeans_Parent_Hyperparameter_Optimization`.

### 3.3 Logged Evaluation Metrics
1. **Silhouette Score** ($s \in [-1, 1]$):
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   Measures whether an observation is much closer to its own cluster ($a(i)$) than to the nearest neighboring cluster ($b(i)$). Higher is better.
2. **Davies–Bouldin Index** ($\text{DB} \ge 0$):
   Measures the average similarity between each cluster and its most similar peer (ratio of within-cluster scatter to centroid separation). Lower is better.
3. **Calinski–Harabasz Score** ($\text{CH} \ge 0$):
   Variance ratio criterion (between-cluster dispersion divided by within-cluster dispersion). Higher is better.
4. **Inertia**: Total sum of squared distances to closest centroid. Decreases monotonically with $K$.

### 3.4 Key Experimental Findings
- **Optimal Model**: `n_clusters = 3`, `init = 'random'`, `n_init = 10`
- **Validation Silhouette Score**: `0.1297`
- **Davies-Bouldin Index**: `2.1851`
- **Calinski-Harabasz**: `214.0`
- **Clinical Interpretation**: $K=3$ groups patients into three natural radiographic phenotypes:
  1. *Clear / Normal* parenchymal patterns (low opacity, low finding count).
  2. *Cardiomegaly / Structural* patterns (enlarged cardiac silhouette, AP view bias).
  3. *Infiltrative / Fluid* disease (effusion, consolidation, high parenchymal attenuation).

---

## 4. Unit 3: Gaussian Mixture Models (`unit3_clustering_mlflow.py`)

### 4.1 Mathematical Foundation
Unlike K-Means (which assigns hard, binary cluster membership), Gaussian Mixture Models (GMM) assign **soft probabilistic memberships** by modeling the data density as a linear combination of $K$ multivariate Gaussian components:
$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k)$$
where $\pi_k \ge 0$, $\sum_{k=1}^K \pi_k = 1$, and $\mathcal{N}(x \mid \mu_k, \Sigma_k)$ is the multivariate Gaussian probability density function.

### 4.2 The Expectation-Maximization (EM) Algorithm
Parameters $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ are estimated iteratively via EM:
1. **E-step (Expectation)**:
   Compute the posterior probability (responsibility) that component $k$ generated data point $x_i$:
   $$\gamma_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
2. **M-step (Maximization)**:
   Update parameters using the effective number of points assigned to cluster $k$ ($N_k = \sum_{i=1}^N \gamma_{ik}$):
   $$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i, \quad \Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T, \quad \pi_k^{\text{new}} = \frac{N_k}{N}$$
3. **Convergence**: Monotonically increases the Evidence Lower Bound (ELBO) until $\ln p(X \mid \theta) - \ln p(X \mid \theta^{\text{old}}) < \text{tol}$.

### 4.3 Hyperparameter Grid (60 Combinations)
- **`n_components` ($K$)**: `[2, 3, 4, 5, 6]`
- **`covariance_type`**:
  - `"full"`: Each component has an independent, unconstrained covariance matrix $\Sigma_k$ ($D(D+1)/2$ parameters per component). Captures arbitrary ellipsoidal shapes and feature correlations.
  - `"tied"`: All $K$ components share a single common covariance matrix $\Sigma$.
  - `"diag"`: Covariance matrices are diagonal; assumes features are conditionally independent along coordinate axes.
  - `"spherical"`: Covariance matrices are isotropic ($\Sigma_k = \sigma_k^2 I$). Clusters are spherical, similar to soft K-Means.
- **`reg_covar`**: `[1e-6, 1e-4, 1e-2]` — positive regularization constant added to the diagonal of $\Sigma_k$ ($\Sigma_k \leftarrow \Sigma_k + \epsilon I$) to prevent numerical instability and matrix collapse.
- **Parent Run**: `GMM_Parent_Hyperparameter_Optimization` (60 nested child runs: `Trial_1` to `Trial_60`).

### 4.4 Model Selection Criterion (BIC vs Log-Likelihood)
To prevent overfitting complex full-covariance models, model selection uses:
1. **Validation Log-Likelihood**: Measures generalization to unseen patients.
2. **Bayesian Information Criterion (BIC)**:
   $$\text{BIC} = k_{\text{params}} \ln N - 2 \ln \hat{L}$$
   Penalizes model complexity ($k_{\text{params}}$) heavily to select the most parsimonious model.
3. **Selection Rule**: Filter all models within 2% of the maximum validation log-likelihood, then pick the one with the lowest training BIC.

### 4.5 Key Experimental Findings
- **Optimal Configuration**: `n_components = 6`, `covariance_type = 'full'`, `reg_covar = 1e-06`
- **Validation Log-Likelihood**: `-37.5470`
- **Training BIC**: `525,269.7`
- **EM Diagnostics**: Converged in 14 iterations (`em_converged = 1.0`, lower bound ELBO = `-36.879`).

---

## 5. Unit 4: Hidden Markov Models with Causal Forward Filtering (`unit4_hmm_mlflow.py`)

### 5.1 The Need for HMM in Longitudinal Medicine
While GMM models individual scans, it ignores the temporal sequence of a patient's visits. A Hidden Markov Model (HMM) models the patient's latent health state $S_t \in \{1, \dots, K\}$ as a discrete Markov chain, with continuous radiographic observations $x_t \in \mathbb{R}^D$ emitted from state-dependent Gaussian distributions.

### 5.2 Model Components
1. **Initial State Distribution**: $\pi_j = P(S_0 = j)$
2. **Transition Probability Matrix**: $A_{ij} = P(S_t = j \mid S_{t-1} = i)$
3. **Continuous Gaussian Emission**:
   $$b_j(x_t) = P(x_t \mid S_t = j) = \mathcal{N}(x_t \mid \mu_j, \Sigma_j)$$

### 5.3 The Causal Forward Recursion (No Future Leakage)
Standard HMM smoothing (the Forward-Backward algorithm) computes $P(S_t \mid x_{0:T-1})$, which uses future scans to infer the current state. In medical prognosis, clinicians only have past and current scans $x_{0:t}$. We implement a custom **causal forward filter** operating in log-space:

1. **Step 0 (Sequence Initialization)**:
   $$\alpha_0(j) = \ln \pi_j + \ln b_j(x_0)$$
   $$c_0 = \operatorname{logsumexp}_k \alpha_0(k), \quad P(S_0 = j \mid x_0) = \exp(\alpha_0(j) - c_0)$$
2. **Step $t$ (Causal Forward Recursion)**:
   $$\alpha_t(j) = \ln b_j(x_t) + \operatorname{logsumexp}_{i=1}^K \left( \alpha_{t-1}(i) + \ln A_{ij} \right)$$
   $$c_t = \operatorname{logsumexp}_k \alpha_t(k), \quad P(S_t = j \mid x_{0:t}) = \exp(\alpha_t(j) - c_t)$$
3. **Patient Boundary Resets**:
   When moving from patient $p$ to patient $p+1$, the forward recursion **resets** to $\ln \pi$, ensuring patient independence.

### 5.4 Hyperparameter Grid (24 Combinations)
- **`n_components` ($K$)**: `[2, 3, 4, 5]`
- **`covariance_type`**: `["diag", "spherical"]`
- **`n_pca_components`**: `[8, 12, 16]`
- **Parent Run**: `HMM_Parent_Hyperparameter_Optimization` (24 nested child runs: `Trial_1` to `Trial_24`).

### 5.5 Clinical Validation: Contingency Chi-Square ($\chi^2$) Test
Since HMM hidden states are unsupervised, we test their clinical relevance by computing the contingency table between the most probable causal state $s_t^* = \arg\max_j P(S_t = j \mid x_{0:t})$ and the binary presence of active pathology. A significant $\chi^2$ statistic with $p < 0.05$ proves that the latent states meaningfully reflect disease status.

### 5.6 Key Experimental Findings
- **Optimal Configuration**: `n_components = 5`, `covariance_type = 'diag'`, `n_pca_components = 8`
- **Validation Forward Log-Likelihood**: `-26.8461` (highest among all configurations)
- **Training BIC**: `373,984.2`
- **Clinical Significance**: $\chi^2 = 28.22$, $p = 1.1 \times 10^{-5}$ (statistically significant correlation with pathology).
- **Test Set Generalization (Evaluated Exactly Once)**:
  - Test Forward Log-Likelihood: `-26.7891` (closely matches validation, proving zero overfitting).
  - Test Pathology $\chi^2$: `74.77`, $p = 2.2 \times 10^{-15}$.

---

## 6. MLflow Experiment Architecture

All runs are organized hierarchically in MLflow:

```
MLflow Tracking Server
├── Experiment: CXR8_Unit3_KMeans_GMM_Optimization
│   ├── Parent Run: KMeans_Parent_Hyperparameter_Optimization
│   │   ├── Trial_1  (k=2, init=k-means++, n_init=10)
│   │   ├── ...
│   │   ├── Trial_20 (k=6, init=random, n_init=20)
│   │   └── Logged Model: scikit-learn KMeans (best k=3)
│   │
│   └── Parent Run: GMM_Parent_Hyperparameter_Optimization
│       ├── Trial_1  (comp=2, cov=full, reg=1e-6)
│       ├── ...
│       ├── Trial_60 (comp=6, cov=spherical, reg=1e-2)
│       └── Logged Model: scikit-learn GaussianMixture (best comp=6)
│
└── Experiment: CXR8_Unit4_HMM_Forward_Optimization
    └── Parent Run: HMM_Parent_Hyperparameter_Optimization
        ├── Trial_1  (states=2, cov=diag, PCA=8)
        ├── ...
        ├── Trial_24 (states=5, cov=spherical, PCA=16)
        └── Logged Metrics: Best params, BIC, AIC, Val LL, Test LL, Chi2
```

To launch the MLflow UI and view all metrics and parameters in your browser:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```

---

## 7. Review & Viva Q&A Cheat Sheet

Use these exact answers when questioned by your professor or examiner:

### Q1: "Why did you use K-Means and GMM in Unit 3, and what is the fundamental difference between them?"
**Answer**:
"K-Means performs hard assignment by minimizing Euclidean inertia, assuming spherical clusters of equal variance. GMM is a probabilistic generative model that performs soft clustering using the Expectation-Maximization (EM) algorithm. GMM assigns each scan a continuous posterior responsibility $\gamma_{ik}$ and models cluster geometry using covariance matrices (ellipsoidal, tied, diagonal, or spherical), allowing clusters with different sizes and orientations."

### Q2: "How does the Expectation-Maximization (EM) algorithm work in your GMM implementation?"
**Answer**:
"EM maximizes the log-likelihood of the data when latent component indicators are unobserved.
- In the **E-step**, we compute the posterior responsibility $\gamma_{ik} = P(z_i = k \mid x_i, \theta)$ using the current parameters.
- In the **M-step**, we update the mixture weights $\pi_k$, mean vectors $\mu_k$, and covariance matrices $\Sigma_k$ weighted by these responsibilities.
Our MLflow suite logs the number of EM iterations, convergence status, and the Evidence Lower Bound (ELBO) to verify monotonic convergence."

### Q3: "Why did you avoid Agglomerative/Hierarchical Clustering?"
**Answer**:
"Our course syllabus strictly specifies K-Means, Gaussian Mixture Models, and EM for Unit 3. Hierarchical clustering (and dendrograms/Ward linkage) has an $O(N^2)$ memory and $O(N^3)$ computational complexity, which is impractical for large chest X-ray cohorts, and was explicitly excluded to adhere strictly to the syllabus requirements."

### Q4: "How does your Hidden Markov Model differ from a standard GMM?"
**Answer**:
"A GMM treats each chest X-ray as an independent sample. In contrast, an HMM explicitly models the temporal sequence of a patient's visits using a discrete transition probability matrix $A_{ij} = P(S_t = j \mid S_{t-1} = i)$. The hidden state represents the latent disease stage of the patient, and the observed scan features are emitted from state-specific Gaussian distributions."

### Q5: "Why did you write a custom causal forward algorithm instead of using standard `model.predict_proba()` in hmmlearn?"
**Answer**:
"`hmmlearn`'s built-in posterior inference uses the Forward-Backward algorithm (smoothing), which conditions on future observations ($x_{t+1:T}$). In a real clinical setting, a doctor cannot look into a patient's future visits to diagnose their current scan. Our custom causal forward algorithm computes $P(S_t \mid x_{0:t})$ using past and present scans only, implemented in log-space using `logsumexp` for numerical stability, and resets at patient sequence boundaries so different patients remain strictly independent."

### Q6: "How did you prevent data leakage in your longitudinal split?"
**Answer**:
"We partitioned the data strictly at the **patient level** (1,050 training, 225 validation, 225 test patients). All visits for any single patient are kept in the same split. Furthermore, both `StandardScaler` and `PCA` were fitted **exclusively on the training patients**, and then applied to transform validation and test patients. Finally, the test set was evaluated **exactly once** on the optimal model selected by validation log-likelihood and training BIC."

---

## 8. Summary of Execution Commands

```powershell
# 1. Run Unit 3 (K-Means & GMM Hyperparameter Optimization)
& "C:\Users\Heytish V\AppData\Local\Programs\Python\Python312\python.exe" unit3_clustering_mlflow.py

# 2. Run Unit 4 (Gaussian HMM Causal Forward Optimization)
& "C:\Users\Heytish V\AppData\Local\Programs\Python\Python312\python.exe" unit4_hmm_mlflow.py

# 3. View MLflow Dashboard (optional)
mlflow ui --host 127.0.0.1 --port 5000
```
