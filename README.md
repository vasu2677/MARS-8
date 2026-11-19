# MARS-8: CXR8 Longitudinal MLflow Hyperparameter Optimization Suite

A comprehensive, leak-free, reproducible MLflow Hyperparameter Optimization suite applied to the **NIH Chest X-Ray (CXR8) longitudinal dataset**, specifically targeting foundational topics from:
- **Unit 3: Mixture Models and Expectation-Maximization (EM)**
- **Unit 4: Hidden Markov Models (HMM)**
- **Unit 5: Ensemble Learning & Tree-Based Methods (Random Forest & XGBoost / Gradient Boosting)**

---

## 1. Syllabus-to-Project Alignment

| Official Syllabus Topic | Implementation in Suite | Alignment Status |
| :--- | :--- | :--- |
| **K-Means Clustering** | Multi-initialization K-Means hyperparameter optimization | **Direct** |
| **Mixtures of Gaussians** | Gaussian Mixture Models (`GaussianMixture`) across 4 covariance structures | **Direct** |
| **Alternative View of EM** | EM optimization diagnostics (`em_n_iter`, `em_converged`, `em_lower_bound`) | **Direct** |
| **Factor Analysis** | *Not implemented* | Excluded |
| **PCA & Latent Dimensions** | Leak-free PCA preprocessing fit strictly on train patients (8, 12, 16 dims) | **Direct** |
| **Clustering Evaluation** | Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Score, Inertia | **Direct** |
| **Hierarchical Clustering** | *Not implemented* | **Explicitly Excluded** |
| **Markov Models** | Transition probability matrix estimation ($A_{ij}$) | **Direct** |
| **Hidden Markov Models** | `GaussianHMM` with continuous multi-dimensional emissions | **Direct** |
| **Maximum Likelihood for HMM** | EM/Baum-Welch parameter estimation on training sequences | **Direct** |
| **Forward & Backward Algorithms**| Custom log-space causal forward recursion (filtering) | **Direct** |
| **Sum-Product Algorithm** | Log-sum-exp recursive forward message passing | **Related** |
| **Scaling Factors** | Numerically stable log-space normalization ($c_t = \text{logsumexp}$) | **Related** |
| **Viterbi Algorithm** | *Not implemented* | Excluded |
| **Linear Dynamical Systems** | *Not implemented* | Excluded |

> [!IMPORTANT]
> **Strict Syllabus Boundary**: No Agglomerative Clustering, Hierarchical Clustering, Ward Linkage, Dendrograms, Factor Analysis, Viterbi Decoding, or Linear Dynamical Systems are implemented or referenced.

---

## 2. Dataset and Feature Engineering

### Verified Raw Data Assets
- `Data_Entry_2017_v2020.csv`: 112,120 chest X-ray records across 30,805 unique patients.
- Three real image archives in `images/`:
  - `images_001.tar.gz` (5,000 images, Patient IDs 1 – 1,335)
  - `images_002.tar.gz` (10,001 images, Patient IDs 1,336 – 3,923)
  - `images_003.tar.gz` (10,001 images, Patient IDs 3,923 – 6,585)
- **3,275 longitudinal patients ($\ge 2$ visits)** have **100% complete image series** across all visits (21,683 scans total).

### Longitudinal Cohort Sampling & Patient-Wise Split
To prevent data leakage across visits of the same patient:
1. Candidate longitudinal patients ($\ge 2$ scans with complete images) are deterministically sampled to form a reproducible cohort of **1,500 patients** (RandomState Seed = 42).
2. The cohort is partitioned strictly at the patient level:
   - **Training Set (70%)**: 1,050 patients (~6,900 visits)
   - **Validation Set (15%)**: 225 patients (~1,500 visits)
   - **Test Set (15%)**: 225 patients (~1,500 visits)
3. No patient's visits ever span multiple splits.

### Multimodal Feature Extraction
1. **Real Image HOG Features**:
   - Each scan's PNG image is decompressed, converted to grayscale, resized to $128 \times 128$, and processed using `skimage.feature.hog`:
     - Orientations: 8
     - Pixels per cell: $(16, 16)$
     - Cells per block: $(2, 2)$
     - Yields a 1,568-dimensional visual gradient vector capturing cardiac contour and parenchymal opacities.
2. **Clinical and Acquisition Metadata**:
   - 14 binary thoracic pathology indicators (`Atelectasis`, `Cardiomegaly`, `Consolidation`, `Edema`, `Effusion`, `Emphysema`, `Fibrosis`, `Hernia`, `Infiltration`, `Mass`, `Nodule`, `Pleural_Thickening`, `Pneumonia`, `Pneumothorax`).
   - `Finding_Count`: Number of concurrent findings (0 for `No Finding`).
   - Demographics: `Patient Age` (continuous), `Patient Sex` (binary: 1=M, 0=F).
   - Acquisition parameters: `View Position` (binary: 1=PA, 0=AP), `Follow-up #` (integer visit count).
3. **Feature Caching**:
   - Extracted features and patient split masks are cached to `cxr8_features_cohort1500.npz`. All subsequent model runs load in $<1$ second.
4. **Leak-Free Scaler & PCA**:
   - `StandardScaler` is fit strictly on training patients.
   - `PCA` is fit strictly on the standardized training patients ($D \in \{8, 12, 16\}$ components).
   - Validation and test sets are transformed using the fitted training scaler and PCA.

---

## 3. Algorithms and Optimization Suites

### Unit 3: K-Means & GMM Hyperparameter Optimization
**MLflow Experiment**: `CXR8_Unit3_KMeans_GMM_Optimization`

#### A. K-Means Clustering (`sklearn.cluster.KMeans`)
- **Grid**:
  - `n_clusters`: `[2, 3, 4, 5, 6]`
  - `init`: `["k-means++", "random"]`
  - `n_init`: `[10, 20]`
  - Total: 20 nested child trials (`Trial_1` to `Trial_20`)
- **Parent Run**: `KMeans_Parent_Hyperparameter_Optimization`
- **Logged Metrics**: `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`, `inertia`, `num_clusters_found`, `fit_time_seconds`.
- **Selection Rule**: Selected by highest validation `silhouette_score`.

#### B. Gaussian Mixture Models (`sklearn.mixture.GaussianMixture`)
- **Grid**:
  - `n_components`: `[2, 3, 4, 5, 6]`
  - `covariance_type`: `["full", "tied", "diag", "spherical"]`
  - `reg_covar`: `[1e-6, 1e-4, 1e-2]`
  - Total: 60 nested child trials (`Trial_1` to `Trial_60`)
- **Parent Run**: `GMM_Parent_Hyperparameter_Optimization`
- **EM Diagnostics Logged**:
  - `em_n_iter`: Iterations required for EM to terminate
  - `em_converged`: Boolean indicator of EM convergence
  - `em_lower_bound`: Final log-likelihood lower bound (ELBO)
- **Model Fit & Cluster Metrics**: `train_bic`, `train_aic`, `validation_log_likelihood`, `silhouette_score`, `davies_bouldin_index`, `calinski_harabasz_score`.
- **Selection Rule**: Candidates within 2% of maximum validation log-likelihood; selected based on lowest training BIC (parsimony criterion).

---

### Unit 4: Hidden Markov Models with Causal Forward Filtering
**MLflow Experiment**: `CXR8_Unit4_HMM_Forward_Optimization`
**Model**: `hmmlearn.hmm.GaussianHMM`

#### Hyperparameter Grid
- `n_components`: `[2, 3, 4, 5]`
- `covariance_type`: `["diag", "spherical"]`
- `n_pca_components`: `[8, 12, 16]`
- Total: 24 nested child trials (`Trial_1` to `Trial_24`)
- **Parent Run**: `HMM_Parent_Hyperparameter_Optimization`

#### Sequence Integrity & Causal Forward Recursion
1. **Patient Boundary Reset**: Sequence boundaries are enforced using `lengths`. For every patient sequence $x_0, \dots, x_{T-1}$, the forward recursion resets to $\ln \pi$:
   $$\alpha_0(j) = \ln \pi_j + \ln b_j(x_0)$$
   $$c_0 = \operatorname{logsumexp}_k \alpha_0(k), \quad P(S_0 = j \mid x_0) = \exp(\alpha_0(j) - c_0)$$
2. **Causal Forward Recursion**:
   $$\alpha_t(j) = \ln b_j(x_t) + \operatorname{logsumexp}_{i} \left( \alpha_{t-1}(i) + \ln A_{ij} \right)$$
   $$c_t = \operatorname{logsumexp}_k \alpha_t(k), \quad P(S_t = j \mid x_{0:t}) = \exp(\alpha_t(j) - c_t)$$
   - Conditions strictly on past and current observations $x_{0:t}$.
   - No backward pass, no smoothing, and no future observation leakage.
   - Sequence log-likelihood: $\ln P(x_{0:T-1}) = \operatorname{logsumexp}_k \alpha_{T-1}(k)$.
3. **Latent State vs Pathology Evaluation**:
   - Hidden states are treated as latent radiographic progression patterns rather than diagnostic labels.
   - Evaluated via Chi-Square test of independence ($\chi^2$ statistic and p-value) against binary pathology indicators across all visits.
4. **Strict Test Set Isolation**:
   - Hyperparameter selection uses validation forward log-likelihood and training BIC only.
   - The test set is evaluated **exactly once** on the optimal model for final unbiased generalization reporting.

---

## 4. How to Run the Project

### Prerequisites
Python 3.12 with required libraries:
```powershell
python -m pip install mlflow hmmlearn scikit-image scikit-learn pandas numpy scipy pillow
```

### Step 1: Data Preparation & Feature Extraction
Builds the 1,500-patient cohort, extracts real HOG features and clinical metadata, and caches to `cxr8_features_cohort1500.npz`:
```powershell
python cxr8_data_helper.py
```

### Step 2: Run Unit 3 K-Means & GMM Optimization
Executes 20 K-Means trials and 60 GMM trials with EM diagnostics:
```powershell
python unit3_clustering_mlflow.py
```

### Step 3: Run Unit 4 HMM Causal Forward Optimization
Executes 24 GaussianHMM trials across components, covariance structures, and PCA dimensions:
```powershell
python unit4_hmm_mlflow.py
```

### Step 4: Launch MLflow Tracking UI
Start the local MLflow tracking server to inspect experiments, parent runs, nested trials, parameter comparisons, and metric curves:
```powershell
mlflow ui --host 127.0.0.1 --port 5000
```
Open your browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)
