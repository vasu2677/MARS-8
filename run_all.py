"""
run_all.py
==========
Master Execution Script for CXR8 Longitudinal MLflow Suite.
Runs the entire machine learning project in a single command:
  - Unit 3: K-Means Clustering & Gaussian Mixture Models (GMM)
  - Unit 4: Hidden Markov Models (HMM) with Causal Forward Filtering
  - Unit 5: Ensemble Learning & Tree-Based Methods (Random Forest & XGBoost / Gradient Boosting)

Usage:
    python run_all.py
"""

import sys
import time

print("=" * 75)
print("  CXR8 LONGITUDINAL MACHINE LEARNING HYPERPARAMETER OPTIMIZATION SUITE")
print("  Targeting: Unit 3 (Clustering/GMM), Unit 4 (HMM), Unit 5 (Ensembles)")
print("=" * 75)

# Step 0: Check dependencies
required_packages = [
    ("mlflow", "mlflow"),
    ("hmmlearn", "hmmlearn"),
    ("sklearn", "scikit-learn"),
    ("skimage", "scikit-image"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "pillow"),
]

missing = []
for module_name, pip_name in required_packages:
    try:
        __import__(module_name)
    except ImportError:
        missing.append(pip_name)

if missing:
    print(f"\n[ERROR] Missing required packages: {', '.join(missing)}")
    print(f"Please install them using:\n    pip install {' '.join(missing)}\n")
    sys.exit(1)

import cxr8_data_helper as helper
import unit3_clustering_mlflow as unit3
import unit4_hmm_mlflow as unit4
import unit5_ensemble_mlflow as unit5

total_start = time.time()

# Step 1: Load or build CXR8 cohort data
print("\n>>> STEP 1: Verifying Data Assets & Loading 1,500-Patient Cohort...")
t0 = time.time()
data = helper.load_cxr8_data()
print(f">>> Cohort loaded successfully in {time.time() - t0:.2f}s.")

# Step 2: Run Unit 3 K-Means & GMM Optimization
print("\n>>> STEP 2: Running Unit 3 (K-Means & Gaussian Mixture Models)...")
t0 = time.time()
X_train_pca, X_val_pca, _, _, _ = helper.get_preprocessed_features(data, n_pca_components=12)

best_km_params, best_km_score = unit3.run_kmeans_optimization(X_train_pca, X_val_pca)
best_gmm_rec = unit3.run_gmm_optimization(X_train_pca, X_val_pca)
unit3_time = time.time() - t0
print(f">>> Unit 3 finished in {unit3_time:.1f}s.")

# Step 3: Run Unit 4 Hidden Markov Model Causal Forward Optimization
print("\n>>> STEP 3: Running Unit 4 (Hidden Markov Models with Causal Forward Filtering)...")
t0 = time.time()
best_hmm_rec = unit4.run_hmm_optimization()
unit4_time = time.time() - t0
print(f">>> Unit 4 finished in {unit4_time:.1f}s.")

# Step 4: Run Unit 5 Ensemble Learning & Tree-Based Optimization
print("\n>>> STEP 4: Running Unit 5 (Ensemble Methods: Random Forest & Gradient Boosting)...")
t0 = time.time()
unit5_results = unit5.run_ensemble_optimization()
unit5_time = time.time() - t0
print(f">>> Unit 5 finished in {unit5_time:.1f}s.")

total_elapsed = time.time() - total_start

# Step 5: Final Summary Report
print("\n" + "=" * 75)
print("                        FINAL PROJECT REVIEW SUMMARY")
print("=" * 75)
print(f"Total Execution Time: {total_elapsed:.1f} seconds ({total_elapsed/60:.2f} minutes)")

print("\n1. UNIT 3: K-MEANS CLUSTERING")
print(f"   - Best Parameters : {best_km_params}")
print(f"   - Silhouette Score: {best_km_score:.4f}")

print("\n2. UNIT 3: GAUSSIAN MIXTURE MODEL (GMM)")
print(f"   - Best Parameters : {best_gmm_rec['params']}")
print(f"   - Val Log-Likelihood: {best_gmm_rec['val_log_lik']:.4f}")
print(f"   - Train BIC       : {best_gmm_rec['train_bic']:.1f}")
print(f"   - EM Iterations   : {best_gmm_rec['em_n_iter']} (Lower Bound ELBO: {best_gmm_rec['em_lower_bound']:.3f})")

print("\n3. UNIT 4: HIDDEN MARKOV MODEL (HMM)")
print(f"   - Best Parameters : {best_hmm_rec['params']}")
print(f"   - Val Forward LL  : {best_hmm_rec['val_fwd_ll']:.4f}")
print(f"   - Train BIC       : {best_hmm_rec['train_bic']:.1f}")
print("   - Inference       : Log-Space Causal Forward Filtering with Patient Boundary Resets")

test_res = unit5_results["test_evaluation"]
print("\n4. UNIT 5: ENSEMBLE LEARNING & TREE-BASED CLASSIFICATION")
print(f"   - Champion Model  : {test_res['winning_name']}")
print(f"   - Best Parameters : {test_res['winning_params']}")
print(f"   - Val Macro F1    : {test_res['winning_val_f1']:.4f}")
print(f"   - Test Macro F1   : {test_res['test_metrics']['macro_f1']:.4f}")
print(f"   - Test Balanced Acc: {test_res['test_metrics']['balanced_accuracy']:.4f}")
print(f"   - Test ROC AUC    : {test_res['test_metrics']['roc_auc']:.4f}")
print("=" * 75)
print("\n>>> LAUNCH MLFLOW FRONTEND DASHBOARD:")
print("Run the following command to open the visual UI in your browser:")
print("    mlflow ui --host 127.0.0.1 --port 5000")
print("Then open: http://127.0.0.1:5000\n")
