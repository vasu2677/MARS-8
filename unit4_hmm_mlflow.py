"""
unit4_hmm_mlflow.py
===================
Unit 4: Gaussian Hidden Markov Model (HMM) Hyperparameter Optimization with Causal Forward Filtering.
Clean execution with MLflow tracking and causal sequence inference.
"""

import itertools
from typing import Any

import mlflow
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import chi2_contingency

import cxr8_data_helper as helper

EXPERIMENT_NAME = "CXR8_Unit4_HMM_Forward_Optimization"


def causal_forward_filtering(model, X, lengths):
    """
    Computes causal filtered hidden state probabilities P(S_t = j | x_{0:t})
    and total log-likelihood in log space using logsumexp.
    Resets initial distribution at each patient sequence boundary.
    """
    log_pi = np.log(np.maximum(model.startprob_, 1e-12))
    log_A = np.log(np.maximum(model.transmat_, 1e-12))
    n_components = model.n_components
    framelogprob = model._compute_log_likelihood(X)
    n_samples = len(X)

    filtered_probs = np.zeros((n_samples, n_components), dtype=np.float64)
    total_log_likelihood = 0.0

    start_idx = 0
    for seq_len in lengths:
        end_idx = start_idx + seq_len
        seq_emissions = framelogprob[start_idx:end_idx]

        # Step 0: reset to log(pi)
        alpha_t = log_pi + seq_emissions[0]
        c_t = logsumexp(alpha_t)
        filtered_probs[start_idx] = np.exp(alpha_t - c_t)

        # Steps 1 to T-1: causal forward recursion
        for t in range(1, seq_len):
            trans_message = logsumexp(alpha_t[:, None] + log_A, axis=0)
            alpha_t = seq_emissions[t] + trans_message
            c_t = logsumexp(alpha_t)
            filtered_probs[start_idx + t] = np.exp(alpha_t - c_t)

        total_log_likelihood += logsumexp(alpha_t)
        start_idx = end_idx

    per_obs_log_likelihood = float(total_log_likelihood / max(1, n_samples))
    return filtered_probs, float(total_log_likelihood), per_obs_log_likelihood


def compute_hmm_information_criteria(model, train_total_log_lik, n_samples, n_pca_components):
    k_comp = model.n_components
    d_dim = n_pca_components
    if model.covariance_type == "diag":
        p = (k_comp - 1) + k_comp * (k_comp - 1) + 2 * k_comp * d_dim
    elif model.covariance_type == "spherical":
        p = (k_comp - 1) + k_comp * (k_comp - 1) + k_comp * d_dim + k_comp
    else:
        p = (k_comp - 1) + k_comp * (k_comp - 1) + 2 * k_comp * d_dim

    aic = 2.0 * p - 2.0 * train_total_log_lik
    bic = p * np.log(n_samples) - 2.0 * train_total_log_lik
    return float(aic), float(bic)


def compute_latent_pathology_contingency(filtered_probs, y_has_pathology, n_components):
    latent_states = np.argmax(filtered_probs, axis=1)
    contingency = np.zeros((n_components, 2), dtype=np.int32)
    for k in range(n_components):
        mask_k = (latent_states == k)
        contingency[k, 0] = int(np.sum(mask_k & (y_has_pathology == 0.0)))
        contingency[k, 1] = int(np.sum(mask_k & (y_has_pathology == 1.0)))
    if contingency.sum() > 0 and (contingency.sum(axis=1) > 0).sum() >= 2:
        try:
            chi2_stat, p_val, _, _ = chi2_contingency(contingency)
            return float(chi2_stat), float(p_val)
        except (ValueError, ZeroDivisionError):
            return 0.0, 1.0
    return 0.0, 1.0


def run_hmm_optimization():
    print("\n" + "=" * 70)
    print("3. GAUSSIAN HMM CAUSAL FORWARD HYPERPARAMETER OPTIMIZATION (UNIT 4)")
    print("=" * 70)

    param_grid = {
        "n_components": [2, 3, 4, 5],
        "covariance_type": ["diag", "spherical"],
        "n_pca_components": [8, 12, 16],
    }

    data = helper.load_cxr8_data()
    train_lengths = data['train_lengths']
    val_lengths = data['val_lengths']
    test_lengths = data['test_lengths']
    y_val = data['y_val']
    y_test = data['y_test']

    pca_cache = {}
    for n_pca in param_grid["n_pca_components"]:
        x_tr, x_va, x_te, _, _ = helper.get_preprocessed_features(
            data, n_pca_components=n_pca
        )
        pca_cache[n_pca] = {'X_train': x_tr, 'X_val': x_va, 'X_test': x_te}

    trials_record = []

    with mlflow.start_run(run_name="HMM_Parent_Hyperparameter_Optimization"):
        mlflow.set_tag("model_family", "GaussianHMM")
        mlflow.set_tag("syllabus_unit", "Unit 4: Hidden Markov Models")
        mlflow.set_tag("inference_type", "Log-Space Causal Forward Filtering")

        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())
        trial_models: dict[int, GaussianHMM] = {}
        trials_record: list[dict[str, Any]] = []

        for idx, values in enumerate(param_combinations, 1):
            params: dict[str, Any] = dict(zip(param_keys, values, strict=False))
            trial_name = f"Trial_{idx}"

            pca_k = int(params["n_pca_components"])
            x_tr = pca_cache[pca_k]['X_train']
            x_va = pca_cache[pca_k]['X_val']

            with mlflow.start_run(run_name=trial_name, nested=True):
                hmm = GaussianHMM(
                    n_components=int(params["n_components"]),
                    covariance_type=str(params["covariance_type"]),
                    n_iter=80,
                    tol=1e-3,
                    random_state=42,
                )
                hmm.fit(x_tr, lengths=train_lengths)

                _, train_tot_ll, train_fwd_ll = causal_forward_filtering(
                    hmm, x_tr, train_lengths
                )
                val_probs, _, val_fwd_ll = causal_forward_filtering(
                    hmm, x_va, val_lengths
                )

                aic, bic = compute_hmm_information_criteria(
                    hmm, train_tot_ll, len(x_tr), pca_k
                )
                chi2_stat, p_val = compute_latent_pathology_contingency(
                    val_probs, y_val, int(params["n_components"])
                )

                mlflow.log_params(params)
                mlflow.log_param("trial_number", idx)
                mlflow.log_metric("train_forward_log_likelihood", train_fwd_ll)
                mlflow.log_metric("validation_forward_log_likelihood", val_fwd_ll)
                mlflow.log_metric("train_bic", bic)
                mlflow.log_metric("train_aic", aic)
                mlflow.log_metric("state_pathology_chi2_stat", chi2_stat)

                trial_models[idx] = hmm
                trials_record.append({
                    "trial_idx": idx,
                    "params": params,
                    "val_fwd_ll": float(val_fwd_ll),
                    "train_bic": float(bic),
                    "train_aic": float(aic),
                    "chi2_stat": float(chi2_stat),
                })

                print(
                    f"[{trial_name:8s}] states={int(params['n_components']):1d} | cov={params['covariance_type']!s:9s} | PCA={pca_k:2d} | "
                    f"Val Fwd LL: {val_fwd_ll:7.3f} | Train BIC: {bic:10.1f} | Chi2: {chi2_stat:8.2f} (p={p_val:.1e})"
                )

        max_val_ll: float = max(float(r["val_fwd_ll"]) for r in trials_record)
        ll_threshold: float = (
            max_val_ll - abs(max_val_ll) * 0.02
            if max_val_ll < 0
            else max_val_ll * 0.98
        )
        qualifying = [r for r in trials_record if float(r["val_fwd_ll"]) >= ll_threshold]
        best_rec = min(
            qualifying if qualifying else trials_record,
            key=lambda r: float(r["train_bic"]),
        )
        best_model = trial_models[int(best_rec["trial_idx"])]

        best_pca_k = int(best_rec["params"]["n_pca_components"])
        best_n_comp = int(best_rec["params"]["n_components"])
        x_te = pca_cache[best_pca_k]['X_test']
        test_probs, _, test_fwd_ll = causal_forward_filtering(
            best_model, x_te, test_lengths
        )
        test_chi2, _ = compute_latent_pathology_contingency(
            test_probs, y_test, best_n_comp
        )

        mlflow.log_params(
            {f"best_{k}": v for k, v in best_rec["params"].items()}
        )
        mlflow.log_metric(
            "best_validation_forward_log_likelihood", float(best_rec["val_fwd_ll"])
        )
        mlflow.log_metric("best_train_bic", float(best_rec["train_bic"]))
        mlflow.log_metric("best_train_aic", float(best_rec["train_aic"]))
        mlflow.log_metric("test_forward_log_likelihood", float(test_fwd_ll))
        mlflow.log_metric("test_state_pathology_chi2_stat", float(test_chi2))

    print(
        f"\nOptimization complete. Best Params: {best_rec['params']} | Best Val Forward Log-Likelihood: {best_rec['val_fwd_ll']:.4f} | Train BIC: {best_rec['train_bic']:.1f} | Test Forward Log-Likelihood: {test_fwd_ll:.4f}\n"
    )
    return best_rec


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_hmm_optimization()
