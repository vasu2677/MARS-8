"""
unit3_clustering_mlflow.py
==========================
Unit 3: K-Means Clustering and Gaussian Mixture Models (GMM) Hyperparameter Optimization.
Clean execution with MLflow tracking and model logging.
"""

import itertools
from typing import Any, cast

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

import cxr8_data_helper as helper

EXPERIMENT_NAME = "CXR8_Unit3_KMeans_GMM_Optimization"


def run_kmeans_optimization(X_train, X_val):
    print("\n" + "=" * 70)
    print("1. K-MEANS HYPERPARAMETER OPTIMIZATION (UNIT 3)")
    print("=" * 70)

    param_grid = {
        "n_clusters": [2, 3, 4, 5, 6],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
    }

    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_model = None
    best_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name="KMeans_Parent_Hyperparameter_Optimization"):
        mlflow.set_tag("model_family", "KMeans")
        mlflow.set_tag("syllabus_unit", "Unit 3: Mixture Models and EM")

        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())

        for idx, values in enumerate(param_combinations, 1):
            params: dict[str, Any] = dict(zip(param_keys, values, strict=False))
            trial_name = f"Trial_{idx}"

            with mlflow.start_run(run_name=trial_name, nested=True):
                km = KMeans(
                    n_clusters=int(params["n_clusters"]),
                    init=cast(Any, params["init"]),
                    n_init=int(params["n_init"]),
                    random_state=42,
                )
                km.fit(X_train)

                val_labels = km.predict(X_val)
                n_found = len(np.unique(val_labels))

                if n_found >= 2:
                    val_sil = float(silhouette_score(X_val, val_labels))
                    val_db = float(davies_bouldin_score(X_val, val_labels))
                    val_ch = float(calinski_harabasz_score(X_val, val_labels))
                else:
                    val_sil = -1.0
                    val_db = 999.0
                    val_ch = 0.0

                inertia = float(km.inertia_)

                mlflow.log_params(params)
                mlflow.log_param("trial_number", idx)
                mlflow.log_metric("silhouette_score", val_sil)
                mlflow.log_metric("davies_bouldin_index", val_db)
                mlflow.log_metric("calinski_harabasz_score", val_ch)
                mlflow.log_metric("inertia", inertia)
                mlflow.log_metric("num_clusters_found", n_found)

                print(
                    f"[{trial_name:8s}] k={int(params['n_clusters']):1d} | init={params['init']!s:9s} | n_init={int(params['n_init']):2d} | "
                    f"Val Sil: {val_sil:6.4f} | DB: {val_db:6.4f} | CH: {val_ch:8.1f} | Inertia: {inertia:10.1f}"
                )

                if val_sil > best_score:
                    best_score = val_sil
                    best_params = params
                    best_model = km
                    best_metrics = {
                        "best_validation_silhouette": val_sil,
                        "best_davies_bouldin": val_db,
                        "best_calinski_harabasz": val_ch,
                        "best_inertia": inertia,
                    }

        if best_params:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(best_metrics)
        if best_model is not None:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

    print(
        f"\nOptimization complete. Best Params: {best_params} | Best Silhouette Score: {best_score:.4f}\n"
    )
    return best_params, best_score


def run_gmm_optimization(X_train, X_val):
    print("=" * 70)
    print("2. GAUSSIAN MIXTURE MODEL HYPERPARAMETER OPTIMIZATION (UNIT 3)")
    print("=" * 70)

    param_grid = {
        "n_components": [2, 3, 4, 5, 6],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "reg_covar": [1e-6, 1e-4, 1e-2],
    }

    trials_record = []

    with mlflow.start_run(run_name="GMM_Parent_Hyperparameter_Optimization"):
        mlflow.set_tag("model_family", "GaussianMixture")
        mlflow.set_tag("syllabus_unit", "Unit 3: Mixture Models and EM")
        mlflow.set_tag("learning_algorithm", "Expectation-Maximization (EM)")

        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())
        trial_gmm_models: dict[int, GaussianMixture] = {}
        trials_record: list[dict[str, Any]] = []

        for idx, values in enumerate(param_combinations, 1):
            params: dict[str, Any] = dict(zip(param_keys, values, strict=False))
            trial_name = f"Trial_{idx}"

            with mlflow.start_run(run_name=trial_name, nested=True):
                gmm = GaussianMixture(
                    n_components=int(params["n_components"]),
                    covariance_type=cast(Any, params["covariance_type"]),
                    reg_covar=float(params["reg_covar"]),
                    max_iter=150,
                    random_state=42,
                )
                gmm.fit(X_train)

                em_n_iter = int(gmm.n_iter_)
                em_converged = 1.0 if gmm.converged_ else 0.0
                em_lower_bound = float(gmm.lower_bound_)

                train_bic = float(gmm.bic(X_train))
                train_aic = float(gmm.aic(X_train))
                val_log_lik = float(gmm.score(X_val))

                val_labels = gmm.predict(X_val)
                n_found = len(np.unique(val_labels))
                val_sil = (
                    float(silhouette_score(X_val, val_labels))
                    if n_found >= 2
                    else -1.0
                )
                val_db = (
                    float(davies_bouldin_score(X_val, val_labels))
                    if n_found >= 2
                    else 999.0
                )
                val_ch = (
                    float(calinski_harabasz_score(X_val, val_labels))
                    if n_found >= 2
                    else 0.0
                )

                mlflow.log_params(params)
                mlflow.log_param("trial_number", idx)
                mlflow.log_metric("em_n_iter", em_n_iter)
                mlflow.log_metric("em_converged", em_converged)
                mlflow.log_metric("em_lower_bound", em_lower_bound)
                mlflow.log_metric("train_bic", train_bic)
                mlflow.log_metric("train_aic", train_aic)
                mlflow.log_metric("validation_log_likelihood", val_log_lik)
                mlflow.log_metric("silhouette_score", val_sil)
                mlflow.log_metric("davies_bouldin_index", val_db)
                mlflow.log_metric("calinski_harabasz_score", val_ch)

                trial_gmm_models[idx] = gmm
                trials_record.append({
                    "trial_idx": idx,
                    "params": params,
                    "val_log_lik": float(val_log_lik),
                    "train_bic": float(train_bic),
                    "train_aic": float(train_aic),
                    "em_n_iter": int(em_n_iter),
                    "em_lower_bound": float(em_lower_bound),
                    "val_sil": float(val_sil),
                })

                print(
                    f"[{trial_name:8s}] comp={int(params['n_components']):1d} | cov={params['covariance_type']!s:9s} | reg={float(params['reg_covar']):8.1e} | "
                    f"EM iters={em_n_iter:2d} conv={int(em_converged)} | Val LL: {val_log_lik:7.3f} | BIC: {train_bic:10.1f}"
                )

        max_val_ll: float = max(float(r["val_log_lik"]) for r in trials_record)
        ll_threshold: float = (
            max_val_ll - abs(max_val_ll) * 0.02
            if max_val_ll < 0
            else max_val_ll * 0.98
        )
        qualifying = [
            r for r in trials_record if float(r["val_log_lik"]) >= ll_threshold
        ]
        best_gmm_rec = min(
            qualifying if qualifying else trials_record,
            key=lambda r: float(r["train_bic"]),
        )
        best_gmm_model = trial_gmm_models[int(best_gmm_rec["trial_idx"])]

        mlflow.log_params(
            {f"best_{k}": v for k, v in best_gmm_rec["params"].items()}
        )
        mlflow.log_metric(
            "best_validation_log_likelihood", float(best_gmm_rec["val_log_lik"])
        )
        mlflow.log_metric("best_train_bic", float(best_gmm_rec["train_bic"]))
        mlflow.log_metric("best_train_aic", float(best_gmm_rec["train_aic"]))
        mlflow.log_metric("best_em_n_iter", int(best_gmm_rec["em_n_iter"]))
        mlflow.log_metric(
            "best_em_lower_bound", float(best_gmm_rec["em_lower_bound"])
        )
        mlflow.log_metric("best_silhouette_score", float(best_gmm_rec["val_sil"]))
        if best_gmm_model is not None:
            mlflow.sklearn.log_model(best_gmm_model, artifact_path="model")

    print(
        f"\nOptimization complete. Best Params: {best_gmm_rec['params']} | Best Val Log-Likelihood: {best_gmm_rec['val_log_lik']:.4f} | Train BIC: {best_gmm_rec['train_bic']:.1f}\n"
    )
    return best_gmm_rec


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)
    data = helper.load_cxr8_data()
    X_train_pca, X_val_pca, _, _, _ = (
        helper.get_preprocessed_features(data, n_pca_components=12)
    )

    run_kmeans_optimization(X_train_pca, X_val_pca)
    run_gmm_optimization(X_train_pca, X_val_pca)
