"""
unit5_ensemble_mlflow.py
========================
Unit 5: Ensemble Learning & Tree-Based Methods (Random Forest & XGBoost / Gradient Boosting).
Longitudinal thoracic pathology prediction with strict zero patient-level leakage,
class-imbalance handling, nested MLflow trial tracking, and disciplined test set isolation.

Key Enhancement:
- Multi-dataset representation: Each model family is trained on a distinct, domain-specific
  dataset input to evaluate different clinical paradigms:
    1. Dummy Majority Floor   -> Label Prior Distribution Baseline
    2. Logistic Regression    -> Clinical & Demographic Metadata Only (4 features)
    3. Random Forest (Bagging)-> Radiographic Visual HOG Embeddings Only (12 PCA components)
    4. XGBoost (Boosting)     -> Multimodal Engineered Fusion (22 features: Visual PCA + Clinical + Interactions)
"""

import itertools
import os
from typing import Any, cast

import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier  # type: ignore
    HAS_XGBOOST = False

try:
    import mlflow.xgboost
    HAS_MLFLOW_XGB = True
except ImportError:
    HAS_MLFLOW_XGB = False

import cxr8_data_helper as helper

EXPERIMENT_NAME = "CXR8_Unit5_Ensemble_Optimization"


def configure_mlflow() -> str:
    """Configures MLflow tracking store and selects or restores the Unit 5 MLflow experiment."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    # Respect MLFLOW_TRACKING_URI if set, otherwise point to project sqlite database if present
    if "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        db_candidate = helper.find_file("mlflow.db")
        if not os.path.isabs(db_candidate):
            db_candidate = os.path.abspath(db_candidate)
        db_uri_path = db_candidate.replace("\\", "/")
        mlflow.set_tracking_uri(f"sqlite:///{db_uri_path}")

    client = mlflow.tracking.MlflowClient()
    try:
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is not None:
            if exp.lifecycle_stage == "deleted":
                client.restore_experiment(exp.experiment_id)
            mlflow.set_experiment(experiment_id=exp.experiment_id)
        else:
            mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception:
        mlflow.set_experiment(EXPERIMENT_NAME)
        
    return mlflow.get_tracking_uri()


def log_dataset_safely(X: np.ndarray, y: np.ndarray, dataset_name: str, context: str = "training") -> None:
    """Logs dataset metadata and schema safely into MLflow."""
    try:
        ds = mlflow.data.from_numpy(features=X, targets=y, name=dataset_name)
        mlflow.log_input(ds, context=context)
    except Exception:
        pass


def log_sklearn_model_safely(model: Any, artifact_path: str = "model") -> None:
    """Logs scikit-learn model with explicit skops trusted types and cloudpickle fallback."""
    try:
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )
    except Exception:
        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                serialization_format="cloudpickle",
            )
        except Exception as e:
            print(f"[WARNING] Could not log sklearn model artifact: {e}")


def log_xgboost_model_safely(model: Any, artifact_path: str = "model") -> None:
    """Logs XGBoost model safely to MLflow."""
    try:
        if HAS_XGBOOST and HAS_MLFLOW_XGB:
            mlflow.xgboost.log_model(model, artifact_path=artifact_path)
        else:
            log_sklearn_model_safely(model, artifact_path=artifact_path)
    except Exception as e:
        print(f"[WARNING] Could not log XGBoost model artifact: {e}")


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    """Computes Macro F1, Balanced Accuracy, Standard Accuracy, and ROC AUC."""
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5
    else:
        auc = 0.5

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "roc_auc": auc,
    }


def prepare_distinct_model_datasets(data: dict[str, Any]) -> dict[str, Any]:
    """
    Builds 3 distinct, leak-free feature datasets tailored for different model paradigms:
    
    1. Clinical & Demographic Metadata Dataset (4 features):
       - Features: Patient_Age, Patient_Sex, View_Position, Followup_Num
       - Used by: Logistic Regression Baseline
       - Rationale: Establishes the non-imaging demographic floor.
       
    2. Radiographic Visual HOG Embeddings Dataset (12 PCA components):
       - Features: 12 PCA components fit strictly on 1,568 raw HOG visual features.
       - Used by: Random Forest (Bagging)
       - Rationale: Evaluates pure image visual pattern recognition across decorrelated trees.
       
    3. Multimodal Engineered Fusion Dataset (22 features):
       - Features: 16 Visual PCA components + 4 Clinical Demographics + 2 Non-linear Interaction terms
         ('Age_x_Followup', 'Visual_Gradient_Norm')
       - Used by: XGBoost / Gradient Boosting (Boosting)
       - Rationale: Evaluates boosting's ability to exploit heterogeneous multimodal interactions.
    """
    X_train_raw = data['X_train']
    X_val_raw = data['X_val']
    X_test_raw = data['X_test']
    
    y_train = data['y_train'].astype(np.int32)
    y_val = data['y_val'].astype(np.int32)
    y_test = data['y_test'].astype(np.int32)
    
    # -------------------------------------------------------------
    # Dataset 1: Clinical Demographics Only (4 features)
    # -------------------------------------------------------------
    if X_train_raw.shape[1] >= 1587:
        raw_tr_clin = X_train_raw[:, 1583:1587]
        raw_val_clin = X_val_raw[:, 1583:1587]
        raw_ts_clin = X_test_raw[:, 1583:1587]
        raw_tr_vis = X_train_raw[:, :1568]
        raw_val_vis = X_val_raw[:, :1568]
        raw_ts_vis = X_test_raw[:, :1568]
    else:
        raw_tr_clin = X_train_raw[:, 15:19]
        raw_val_clin = X_val_raw[:, 15:19]
        raw_ts_clin = X_test_raw[:, 15:19]
        raw_tr_vis = X_train_raw[:, :15]
        raw_val_vis = X_val_raw[:, :15]
        raw_ts_vis = X_test_raw[:, :15]
        
    scaler_clin = StandardScaler()
    X_train_clin = scaler_clin.fit_transform(raw_tr_clin)
    X_val_clin = scaler_clin.transform(raw_val_clin)
    X_test_clin = scaler_clin.transform(raw_ts_clin)
    clin_names = ["Patient_Age", "Patient_Sex", "View_Position", "Followup_Num"]
    
    # -------------------------------------------------------------
    # Dataset 2: Visual HOG Embeddings Only (12 PCA components)
    # -------------------------------------------------------------
    scaler_vis = StandardScaler()
    X_train_vis_sc = scaler_vis.fit_transform(raw_tr_vis)
    X_val_vis_sc = scaler_vis.transform(raw_val_vis)
    X_test_vis_sc = scaler_vis.transform(raw_ts_vis)
    
    pca_12 = PCA(n_components=12, random_state=42)
    X_train_vis_pca = pca_12.fit_transform(X_train_vis_sc)
    X_val_vis_pca = pca_12.transform(X_val_vis_sc)
    X_test_vis_pca = pca_12.transform(X_test_vis_sc)
    vis_names = [f"Visual_HOG_PCA_{i+1:02d}" for i in range(12)]
    
    # -------------------------------------------------------------
    # Dataset 3: Multimodal Engineered Fusion (22 features)
    # -------------------------------------------------------------
    pca_16 = PCA(n_components=16, random_state=42)
    X_train_vis16 = pca_16.fit_transform(X_train_vis_sc)
    X_val_vis16 = pca_16.transform(X_val_vis_sc)
    X_test_vis16 = pca_16.transform(X_test_vis_sc)
    
    # Interaction 1: Age x Follow-up progression
    tr_age_follow = (X_train_clin[:, 0] * X_train_clin[:, 3]).reshape(-1, 1)
    val_age_follow = (X_val_clin[:, 0] * X_val_clin[:, 3]).reshape(-1, 1)
    ts_age_follow = (X_test_clin[:, 0] * X_test_clin[:, 3]).reshape(-1, 1)
    
    # Interaction 2: Visual Gradient Energy (L2 norm of top visual components)
    tr_vis_energy = np.linalg.norm(X_train_vis16[:, :8], axis=1, keepdims=True)
    val_vis_energy = np.linalg.norm(X_val_vis16[:, :8], axis=1, keepdims=True)
    ts_vis_energy = np.linalg.norm(X_test_vis16[:, :8], axis=1, keepdims=True)
    
    scaler_energy = StandardScaler()
    tr_vis_energy = scaler_energy.fit_transform(tr_vis_energy)
    val_vis_energy = scaler_energy.transform(val_vis_energy)
    ts_vis_energy = scaler_energy.transform(ts_vis_energy)
    
    X_train_multi = np.hstack([X_train_vis16, X_train_clin, tr_age_follow, tr_vis_energy])
    X_val_multi = np.hstack([X_val_vis16, X_val_clin, val_age_follow, val_vis_energy])
    X_test_multi = np.hstack([X_test_vis16, X_test_clin, ts_age_follow, ts_vis_energy])
    
    multi_names = (
        [f"Visual_PCA_{i+1:02d}" for i in range(16)]
        + clin_names
        + ["Inter_Age_x_Followup", "Visual_Gradient_Energy"]
    )
    
    return {
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "clinical": {
            "name": "Clinical_Demographics_Metadata",
            "modality": "Tabular Clinical / Demographics (Non-imaging)",
            "X_train": X_train_clin,
            "X_val": X_val_clin,
            "X_test": X_test_clin,
            "feature_names": clin_names,
            "feature_count": len(clin_names),
        },
        "visual": {
            "name": "Radiographic_Visual_HOG_Embeddings",
            "modality": "Radiographic Visual HOG Embeddings (PCA=12)",
            "X_train": X_train_vis_pca,
            "X_val": X_val_vis_pca,
            "X_test": X_test_vis_pca,
            "feature_names": vis_names,
            "feature_count": len(vis_names),
        },
        "multimodal": {
            "name": "Multimodal_Engineered_Fusion",
            "modality": "Multimodal Fusion (Visual PCA 16 + Demographics + Interaction)",
            "X_train": X_train_multi,
            "X_val": X_val_multi,
            "X_test": X_test_multi,
            "feature_names": multi_names,
            "feature_count": len(multi_names),
        },
    }


def run_baseline_suite(
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """
    Sub-suite 1: Baseline Floor Models.
    Evaluates:
      1. DummyClassifier (most frequent): Class Prior distribution floor.
      2. Balanced LogisticRegression: Evaluated on Clinical & Demographic Metadata Only.
    """
    print("\n" + "=" * 70)
    print("1. BASELINE BENCHMARK MODELS (UNIT 5)")
    print("=" * 70)

    y_train = datasets["y_train"]
    y_val = datasets["y_val"]
    clin = datasets["clinical"]
    
    X_train_clin = clin["X_train"]
    X_val_clin = clin["X_val"]

    results: dict[str, Any] = {}

    # 1. Dummy Majority Baseline Run
    with mlflow.start_run(run_name="1_Dummy_Majority_Baseline") as dummy_run:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_clin[:, :1], y_train)
        pred_val = dummy.predict(X_val_clin[:, :1])
        prob_val = dummy.predict_proba(X_val_clin[:, :1])[:, 1] if hasattr(dummy, "predict_proba") else None
        dummy_metrics = compute_classification_metrics(y_val, pred_val, prob_val)

        mlflow.set_tag("model_family", "DummyClassifier")
        mlflow.set_tag("model_name", "DummyMajorityClassifier")
        mlflow.set_tag("syllabus_unit", "Unit 5: Ensemble Learning & Tree-Based Methods")
        mlflow.set_tag("dataset_name", "Label_Prior_Distribution")
        mlflow.set_tag("dataset_modality", "Target Class Prior Floor")

        mlflow.log_params({
            "model": "DummyClassifier",
            "model_family": "DummyClassifier",
            "strategy": "most_frequent",
            "dataset_input": "Label_Prior_Distribution",
            "feature_count": 0,
        })
        
        # Standardized metrics for MLflow Model Compare
        mlflow.log_metrics({
            "val_macro_f1": dummy_metrics["macro_f1"],
            "val_balanced_accuracy": dummy_metrics["balanced_accuracy"],
            "val_roc_auc": dummy_metrics["roc_auc"],
            "val_accuracy": dummy_metrics["accuracy"],
            **{f"val_{k}": v for k, v in dummy_metrics.items()}
        })
        log_sklearn_model_safely(dummy, artifact_path="dummy_majority_model")

        print(
            f"[Dummy_Maj] (Input: Prior Labels) | "
            f"Val Macro F1: {dummy_metrics['macro_f1']:6.4f} | "
            f"Bal Acc: {dummy_metrics['balanced_accuracy']:6.4f} | "
            f"Acc: {dummy_metrics['accuracy']:6.4f} | "
            f"AUC: {dummy_metrics['roc_auc']:6.4f}"
        )
        results["dummy"] = dummy_metrics
        results["dummy_run_id"] = dummy_run.info.run_id

    # 2. Balanced Logistic Regression on Clinical Metadata
    with mlflow.start_run(run_name="2_Clinical_Logistic_Regression") as lr_run:
        log_dataset_safely(X_train_clin, y_train, dataset_name=clin["name"])
        
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(X_train_clin, y_train)
        pred_val = lr.predict(X_val_clin)
        prob_val = lr.predict_proba(X_val_clin)[:, 1]
        lr_metrics = compute_classification_metrics(y_val, pred_val, prob_val)

        mlflow.set_tag("model_family", "LogisticRegression")
        mlflow.set_tag("model_name", "ClinicalLogisticRegression")
        mlflow.set_tag("syllabus_unit", "Unit 5: Ensemble Learning & Tree-Based Methods")
        mlflow.set_tag("dataset_name", clin["name"])
        mlflow.set_tag("dataset_modality", clin["modality"])
        mlflow.set_tag("feature_count", str(clin["feature_count"]))

        mlflow.log_params({
            "model": "LogisticRegression",
            "model_family": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "dataset_input": clin["name"],
            "feature_count": clin["feature_count"],
        })
        
        # Standardized metrics for MLflow Model Compare
        mlflow.log_metrics({
            "val_macro_f1": lr_metrics["macro_f1"],
            "val_balanced_accuracy": lr_metrics["balanced_accuracy"],
            "val_roc_auc": lr_metrics["roc_auc"],
            "val_accuracy": lr_metrics["accuracy"],
            **{f"val_{k}": v for k, v in lr_metrics.items()}
        })
        log_sklearn_model_safely(lr, artifact_path="clinical_logistic_model")

        print(
            f"[Logistic ] (Input: Clinical D={clin['feature_count']}) | "
            f"Val Macro F1: {lr_metrics['macro_f1']:6.4f} | "
            f"Bal Acc: {lr_metrics['balanced_accuracy']:6.4f} | "
            f"Acc: {lr_metrics['accuracy']:6.4f} | "
            f"AUC: {lr_metrics['roc_auc']:6.4f}"
        )
        results["logistic"] = lr_metrics
        results["logistic_run_id"] = lr_run.info.run_id

    return results


def run_random_forest_optimization(
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """
    Sub-suite 2: Random Forest Hyperparameter Optimization.
    Trained strictly on Dataset 2: Radiographic Visual HOG Embeddings (12 PCA components).
    Performs 12-trial grid search over estimators, depth, and leaf sizes.
    """
    print("\n" + "=" * 70)
    print("2. RANDOM FOREST HYPERPARAMETER OPTIMIZATION (UNIT 5)")
    print("   Dataset Input: Radiographic Visual HOG Embeddings Only (PCA=12)")
    print("=" * 70)

    vis = datasets["visual"]
    X_train = vis["X_train"]
    X_val = vis["X_val"]
    y_train = datasets["y_train"]
    y_val = datasets["y_val"]
    feature_names = vis["feature_names"]

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [5, 10],
        "class_weight": ["balanced"],
    }

    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_model: RandomForestClassifier | None = None
    best_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name="3_Visual_Random_Forest_Optimization") as parent_run:
        parent_id = parent_run.info.run_id
        log_dataset_safely(X_train, y_train, dataset_name=vis["name"])

        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("model_name", "VisualRandomForest")
        mlflow.set_tag("syllabus_unit", "Unit 5: Ensemble Learning & Tree-Based Methods")
        mlflow.set_tag("ensemble_paradigm", "Bagging (Bootstrap Aggregation)")
        mlflow.set_tag("dataset_name", vis["name"])
        mlflow.set_tag("dataset_modality", vis["modality"])
        mlflow.set_tag("feature_count", str(vis["feature_count"]))

        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())

        for idx, values in enumerate(param_combinations, 1):
            params: dict[str, Any] = dict(zip(param_keys, values, strict=False))
            trial_name = f"Trial_{idx}"

            with mlflow.start_run(run_name=trial_name, nested=True):
                rf = RandomForestClassifier(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=params["max_depth"],
                    min_samples_leaf=int(params["min_samples_leaf"]),
                    class_weight=cast(Any, params["class_weight"]),
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train, y_train)

                val_preds = rf.predict(X_val)
                val_probs = rf.predict_proba(X_val)[:, 1]
                metrics = compute_classification_metrics(y_val, val_preds, val_probs)

                logged_params = {k: ("None" if v is None else v) for k, v in params.items()}
                logged_params["model_family"] = "RandomForest"
                logged_params["dataset_input"] = vis["name"]
                logged_params["feature_count"] = vis["feature_count"]
                mlflow.log_params(logged_params)
                mlflow.log_param("trial_number", idx)
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

                depth_str = str(params["max_depth"]) if params["max_depth"] is not None else "None"
                print(
                    f"[{trial_name:8s}] trees={int(params['n_estimators']):3d} | depth={depth_str:4s} | "
                    f"leaf={int(params['min_samples_leaf']):2d} | "
                    f"Val F1: {metrics['macro_f1']:6.4f} | "
                    f"Bal Acc: {metrics['balanced_accuracy']:6.4f} | "
                    f"Acc: {metrics['accuracy']:6.4f} | "
                    f"AUC: {metrics['roc_auc']:6.4f}"
                )

                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    best_params = params
                    best_model = rf
                    best_metrics = {f"best_val_{k}": v for k, v in metrics.items()}

        if best_params:
            best_logged_params = {f"best_{k}": ("None" if v is None else v) for k, v in best_params.items()}
            best_logged_params["dataset_input"] = vis["name"]
            best_logged_params["feature_count"] = vis["feature_count"]
            mlflow.log_params(best_logged_params)
        
        # Standardized metrics for MLflow Model Compare
        standardized_metrics = {
            "val_macro_f1": best_score,
            "val_balanced_accuracy": best_metrics.get("best_val_balanced_accuracy", 0.0),
            "val_roc_auc": best_metrics.get("best_val_roc_auc", 0.0),
            "val_accuracy": best_metrics.get("best_val_accuracy", 0.0),
            **best_metrics
        }
        mlflow.log_metrics(standardized_metrics)
        if best_model is not None:
            log_sklearn_model_safely(best_model, artifact_path="visual_random_forest_model")

    print(
        f"\nVisual Random Forest complete. Best Params: {best_params} | "
        f"Best Val Macro F1: {best_score:.4f}\n"
    )

    return {
        "parent_run_id": parent_id,
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "feature_names": feature_names,
        "dataset_key": "visual",
    }


def run_xgboost_optimization(
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """
    Sub-suite 3: Gradient Boosting / XGBoost Hyperparameter Optimization.
    Trained strictly on Dataset 3: Multimodal Engineered Fusion (22 features: Visual PCA + Demographics + Interactions).
    Performs 8-trial grid search over estimators, depth, learning rate, and subsample ratio.
    """
    print("\n" + "=" * 70)
    print("3. GRADIENT BOOSTING / XGBOOST OPTIMIZATION (UNIT 5)")
    print("   Dataset Input: Multimodal Engineered Fusion (Visual PCA + Clinical + Interactions, D=22)")
    print("=" * 70)

    multi = datasets["multimodal"]
    X_train = multi["X_train"]
    X_val = multi["X_val"]
    y_train = datasets["y_train"]
    y_val = datasets["y_val"]
    feature_names = multi["feature_names"]

    # Class imbalance weight: count(neg) / count(pos)
    neg_count = float(np.sum(y_train == 0))
    pos_count = float(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1.0)
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.3f} (Neg: {int(neg_count)}, Pos: {int(pos_count)})")

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8],
    }

    best_score = -1.0
    best_params: dict[str, Any] = {}
    best_model: Any = None
    best_metrics: dict[str, float] = {}

    with mlflow.start_run(run_name="4_Multimodal_XGBoost_Optimization") as parent_run:
        parent_id = parent_run.info.run_id
        log_dataset_safely(X_train, y_train, dataset_name=multi["name"])

        mlflow.set_tag("model_family", "XGBoost" if HAS_XGBOOST else "HistGradientBoosting")
        mlflow.set_tag("model_name", "MultimodalXGBoost")
        mlflow.set_tag("syllabus_unit", "Unit 5: Ensemble Learning & Tree-Based Methods")
        mlflow.set_tag("ensemble_paradigm", "Boosting (Gradient Boosted Decision Trees)")
        mlflow.set_tag("dataset_name", multi["name"])
        mlflow.set_tag("dataset_modality", multi["modality"])
        mlflow.set_tag("feature_count", str(multi["feature_count"]))

        param_combinations = list(itertools.product(*param_grid.values()))
        param_keys = list(param_grid.keys())

        for idx, values in enumerate(param_combinations, 1):
            params: dict[str, Any] = dict(zip(param_keys, values, strict=False))
            trial_name = f"Trial_{idx}"

            with mlflow.start_run(run_name=trial_name, nested=True):
                if HAS_XGBOOST:
                    xgb = XGBClassifier(
                        n_estimators=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        subsample=float(params["subsample"]),
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    )
                else:
                    xgb = XGBClassifier(
                        max_iter=int(params["n_estimators"]),
                        max_depth=int(params["max_depth"]),
                        learning_rate=float(params["learning_rate"]),
                        class_weight="balanced",
                        random_state=42,
                    )

                xgb.fit(X_train, y_train)

                val_preds = xgb.predict(X_val)
                val_probs = xgb.predict_proba(X_val)[:, 1]
                metrics = compute_classification_metrics(y_val, val_preds, val_probs)

                xgb_params = dict(params)
                xgb_params["model_family"] = "XGBoost"
                xgb_params["dataset_input"] = multi["name"]
                xgb_params["feature_count"] = multi["feature_count"]
                mlflow.log_params(xgb_params)
                mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
                mlflow.log_param("trial_number", idx)
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})

                print(
                    f"[{trial_name:8s}] trees={int(params['n_estimators']):3d} | depth={int(params['max_depth']):1d} | "
                    f"lr={float(params['learning_rate']):4.2f} | sub={float(params['subsample']):3.1f} | "
                    f"Val F1: {metrics['macro_f1']:6.4f} | "
                    f"Bal Acc: {metrics['balanced_accuracy']:6.4f} | "
                    f"Acc: {metrics['accuracy']:6.4f} | "
                    f"AUC: {metrics['roc_auc']:6.4f}"
                )

                if metrics["macro_f1"] > best_score:
                    best_score = metrics["macro_f1"]
                    best_params = params
                    best_model = xgb
                    best_metrics = {f"best_val_{k}": v for k, v in metrics.items()}

        if best_params:
            best_logged_xgb = {f"best_{k}": v for k, v in best_params.items()}
            best_logged_xgb["dataset_input"] = multi["name"]
            best_logged_xgb["feature_count"] = multi["feature_count"]
            mlflow.log_params(best_logged_xgb)
        
        # Standardized metrics for MLflow Model Compare
        standardized_xgb_metrics = {
            "val_macro_f1": best_score,
            "val_balanced_accuracy": best_metrics.get("best_val_balanced_accuracy", 0.0),
            "val_roc_auc": best_metrics.get("best_val_roc_auc", 0.0),
            "val_accuracy": best_metrics.get("best_val_accuracy", 0.0),
            **best_metrics
        }
        mlflow.log_metrics(standardized_xgb_metrics)
        if best_model is not None:
            log_xgboost_model_safely(best_model, artifact_path="multimodal_xgboost_model")

    print(
        f"\nMultimodal XGBoost complete. Best Params: {best_params} | "
        f"Best Val Macro F1: {best_score:.4f}\n"
    )

    return {
        "parent_run_id": parent_id,
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "feature_names": feature_names,
        "dataset_key": "multimodal",
    }


def evaluate_test_set_isolation(
    rf_results: dict[str, Any],
    xgb_results: dict[str, Any],
    datasets: dict[str, Any],
) -> dict[str, Any]:
    """
    Evaluates the winning model on its corresponding X_test EXACTLY ONCE to preserve strict test set isolation.
    Logs test metrics, classification report CSV, and feature importance summary to the winning Parent Run.
    """
    print("\n" + "=" * 70)
    print("4. STRICT TEST SET ISOLATION EVALUATION (WINNING ENSEMBLE)")
    print("=" * 70)

    # Model selection based strictly on highest validation Macro F1
    if rf_results["best_score"] >= xgb_results["best_score"]:
        winning_name = "RandomForest"
        winning_model = rf_results["best_model"]
        winning_parent_id = rf_results["parent_run_id"]
        winning_val_f1 = rf_results["best_score"]
        winning_params = rf_results["best_params"]
        dataset_key = rf_results["dataset_key"]
    else:
        winning_name = "XGBoost"
        winning_model = xgb_results["best_model"]
        winning_parent_id = xgb_results["parent_run_id"]
        winning_val_f1 = xgb_results["best_score"]
        winning_params = xgb_results["best_params"]
        dataset_key = xgb_results["dataset_key"]

    active_dataset = datasets[dataset_key]
    X_test = active_dataset["X_test"]
    y_test = datasets["y_test"]
    feature_names = active_dataset["feature_names"]

    print(f">>> Selected Overall Champion Model: {winning_name} (Val Macro F1: {winning_val_f1:.4f})")
    print(f">>> Evaluated on Dataset: {active_dataset['name']} ({active_dataset['modality']})")
    print(f">>> Optimal Hyperparameters: {winning_params}")
    print(">>> Executing UNBIASED Single-Pass Evaluation on Unseen Test Set...")

    test_preds = winning_model.predict(X_test)
    test_probs = winning_model.predict_proba(X_test)[:, 1] if hasattr(winning_model, "predict_proba") else None
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)

    print(f"\n[TEST RESULTS - {winning_name}]")
    print(f"  Test Macro F1         : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Test Standard Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test ROC AUC          : {test_metrics['roc_auc']:.4f}")

    # Generate classification report
    report_dict = classification_report(
        y_test,
        test_preds,
        target_names=["No Finding", "Active Pathology"],
        output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv = "test_classification_report.csv"
    report_df.to_csv(report_csv, index=True)

    # Compute Feature Importances
    if hasattr(winning_model, "feature_importances_"):
        raw_importances = winning_model.feature_importances_
    else:
        raw_importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": raw_importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    importance_csv = "feature_importance.csv"
    importance_df.to_csv(importance_csv, index=False)

    top10 = importance_df.head(10)
    summary_lines = [
        f"CXR8 Ensemble Feature Importance Summary ({winning_name})",
        "=" * 50,
        f"Evaluated on Dataset: {active_dataset['name']}",
        f"Overall Test Macro F1: {test_metrics['macro_f1']:.4f}",
        f"Overall Test Balanced Acc: {test_metrics['balanced_accuracy']:.4f}",
        f"Overall Test ROC AUC: {test_metrics['roc_auc']:.4f}",
        "\nTop 10 Most Predictive Features:",
    ]
    for r_idx, row in top10.iterrows():
        summary_lines.append(f"  {r_idx + 1:2d}. {row['feature']:25s}: {row['importance']:.5f}")
    summary_text = "\n".join(summary_lines)
    summary_txt_path = "feature_importance_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Reopen winning parent run to log test evaluation artifacts and champion tags
    with mlflow.start_run(run_id=winning_parent_id):
        mlflow.set_tag("is_overall_winner", "True")
        mlflow.set_tag("champion_model_name", winning_name)
        mlflow.set_tag("champion_dataset_input", active_dataset["name"])
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(report_csv, artifact_path="evaluation")
        mlflow.log_artifact(importance_csv, artifact_path="feature_importance")
        mlflow.log_artifact(summary_txt_path, artifact_path="feature_importance")

    print("\nTop 5 Most Important Features in Predicting Active Pathology:")
    for _, row in top10.head(5).iterrows():
        print(f"  * {row['feature']:25s}: {row['importance']:.4f}")

    return {
        "winning_name": winning_name,
        "winning_val_f1": winning_val_f1,
        "winning_params": winning_params,
        "test_metrics": test_metrics,
        "top_features": top10.to_dict(orient="records"),
    }


def run_ensemble_optimization() -> dict[str, Any]:
    """Master orchestrator for Unit 5: Ensemble Learning & Tree-Based Methods."""
    configure_mlflow()

    data = helper.load_cxr8_data()
    datasets = prepare_distinct_model_datasets(data)

    # 1. Baseline Benchmark Suite (Dummy Majority & Clinical Metadata Logistic Regression)
    baseline_res = run_baseline_suite(datasets)

    # 2. Random Forest Hyperparameter Optimization (12 Trials on Visual HOG Embeddings)
    rf_res = run_random_forest_optimization(datasets)

    # 3. XGBoost Hyperparameter Optimization (8 Trials on Multimodal Engineered Fusion)
    xgb_res = run_xgboost_optimization(datasets)

    # 4. Strict Test Isolation Single-Pass Evaluation
    test_res = evaluate_test_set_isolation(rf_res, xgb_res, datasets)

    return {
        "baseline": baseline_res,
        "random_forest": rf_res,
        "xgboost": xgb_res,
        "test_evaluation": test_res,
    }


if __name__ == "__main__":
    run_ensemble_optimization()
