"""
03_train_models.py
==================
Train and evaluate three ensemble models for soccer match prediction:
  A. Random Forest (Bagging)
  B. XGBoost (Boosting)
  C. Voting Ensemble (RF + XGBoost + Logistic Regression)

Outputs:
  - figures/confusion_matrix_*.png
  - figures/feature_importance_*.png
  - figures/model_comparison.png
  - figures/roc_curves.png
  - results/model_results.csv
  - results/classification_reports.txt

Usage: python3 03_train_models.py
"""

import pandas as pd
import numpy as np
import time
import os
import warnings

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_DIR = "data"
FIG_DIR = "figures"
RESULTS_DIR = "results"
RANDOM_STATE = 42
CV_FOLDS = 5


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """Load preprocessed train/test splits."""
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    X_train = train.drop(columns=["result"])
    y_train = train["result"]
    X_test = test.drop(columns=["result"])
    y_test = test["result"]

    print(f"  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]:,} rows")
    print(f"  Classes: {sorted(y_train.unique())}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODEL A: RANDOM FOREST
# =============================================================================

def train_random_forest(X_train, y_train):
    """Train Random Forest with GridSearchCV hyperparameter tuning."""
    print("\n  Tuning hyperparameters with GridSearchCV...")

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=0, refit=True
    )

    start = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")
    print(f"  Train time:  {train_time:.1f}s")

    return grid.best_estimator_, train_time


# =============================================================================
# MODEL B: XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train):
    """Train XGBoost with GridSearchCV hyperparameter tuning."""
    print("\n  Tuning hyperparameters with GridSearchCV...")

    # Encode labels for XGBoost (needs integers)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
        use_label_encoder=False,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        xgb, param_grid, cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=0, refit=True
    )

    start = time.time()
    grid.fit(X_train, y_encoded)
    train_time = time.time() - start

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")
    print(f"  Train time:  {train_time:.1f}s")

    return grid.best_estimator_, train_time, le


# =============================================================================
# MODEL C: VOTING ENSEMBLE
# =============================================================================

def train_voting_ensemble(X_train, y_train, rf_model, xgb_model):
    """
    Voting Ensemble: RF + XGBoost + Logistic Regression.
    Soft voting — averages predicted probabilities.
    """
    print("\n  Building soft voting classifier...")

    lr = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
    )

    from sklearn.base import clone
    rf_clone = clone(rf_model)
    xgb_clone = clone(xgb_model)

    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_clone),
            ("xgb", xgb_clone),
            ("lr", lr),
        ],
        voting="soft",
        n_jobs=-1,
    )

    start = time.time()
    ensemble.fit(X_train, y_train)
    train_time = time.time() - start

    # Cross-validation score
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)
    print(f"  CV F1 scores: {cv_scores.round(4)}")
    print(f"  Mean CV F1:   {cv_scores.mean():.4f}")
    print(f"  Train time:   {train_time:.1f}s")

    return ensemble, train_time


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name, le=None):
    """Evaluate a model and return metrics dict."""
    if le is not None:
        y_test_encoded = le.transform(y_test)
        y_pred_encoded = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
        y_proba = model.predict_proba(X_test)
        classes = le.classes_
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        classes = model.classes_

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    y_test_bin = label_binarize(y_test, classes=sorted(y_test.unique()))
    if le is not None:
        sorted_classes = sorted(y_test.unique())
        class_order = [list(le.classes_).index(c) for c in sorted_classes]
        y_proba_aligned = y_proba[:, class_order]
    else:
        sorted_classes = sorted(y_test.unique())
        class_order = [list(classes).index(c) for c in sorted_classes]
        y_proba_aligned = y_proba[:, class_order]

    try:
        auc = roc_auc_score(y_test_bin, y_proba_aligned, multi_class="ovo")
    except ValueError:
        auc = np.nan

    print(f"\n  {model_name} Results:")
    print(f"    Accuracy:     {acc:.4f}")
    print(f"    Weighted F1:  {f1:.4f}")
    print(f"    AUC-ROC:      {auc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    print(f"\n{report}")

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_weighted": f1,
        "auc_roc": auc,
        "y_pred": y_pred,
        "y_proba": y_proba_aligned,
        "report": report,
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred, labels=["H", "D", "A"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Home Win", "Draw", "Away Win"],
        yticklabels=["Home Win", "Draw", "Away Win"],
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()

    fname = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
    plt.close(fig)
    print(f"    Saved {fname}")


def plot_roc_curves(results_list, y_test):
    """Plot ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc as sk_auc

    sorted_classes = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=sorted_classes)
    class_labels = {"A": "Away Win", "D": "Draw", "H": "Home Win"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, cls in enumerate(sorted_classes):
        ax = axes[i]
        for res in results_list:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], res["y_proba"][:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{res["model"]} (AUC={roc_auc:.3f})', linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {class_labels[cls]}")
        ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print("    Saved roc_curves.png")


def plot_model_comparison(results_df, times):
    """Bar chart comparing models on key metrics + training time."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    metrics = [
        ("accuracy", "Accuracy"),
        ("f1_weighted", "Weighted F1"),
        ("auc_roc", "AUC-ROC (OVO)"),
    ]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    models = results_df["model"].tolist()

    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        vals = results_df[col].tolist()
        bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(min(vals) - 0.05, max(vals) + 0.05)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    ax = axes[3]
    time_vals = [times[m] for m in models]
    bars = ax.bar(models, time_vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Seconds")
    ax.set_title("Training Time")
    for bar, v in zip(bars, time_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}s", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)
    print("    Saved model_comparison.png")


def plot_feature_importance(model, X_test, y_test, model_name, le=None):
    """Generate feature importance plot using permutation importance."""
    print(f"    Computing permutation importance for {model_name}...")

    try:
        sample_size = min(500, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]

        if le is not None:
            y_sample_use = le.transform(y_sample)
        else:
            y_sample_use = y_sample

        result = permutation_importance(
            model, X_sample, y_sample_use,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
            scoring="f1_weighted"
        )

        top_idx = np.argsort(result.importances_mean)[-15:]
        top_features = [X_sample.columns[i] for i in top_idx]
        top_importance = result.importances_mean[top_idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_importance, color="#2196F3")
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("Mean Permutation Importance (F1 drop)")
        ax.set_title(f"Top 15 Feature Importance — {model_name}")
        plt.tight_layout()

        fname = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=150)
        plt.close(fig)
        print(f"    Saved {fname}")

    except Exception as e:
        print(f"    Importance failed for {model_name}: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_data()

    # ---- Model A: Random Forest ----
    print("\n" + "=" * 60)
    print("MODEL A: RANDOM FOREST (BAGGING)")
    print("=" * 60)
    rf_model, rf_time = train_random_forest(X_train, y_train)

    # ---- Model B: XGBoost ----
    print("\n" + "=" * 60)
    print("MODEL B: XGBOOST (BOOSTING)")
    print("=" * 60)
    xgb_model, xgb_time, xgb_le = train_xgboost(X_train, y_train)

    # ---- Model C: Voting Ensemble ----
    print("\n" + "=" * 60)
    print("MODEL C: VOTING ENSEMBLE (STACKING)")
    print("=" * 60)
    voting_model, voting_time = train_voting_ensemble(
        X_train, y_train, rf_model, xgb_model
    )

    # ---- Evaluate all models ----
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost", le=xgb_le)
    voting_results = evaluate_model(voting_model, X_test, y_test, "Voting Ensemble")

    results_list = [rf_results, xgb_results, voting_results]
    times = {
        "Random Forest": rf_time,
        "XGBoost": xgb_time,
        "Voting Ensemble": voting_time,
    }

    results_df = pd.DataFrame([{
        "model": r["model"],
        "accuracy": r["accuracy"],
        "f1_weighted": r["f1_weighted"],
        "auc_roc": r["auc_roc"],
        "train_time_sec": times[r["model"]],
    } for r in results_list])

    results_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    with open(os.path.join(RESULTS_DIR, "classification_reports.txt"), "w") as f:
        for r in results_list:
            f.write(f"{'='*60}\n")
            f.write(f"{r['model']}\n")
            f.write(f"{'='*60}\n")
            f.write(r["report"])
            f.write("\n\n")
    print("\n  Saved classification_reports.txt")

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    for r in results_list:
        plot_confusion_matrix(y_test, r["y_pred"], r["model"])

    plot_roc_curves(results_list, y_test)
    plot_model_comparison(results_df, times)

    # Feature importance for tree-based models
    plot_feature_importance(rf_model, X_test, y_test, "Random Forest")
    plot_feature_importance(xgb_model, X_test, y_test, "XGBoost", le=xgb_le)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Results: {RESULTS_DIR}/")
    print(f"  Figures: {FIG_DIR}/")
    print(f"\n  Best model: {results_df.loc[results_df['f1_weighted'].idxmax(), 'model']}")
    print(f"  Best F1:    {results_df['f1_weighted'].max():.4f}")


if __name__ == "__main__":
    main()