#!/usr/bin/env python3
"""
Cancer Survival Predictor from Multi-Omics Data
================================================

A clinically interpretable machine learning pipeline that predicts cancer
patient survival from gene expression data. Implements Cox Proportional Hazards,
Random Survival Forest, and DeepSurv (neural network) models with SHAP-based
interpretability to identify genes driving survival risk.

Clinical Relevance:
    Identifying molecular signatures that drive survival enables:
    - Risk stratification for treatment decisions
    - Discovery of potential therapeutic targets
    - Personalized medicine approaches based on individual gene expression profiles

Author: OpenCode Bioinformatics Pipeline
License: MIT
"""

import os
import sys
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def check_imports():
    """Verify all required packages are importable; print install guidance if not."""
    required = {
        "sklearn": "scikit-learn",
        "sksurv": "scikit-survival",
        "lifelines": "lifelines",
        "shap": "shap",
        "torch": "torch",
        "pycox": "pycox",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.warning(
            "Missing packages: %s. Install with: pip install %s",
            ", ".join(missing),
            " ".join(missing),
        )
    return missing


def progress(msg: str):
    """Print a timestamped progress message."""
    logger.info(msg)


# ===========================================================================
# 1. DATA ACQUISITION
# ===========================================================================

def download_tcga_brca():
    """
    Download TCGA-BRCA RNA-seq expression and clinical data via oncofind.
    Returns (expression_df, clinical_df) or (None, None) on failure.
    """
    try:
        from oncofind import search, download as onco_download
        progress("Attempting TCGA-BRCA download via oncofind ...")

        results = search("TCGA-BRCA")
        if results is None or len(results) == 0:
            raise RuntimeError("oncofind returned no results for TCGA-BRCA")

        download_path = Path("tcga_brca_data")
        download_path.mkdir(exist_ok=True)

        onco_download(results, dest=str(download_path))

        expr_file = None
        clin_file = None
        for f in download_path.rglob("*.csv"):
            name_lower = f.name.lower()
            if "expression" in name_lower or "rnaseq" in name_lower or "fpkm" in name_lower:
                expr_file = f
            elif "clinical" in name_lower or "survival" in name_lower:
                clin_file = f
            elif expr_file is None:
                expr_file = f

        if expr_file is not None and clin_file is not None:
            expr = pd.read_csv(expr_file)
            clin = pd.read_csv(clin_file)
            progress(f"Loaded TCGA expression: {expr.shape}, clinical: {clin.shape}")
            return expr, clin

        progress("oncofind download completed but files not in expected format; falling back.")
        return None, None

    except Exception as e:
        progress(f"oncofind download failed: {e}")
        return None, None


def load_metabric():
    """
    Load the METABRIC breast cancer dataset (used by pycox as built-in).
    Returns (expression_df, clinical_df) or (None, None) on failure.
    """
    try:
        from pycox.datasets import metabric
        progress("Loading METABRIC dataset from pycox ...")

        df = metabric.read_df()

        gene_cols = [c for c in df.columns if c not in ["duration", "event"]]
        if len(gene_cols) == 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            gene_cols = [c for c in numeric_cols if c not in ["duration", "event"]]

        clinical = pd.DataFrame({
            "sample_id": df.index.astype(str),
            "time": df["duration"].values,
            "event": df["event"].values.astype(int),
        })

        expression = df[gene_cols].copy()
        expression.insert(0, "sample_id", df.index.astype(str))

        progress(f"METABRIC loaded: expression {expression.shape}, clinical {clinical.shape}")
        return expression, clinical

    except Exception as e:
        progress(f"METABRIC load failed: {e}")
        return None, None


def load_gbsg2():
    """
    Load the GBSG-2 breast cancer dataset from scikit-survival.
    This dataset does not have gene expression but is useful as a minimal fallback.
    Returns (None, clinical_df) since expression is not available.
    """
    try:
        from sksurv.datasets import load_gbsg2
        progress("Loading GBSG-2 dataset from scikit-survival ...")

        X, y = load_gbsg2()
        clinical = pd.DataFrame({
            "sample_id": [f"GBSG2_{i}" for i in range(len(y))],
            "time": [val[1] for val in y],
            "event": [int(val[0]) for val in y],
        })
        cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_processed = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X_processed.insert(0, "sample_id", clinical["sample_id"].values)

        progress(f"GBSG-2 loaded: features {X_processed.shape}, clinical {clinical.shape}")
        return X_processed, clinical

    except Exception as e:
        progress(f"GBSG-2 load failed: {e}")
        return None, None


def generate_synthetic_data(n_patients=500, n_genes=2000):
    """
    Generate a synthetic gene-expression + survival dataset for demonstration.
    Simulates realistic count-like data with survival correlated to gene modules.
    """
    progress("Generating synthetic TCGA-like dataset for demonstration ...")
    rng = np.random.RandomState(RANDOM_SEED)

    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    sample_ids = [f"TCGA-BRCA-{i:04d}" for i in range(n_patients)]

    base_expr = rng.exponential(scale=5.0, size=(n_patients, n_genes))

    signal_genes = min(100, n_genes)
    beta_true = rng.normal(0, 0.5, size=signal_genes)
    linear_pred = base_expr[:, :signal_genes] @ beta_true
    scale_val = np.exp(-linear_pred / np.percentile(np.abs(linear_pred), 75))
    scale_val = np.clip(scale_val, 0.1, 5.0)

    times = rng.exponential(scale=scale_val * 365)
    times = np.clip(times, 30, 3650)
    censor_time = rng.uniform(180, 2000, size=n_patients)
    event = (times <= censor_time).astype(int)
    times = np.minimum(times, censor_time)

    expression = pd.DataFrame(base_expr, columns=gene_names)
    expression.insert(0, "sample_id", sample_ids)

    clinical = pd.DataFrame({"sample_id": sample_ids, "time": times, "event": event})

    progress(f"Synthetic data created: {n_patients} patients, {n_genes} genes")
    return expression, clinical


def load_dataset(dataset_name: str):
    """
    Master loader that tries the requested dataset and falls back gracefully.
    Returns (expression_df, clinical_df, dataset_label).
    """
    if dataset_name == "tcga":
        expr, clin = download_tcga_brca()
        if expr is not None and clin is not None:
            return expr, clin, "TCGA-BRCA"
        progress("Falling back to METABRIC ...")
        dataset_name = "metabric"

    if dataset_name == "metabric":
        expr, clin = load_metabric()
        if expr is not None and clin is not None:
            return expr, clin, "METABRIC"
        progress("Falling back to synthetic data ...")
        return generate_synthetic_data()

    if dataset_name == "gbsg2":
        expr, clin = load_gbsg2()
        if expr is not None and clin is not None:
            return expr, clin, "GBSG-2"

    progress("Falling back to synthetic data ...")
    return generate_synthetic_data()


# ===========================================================================
# 2. DATA PREPROCESSING
# ===========================================================================

def preprocess_data(expression: pd.DataFrame, clinical: pd.DataFrame,
                    max_features: int = 3000, small: bool = False):
    """
    Clean, transform and feature-select the expression + survival data.

    Steps:
        1. Align IDs between expression and clinical tables.
        2. Drop non-numeric / constant columns.
        3. Apply log2(CPM + 1)-like stabilisation if values look like counts.
        4. Keep top `max_features` most variable genes.
        5. Impute missing values with median.
    """
    progress("Preprocessing data ...")

    if small:
        max_features = min(max_features, 500)

    id_col = "sample_id"
    if id_col not in expression.columns:
        expression = expression.reset_index()
        if id_col not in expression.columns:
            expression.columns = [id_col] + list(expression.columns[1:])

    expr_ids = set(expression[id_col].astype(str))
    clin_ids = set(clinical[id_col].astype(str))
    common = sorted(expr_ids & clin_ids)
    progress(f"  Common samples: {len(common)}")

    if len(common) < 50:
        progress("  Very few overlapping samples — using all clinical rows and matching where possible.")
        common = sorted(clin_ids)

    expression = expression[expression[id_col].astype(str).isin(common)].copy()
    clinical = clinical[clinical[id_col].astype(str).isin(common)].copy()

    expression = expression.sort_values(id_col).reset_index(drop=True)
    clinical = clinical.sort_values(id_col).reset_index(drop=True)

    gene_cols = [c for c in expression.columns if c != id_col]
    gene_cols = [c for c in gene_cols if expression[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    constant_cols = [c for c in gene_cols if expression[c].nunique() <= 1]
    gene_cols = [c for c in gene_cols if c not in constant_cols]

    max_val = expression[gene_cols].max().max()
    if max_val > 50:
        progress("  Values appear to be raw counts — applying log2(CPM + 1) stabilisation ...")
        for c in gene_cols:
            expression[c] = np.log2(expression[c] + 1)

    variances = expression[gene_cols].var()
    variances = variances.sort_values(ascending=False)
    top_genes = variances.head(max_features).index.tolist()
    progress(f"  Selected top {len(top_genes)} most variable genes")

    expression = expression[[id_col] + top_genes]

    for c in top_genes:
        med = expression[c].median()
        expression[c] = expression[c].fillna(med)

    clinical["time"] = pd.to_numeric(clinical["time"], errors="coerce")
    clinical["event"] = pd.to_numeric(clinical["event"], errors="coerce")
    clinical = clinical.dropna(subset=["time", "event"])
    clinical["event"] = clinical["event"].astype(int)

    merged = expression.merge(clinical, on=id_col, how="inner")
    progress(f"  Final dataset: {merged.shape[0]} patients, {merged.shape[1]} columns")

    return merged, top_genes


# ===========================================================================
# 3. TRAIN / TEST SPLIT
# ===========================================================================

def split_data(merged: pd.DataFrame, test_size: float = 0.3, seed: int = RANDOM_SEED):
    """
    Stratified train/test split preserving the censoring ratio.
    """
    from sklearn.model_selection import train_test_split

    id_col = "sample_id"
    feature_cols = [c for c in merged.columns if c not in [id_col, "time", "event"]]

    X = merged[feature_cols].values.astype(np.float32)
    y_time = merged["time"].values.astype(np.float32)
    y_event = merged["event"].values.astype(int)

    stratify_by = y_event

    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=stratify_by
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = np.array(
        [(bool(y_event[i]), float(y_time[i])) for i in train_idx],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(y_event[i]), float(y_time[i])) for i in test_idx],
        dtype=[("event", bool), ("time", float)],
    )

    progress(f"  Train: {len(train_idx)} | Test: {len(test_idx)}")
    return X_train, X_test, y_train, y_test, feature_cols


# ===========================================================================
# 4. MODEL TRAINING
# ===========================================================================

def train_cox_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Cox Proportional Hazards with L1 (LASSO) regularisation via lifelines.
    Returns fitted model, predictions, and coefficient DataFrame.
    """
    progress("Training Cox PH (L1-penalised) ...")

    try:
        from lifelines import CoxPHFitter

        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df["time"] = y_train["time"]
        train_df["event"] = y_train["event"].astype(int)

        cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
        cph.fit(train_df, duration_col="time", event_col="event", show_progress=False)

        test_df = pd.DataFrame(X_test, columns=feature_cols)
        risk_scores_test = cph.predict_partial_hazard(test_df).values.ravel()
        risk_scores_train = cph.predict_partial_hazard(
            pd.DataFrame(X_train, columns=feature_cols)
        ).values.ravel()

        coefs = pd.DataFrame({
            "gene": feature_cols,
            "coefficient": cph.params_.values,
            "abs_coefficient": np.abs(cph.params_.values),
        }).sort_values("abs_coefficient", ascending=False)

        progress(f"  Cox model fitted — non-zero coefficients: {(coefs['abs_coefficient'] > 0.01).sum()}")
        return cph, risk_scores_train, risk_scores_test, coefs

    except Exception as e:
        progress(f"  Cox lifelines failed ({e}); trying scikit-survival ...")
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, fit_baseline_model=True)
        cox.fit(X_train, y_train)

        pred_train = cox.predict(X_train)
        pred_test = cox.predict(X_test)

        coefs = pd.DataFrame({
            "gene": feature_cols,
            "coefficient": cox.coef_.ravel(),
            "abs_coefficient": np.abs(cox.coef_.ravel()),
        }).sort_values("abs_coefficient", ascending=False)

        progress(f"  Cox (scikit-survival) fitted — non-zero coefficients: {(coefs['abs_coefficient'] > 0.01).sum()}")
        return cox, pred_train, pred_test, coefs


def train_rsf_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    Random Survival Forest from scikit-survival.
    Returns fitted model and predictions.
    """
    progress("Training Random Survival Forest ...")

    from sksurv.ensemble import RandomSurvivalForest

    rsf = RandomSurvivalForest(
        n_estimators=200,
        min_samples_split=10,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    rsf.fit(X_train, y_train)

    pred_train = rsf.predict(X_train)
    pred_test = rsf.predict(X_test)

    from sksurv.metrics import concordance_index_censored

    try:
        from sklearn.inspection import permutation_importance as sk_perm_importance

        def rsf_scorer(est, X, y):
            pred = est.predict(X)
            return concordance_index_censored(y["event"].astype(bool), y["time"].astype(float), pred)[0]

        perm_imp = sk_perm_importance(rsf, X_test, y_test, n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1, scoring=rsf_scorer)
        importances = perm_imp.importances_mean
    except Exception:
        importances = np.zeros(len(feature_cols))

    feat_imp = pd.DataFrame({
        "gene": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    progress(f"  RSF fitted — top feature: {feat_imp.iloc[0]['gene']} ({feat_imp.iloc[0]['importance']:.4f})")
    return rsf, pred_train, pred_test, feat_imp


def train_deepsurv_model(X_train, y_train, X_test, y_test, feature_cols):
    """
    DeepSurv neural network using PyTorch with manual Cox partial likelihood loss.
    Returns fitted model and predictions.
    """
    progress("Training DeepSurv (neural network) ...")

    try:
        import torch
        import torch.nn as nn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        progress(f"  Using device: {device}")

        num_features = X_train.shape[1]

        class DeepSurvNet(nn.Module):
            def __init__(self, in_features):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                )

            def forward(self, x):
                return self.net(x)

        def cox_partial_likelihood_loss(risk_scores, durations, events):
            """Compute negative Cox partial likelihood loss."""
            sorted_idx = torch.argsort(durations, descending=True)
            risk_scores = risk_scores[sorted_idx]
            events = events[sorted_idx]

            log_cumsum_hazard = torch.logcumsumexp(risk_scores, dim=0)
            uncensored_likelihood = risk_scores - log_cumsum_hazard
            loss = -torch.sum(uncensored_likelihood * events) / (events.sum() + 1e-8)
            return loss

        net = DeepSurvNet(num_features).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        durations_train = y_train["time"].astype(np.float32)
        events_train = y_train["event"].astype(np.float32)

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        durations_t = torch.tensor(durations_train, dtype=torch.float32, device=device)
        events_t = torch.tensor(events_train, dtype=torch.float32, device=device)

        batch_size = 128
        n_samples = len(X_train)
        epochs = 80

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = perm[start:end]

                batch_x = X_train_t[idx]
                batch_durations = durations_t[idx]
                batch_events = events_t[idx]

                risk_scores = net(batch_x).squeeze(-1)
                loss = cox_partial_likelihood_loss(risk_scores, batch_durations, batch_events)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                progress(f"    Epoch {epoch+1}/{epochs} — loss: {epoch_loss/max(n_batches,1):.4f}")

        net.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
            risk_test = net(X_test_t).squeeze(-1).cpu().numpy()

            X_train_t2 = torch.tensor(X_train, dtype=torch.float32, device=device)
            risk_train = net(X_train_t2).squeeze(-1).cpu().numpy()

        progress(f"  DeepSurv trained — output shape: {risk_test.shape}")
        return net, risk_train, risk_test, None

    except Exception as e:
        progress(f"  DeepSurv failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


# ===========================================================================
# 5. MODEL EVALUATION
# ===========================================================================

def compute_cindex(y_true, risk_scores):
    """Compute concordance index using scikit-survival."""
    from sksurv.metrics import concordance_index_censored

    if isinstance(risk_scores, pd.Series):
        risk_scores = risk_scores.values
    risk_scores = np.asarray(risk_scores).ravel()

    result = concordance_index_censored(
        y_true["event"].astype(bool),
        y_true["time"].astype(float),
        risk_scores,
    )
    return result[0]


def kaplan_meier_plot(ax, y_true, risk_scores, model_name, n_groups=2):
    """
    Plot Kaplan-Meier survival curves stratified by predicted risk.
    """
    from lifelines import KaplanMeierFitter

    median_score = np.median(risk_scores)
    low_mask = risk_scores <= median_score
    high_mask = risk_scores > median_score

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(
        y_true["time"][low_mask],
        y_true["event"][low_mask].astype(int),
        label="Low Risk",
    )
    kmf_low.plot_survival_function(ax=ax, ci_show=True, color="#2196F3", linewidth=2)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(
        y_true["time"][high_mask],
        y_true["event"][high_mask].astype(int),
        label="High Risk",
    )
    kmf_high.plot_survival_function(ax=ax, ci_show=True, color="#F44336", linewidth=2)

    try:
        from lifelines.statistics import logrank_test
        result = logrank_test(
            y_true["time"][low_mask],
            y_true["time"][high_mask],
            y_true["event"][low_mask].astype(int),
            y_true["event"][high_mask].astype(int),
        )
        p_val = result.p_value
        ax.set_title(f"{model_name}\n(p = {p_val:.4f})", fontsize=12, fontweight="bold")
    except Exception:
        ax.set_title(f"{model_name}", fontsize=12, fontweight="bold")

    ax.set_xlabel("Time (days)", fontsize=10)
    ax.set_ylabel("Survival Probability", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(0, 1.05)


def evaluate_models(results: dict, y_train, y_test):
    """
    Compute C-index for all models and return comparison DataFrame.
    """
    progress("Evaluating models ...")
    rows = []
    for name, res in results.items():
        if res["risk_test"] is None:
            continue
        try:
            c_train = compute_cindex(y_train, res["risk_train"])
        except Exception:
            c_train = np.nan
        try:
            c_test = compute_cindex(y_test, res["risk_test"])
        except Exception:
            c_test = np.nan

        rows.append({
            "Model": name,
            "C-index (train)": round(c_train, 4),
            "C-index (test)": round(c_test, 4),
            "Features": res.get("n_features", "?"),
        })

    comparison = pd.DataFrame(rows)
    progress("\n" + comparison.to_string(index=False))
    return comparison


# ===========================================================================
# 6. SHAP INTERPRETABILITY
# ===========================================================================

def compute_shap_values(model, X_test, feature_cols, model_name):
    """
    Compute SHAP values for the given model on the test set.
    Works best for tree-based models (RandomSurvivalForest).
    """
    progress(f"Computing SHAP values for {model_name} ...")

    try:
        import shap

        bg = shap.sample(X_test, min(100, X_test.shape[0]))

        def model_predict(x):
            return model.predict(x)

        explainer = shap.KernelExplainer(model_predict, bg)
        n_test = min(50, X_test.shape[0])
        shap_vals = explainer.shap_values(X_test[:n_test], nsamples=200)
        X_test = X_test[:n_test]

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_df = pd.DataFrame({
            "gene": feature_cols,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        progress(f"  Top SHAP gene: {shap_df.iloc[0]['gene']} ({shap_df.iloc[0]['mean_abs_shap']:.4f})")
        return shap_vals, shap_df, X_test

    except Exception as e:
        progress(f"  SHAP computation failed: {e}")
        return None, None, X_test


def plot_shap_summary(shap_vals, X, feature_cols, save_path):
    """Generate SHAP beeswarm summary plot for top 20 genes."""
    import shap

    top_n = min(20, X.shape[1])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    shap_top = shap_vals[:, top_idx]
    X_top = X[:, top_idx]
    names_top = [feature_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_top, X_top, feature_names=names_top, show=False, max_display=top_n)
    plt.title("SHAP Summary — Top 20 Genes Driving Survival Risk", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


def plot_shap_bar(shap_vals, feature_cols, save_path, top_n=20):
    """Bar plot of mean |SHAP| for top genes."""
    import shap

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]

    genes = [feature_cols[i] for i in top_idx]
    vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(genes)))
    ax.barh(range(len(genes)), vals[::-1], color=colors[::-1], edgecolor="grey", linewidth=0.5)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Gene Importance by SHAP (Top {top_n})", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


def plot_shap_waterfall(shap_vals, X, feature_cols, patient_idx, risk_label, save_path):
    """Waterfall plot for a single patient."""
    import shap

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-15:][::-1]

    shap_patient = shap_vals[patient_idx, top_idx]
    X_patient = X[patient_idx, top_idx]
    names_top = [feature_cols[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    sorted_idx = np.argsort(np.abs(shap_patient))[::-1][:15]
    colors = ["#F44336" if v > 0 else "#2196F3" for v in shap_patient[sorted_idx]]

    ax.barh(range(len(sorted_idx)), shap_patient[sorted_idx][::-1], color=colors[::-1], edgecolor="grey")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([names_top[i] for i in sorted_idx][::-1], fontsize=9)
    ax.set_xlabel("SHAP value (← protective | → risk)", fontsize=10)
    ax.set_title(f"SHAP Waterfall — {risk_label} Patient (idx {patient_idx})", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


# ===========================================================================
# 7. VISUALISATION
# ===========================================================================

def plot_km_comparison(results: dict, y_test, save_path):
    """Side-by-side Kaplan-Meier curves for all models."""
    model_names = [k for k, v in results.items() if v["risk_test"] is not None]
    n = len(model_names)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        kaplan_meier_plot(ax, y_test, results[name]["risk_test"], name)

    fig.suptitle("Kaplan-Meier Survival Curves by Predicted Risk Group", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


def plot_model_comparison(comparison_df: pd.DataFrame, save_path):
    """Bar chart of test C-index across models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(comparison_df)]
    bars = ax.bar(comparison_df["Model"], comparison_df["C-index (test)"], color=colors, edgecolor="grey")
    ax.set_ylabel("C-index (test)", fontsize=11)
    ax.set_title("Model Comparison — Concordance Index", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.legend()
    for bar, val in zip(bars, comparison_df["C-index (test)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


def plot_feature_heatmap(merged: pd.DataFrame, feature_cols: list, save_path, top_n=30):
    """Heatmap of top gene expressions across patients."""
    gene_var = merged[feature_cols].var().sort_values(ascending=False)
    top_genes = gene_var.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))
    subset = merged[top_genes].values
    subset_norm = (subset - subset.mean(axis=0)) / (subset.std(axis=0) + 1e-8)

    sns.heatmap(
        subset_norm.T, cmap="RdBu_r", center=0,
        xticklabels=False, yticklabels=top_genes,
        ax=ax, vmin=-3, vmax=3,
    )
    ax.set_title(f"Top {top_n} Most Variable Genes — Expression Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patients (sorted)", fontsize=10)
    ax.set_ylabel("Genes", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    progress(f"  Saved: {save_path}")


# ===========================================================================
# 8. MAIN PIPELINE
# ===========================================================================

def run_pipeline(args):
    """Execute the full survival prediction pipeline."""
    progress("=" * 70)
    progress("  CANCER SURVIVAL PREDICTOR — Multi-Omics ML Pipeline")
    progress("=" * 70)

    check_imports()

    progress(f"\n--- Step 1: Data Acquisition ({args.dataset}) ---")
    expression, clinical, dataset_label = load_dataset(args.dataset)
    progress(f"Dataset loaded: {dataset_label}")

    progress("\n--- Step 2: Preprocessing ---")
    merged, feature_cols = preprocess_data(expression, clinical, max_features=3000, small=args.small)
    progress(f"Final feature count: {len(feature_cols)}")

    progress("\n--- Step 3: Train/Test Split ---")
    X_train, X_test, y_train, y_test, feature_cols = split_data(merged)

    results = {}

    models_to_run = [m.strip().lower() for m in args.models.split(",")]

    if "cox" in models_to_run:
        progress("\n--- Step 4a: Cox Proportional Hazards ---")
        cox_model, cox_pred_train, cox_pred_test, cox_coefs = train_cox_model(
            X_train, y_train, X_test, y_test, feature_cols
        )
        results["Cox PH"] = {
            "model": cox_model,
            "risk_train": cox_pred_train,
            "risk_test": cox_pred_test,
            "coefs": cox_coefs,
            "n_features": len(feature_cols),
        }

    if "rsf" in models_to_run:
        progress("\n--- Step 4b: Random Survival Forest ---")
        rsf_model, rsf_pred_train, rsf_pred_test, rsf_imp = train_rsf_model(
            X_train, y_train, X_test, y_test, feature_cols
        )
        results["RSF"] = {
            "model": rsf_model,
            "risk_train": rsf_pred_train,
            "risk_test": rsf_pred_test,
            "importances": rsf_imp,
            "n_features": len(feature_cols),
        }

    if "deepsurv" in models_to_run:
        progress("\n--- Step 4c: DeepSurv ---")
        ds_model, ds_pred_train, ds_pred_test, ds_imp = train_deepsurv_model(
            X_train, y_train, X_test, y_test, feature_cols
        )
        if ds_model is not None:
            results["DeepSurv"] = {
                "model": ds_model,
                "risk_train": ds_pred_train,
                "risk_test": ds_pred_test,
                "importances": ds_imp,
                "n_features": len(feature_cols),
            }

    progress("\n--- Step 5: Model Evaluation ---")
    comparison = evaluate_models(results, y_train, y_test)

    progress("\n--- Step 6: SHAP Interpretability ---")
    best_name = comparison.loc[comparison["C-index (test)"].idxmax(), "Model"] if len(comparison) > 0 else None
    shap_data = {}
    if best_name and best_name in results:
        shap_vals, shap_df, X_shap = compute_shap_values(
            results[best_name]["model"], X_test, feature_cols, best_name
        )
        if shap_vals is not None:
            shap_data[best_name] = {"shap_vals": shap_vals, "shap_df": shap_df, "X": X_shap}

    for name in results:
        if name == best_name:
            continue
        if hasattr(results[name]["model"], "estimators_") or hasattr(results[name]["model"], "feature_importances_"):
            try:
                sv, sdf, Xs = compute_shap_values(results[name]["model"], X_test, feature_cols, name)
                if sv is not None:
                    shap_data[name] = {"shap_vals": sv, "shap_df": sdf, "X": Xs}
            except Exception:
                pass

    progress("\n--- Step 7: Visualisation & Output ---")
    plot_km_comparison(results, y_test, RESULTS_DIR / "kaplan_meier_comparison.png")
    plot_model_comparison(comparison, RESULTS_DIR / "model_comparison.png")
    plot_feature_heatmap(merged, feature_cols, RESULTS_DIR / "feature_importance_heatmap.png")

    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    progress(f"  Saved: {RESULTS_DIR / 'model_comparison.csv'}")

    if "Cox PH" in results and results["Cox PH"]["coefs"] is not None:
        results["Cox PH"]["coefs"].head(30).to_csv(RESULTS_DIR / "top_genes_cox.csv", index=False)
        progress(f"  Saved: {RESULTS_DIR / 'top_genes_cox.csv'}")

    if "RSF" in results and results["RSF"]["importances"] is not None:
        results["RSF"]["importances"].head(30).to_csv(RESULTS_DIR / "top_genes_rsf.csv", index=False)
        progress(f"  Saved: {RESULTS_DIR / 'top_genes_rsf.csv'}")

    for name, sd in shap_data.items():
        safe_name = name.replace(" ", "_").lower()
        plot_shap_summary(
            sd["shap_vals"], sd["X"], feature_cols,
            RESULTS_DIR / f"shap_summary_{safe_name}.png",
        )
        plot_shap_bar(
            sd["shap_vals"], feature_cols,
            RESULTS_DIR / f"shap_importance_{safe_name}.png",
        )
        sd["shap_df"].head(30).to_csv(RESULTS_DIR / f"top_genes_shap_{safe_name}.csv", index=False)
        progress(f"  Saved: {RESULTS_DIR / f'top_genes_shap_{safe_name}.csv'}")

        mean_shap = sd["shap_vals"].mean(axis=1)
        high_risk_idx = np.argmax(mean_shap)
        low_risk_idx = np.argmin(mean_shap)

        plot_shap_waterfall(
            sd["shap_vals"], sd["X"], feature_cols,
            high_risk_idx, "High-Risk",
            RESULTS_DIR / f"shap_waterfall_high_risk_{safe_name}.png",
        )
        plot_shap_waterfall(
            sd["shap_vals"], sd["X"], feature_cols,
            low_risk_idx, "Low-Risk",
            RESULTS_DIR / f"shap_waterfall_low_risk_{safe_name}.png",
        )

    progress("\n" + "=" * 70)
    progress("  PIPELINE COMPLETE — All results saved to results/ folder")
    progress("=" * 70)

    return results, comparison, shap_data


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cancer Survival Predictor — Multi-Omics ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python survival_predictor.py --dataset tcga
  python survival_predictor.py --dataset metabric --small
  python survival_predictor.py --dataset tcga --models cox,rsf,deepsurv
  python survival_predictor.py --dataset gbsg2 --models rsf
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="metabric",
        choices=["tcga", "metabric", "gbsg2"],
        help="Dataset to use (default: metabric). TCGA requires oncofind.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Run on a small subset for quick testing (fewer features, fewer patients).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cox,rsf,deepsurv",
        help="Comma-separated list of models: cox, rsf, deepsurv (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
