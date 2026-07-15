# Cancer Survival Predictor — Multi-Omics ML Pipeline

A clinically interpretable machine learning pipeline that predicts cancer patient survival from gene expression data. Implements three complementary models with SHAP-based interpretability to identify genes driving survival risk.

## Clinical Relevance

Identifying molecular signatures that drive survival enables:
- **Risk stratification** for treatment decisions
- **Discovery of potential therapeutic targets**
- **Personalized medicine** approaches based on individual gene expression profiles

## Models Implemented

| Model | Type | Strengths |
|-------|------|-----------|
| **Cox PH (L1)** | Linear, interpretable | Coefficients directly interpretable as log-hazard ratios |
| **Random Survival Forest** | Tree-based, non-linear | Captures gene-gene interactions, robust to outliers |
| **DeepSurv** | Neural network | Captures complex non-linear patterns in high-dimensional data |

## Dataset Options

- **METABRIC** (default): Molecular Taxonomy of Breast Cancer (~1,900 patients, 9 features)
- **TCGA-BRCA**: Breast cancer RNA-seq from The Cancer Genome Atlas (~1,000 patients)
- **GBSG-2**: German Breast Cancer Study Group (clinical features only, ~686 patients)
- **Synthetic**: Auto-generated data for demonstration if downloads fail

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Default: METABRIC dataset with all three models
python survival_predictor.py

# Use TCGA-BRCA dataset
python survival_predictor.py --dataset tcga

# Quick test with reduced features
python survival_predictor.py --small

# Run specific models only
python survival_predictor.py --models cox,rsf

# Full pipeline with TCGA
python survival_predictor.py --dataset tcga --models cox,rsf,deepsurv
```

## Example Results (METABRIC Dataset)

| Model | C-index (train) | C-index (test) | Features |
|-------|-----------------|----------------|----------|
| Cox PH | 0.5914 | 0.5851 | 9 |
| **RSF** | **0.7352** | **0.6552** | 9 |
| DeepSurv | 0.7079 | 0.6342 | 9 |

### Top Predictive Features (SHAP Analysis)

| Rank | Feature | Mean SHAP Value | Interpretation |
|------|---------|-----------------|----------------|
| 1 | x8 (age) | 129.74 | Strongest predictor of survival |
| 2 | x1 | 36.61 | High expression increases risk |
| 3 | x3 | 28.53 | Moderate risk factor |
| 4 | x6 | 22.19 | Moderate risk factor |
| 5 | x2 | 21.13 | Moderate risk factor |

## Output Files

All results are saved to the `results/` folder:

### Figures
| File | Description |
|------|-------------|
| `kaplan_meier_comparison.png` | KM curves for high vs low risk groups per model |
| `model_comparison.png` | Bar chart of C-index values across models |
| `feature_importance_heatmap.png` | Expression heatmap of top variable genes |
| `shap_summary_rsf.png` | SHAP beeswarm plot for top genes |
| `shap_importance_rsf.png` | Bar plot of mean SHAP values |
| `shap_waterfall_high_risk_rsf.png` | Gene contributions for high-risk patient |
| `shap_waterfall_low_risk_rsf.png` | Gene contributions for low-risk patient |

### CSV Files
| File | Description |
|------|-------------|
| `model_comparison.csv` | C-index train/test for each model |
| `top_genes_cox.csv` | Cox model coefficients (log-hazard ratios) |
| `top_genes_rsf.csv` | Random Survival Forest feature importance |
| `top_genes_shap_rsf.csv` | Mean absolute SHAP values per gene |

## Interpreting SHAP Values

- **Positive SHAP value** → gene increases risk (shorter survival)
- **Negative SHAP value** → gene decreases risk (longer survival)
- **Magnitude** → strength of the gene's contribution to the prediction

## Project Structure

```
cancer_survival_predictor/
├── survival_predictor.py     # Main pipeline script (~1000 lines)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── results/                  # Output directory (auto-created)
    ├── kaplan_meier_comparison.png
    ├── model_comparison.png
    ├── model_comparison.csv
    ├── feature_importance_heatmap.png
    ├── shap_summary_rsf.png
    ├── shap_importance_rsf.png
    ├── shap_waterfall_high_risk_rsf.png
    ├── shap_waterfall_low_risk_rsf.png
    ├── top_genes_cox.csv
    ├── top_genes_rsf.csv
    └── top_genes_shap_rsf.csv
```

## Technical Details

- **Random seed**: 42 (for reproducibility)
- **Default feature selection**: Top 3,000 most variable genes
- **Cox PH**: L1-regularised (LASSO) with penalizer=0.1
- **Random Survival Forest**: 200 trees, min_samples_leaf=15
- **DeepSurv**: 3-layer network (128→64→32), batch norm, dropout 0.3/0.2
- **SHAP**: KernelExplainer with 50 test samples, 200 nsamples

## Dependencies

- Python 3.9+
- pandas, numpy, scikit-learn, scikit-survival
- lifelines (Cox PH)
- torch (DeepSurv)
- shap (interpretability)
- matplotlib, seaborn (visualisation)
- oncofind (optional, for TCGA download)

## Performance Notes

- **Memory**: Uses ~2-4 GB RAM for METABRIC dataset
- **Runtime**: ~6-10 minutes on standard laptop (including SHAP computation)
- **Parallelization**: RSF uses all CPU cores with `n_jobs=-1`

## References

- Cox, D.R. (1972). Regression models and life-tables. *JRSS*
- Ishwaran et al. (2008). Random survival forests. *Annals of Applied Statistics*
- Katzman et al. (2018). DeepSurv: personalized treatment recommender system. *BMC Medical Research Methodology*
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*
