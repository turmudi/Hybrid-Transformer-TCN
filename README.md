# Hybrid Feature-Attention + TCN for Multiclass Intrusion Detection

A compact PyTorch implementation of **HybridFeatureTCN** for temporal multiclass intrusion detection.

## Repository contents

- `HybridFeatureTCN_UNSW_NB15.ipynb` — recommended notebook for Colab/Jupyter.
- `hybrid_feature_tcn.py` — standalone Python version.
- `requirements.txt` — minimal Python dependencies.

## Dataset

This repository does **not** redistribute UNSW-NB15. Place:

```text
UNSW_NB15_training-set.csv
```

in the repository root, or change `CFG["data_path"]`.

## Run

```bash
pip install -r requirements.txt
python hybrid_feature_tcn.py
```

For Google Colab, upload the CSV to the session and run the notebook from top to bottom.

## Experiment design

The compact pipeline preserves the core methodology of the supplied research notebook:

- chronological train/validation/test partitioning;
- sliding temporal windows (`W=10`);
- full-vocabulary categorical embeddings with `UNK=0`;
- `RobustScaler` fitted on training data only;
- Feature-Attention + causal dilated TCN;
- adaptive gated fusion;
- Focal Loss;
- square-root inverse-frequency `WeightedRandomSampler`;
- AdamW, gradient clipping, and early stopping;
- held-out chronological test evaluation.

The larger research notebook included additional cross-validation and diagnostic analyses. They are intentionally omitted here so this repository stays focused on the proposed model.

## Output metrics

The script reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- Per-class classification report
- Confusion matrix
