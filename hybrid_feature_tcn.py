"""Hybrid Feature-Attention + TCN for multiclass intrusion detection on UNSW-NB15.\nModel-focused script derived from the accompanying research notebook.\n"""\n\n\n# ==============================================================================\nimport os, math, random, copy, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "data_path": "UNSW_NB15_training-set.csv",
    "cat_features": ["proto", "service", "state"],
    "id_col": "id",
    "target_col": "attack_cat",
    "binary_col": "label",
    "test_size": 0.20,
    "val_size": 0.15,

    "window_size": 10,
    "d_token": 32,
    "n_attn_heads": 4,
    "n_attn_layers": 2,
    "tcn_in_dim": 64,
    "tcn_channels": [64, 64, 64],
    "tcn_kernel": 3,
    "fusion_dim": 128,
    "dropout": 0.15,

    "epochs": 40,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "clip_norm": 1.0,
    "patience": 10,
    "min_delta": 1e-4,
    "focal_gamma": 2.0,
}

print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")\n\n# ==============================================================================\nif not os.path.exists(CFG["data_path"]):
    raise FileNotFoundError(
        f"Dataset not found: {CFG['data_path']}\n"
        "Place UNSW_NB15_training-set.csv in the project root "
        "or change CFG['data_path']."
    )

df = pd.read_csv(CFG["data_path"])

# Use stime when available; otherwise id is retained as the chronological proxy
if "stime" in df.columns:
    df = df.sort_values("stime").reset_index(drop=True)
elif CFG["id_col"] in df.columns:
    df = df.sort_values(CFG["id_col"]).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

CAT_COLS = [c for c in CFG["cat_features"] if c in df.columns]
DROP_COLS = [c for c in [CFG["id_col"], CFG["target_col"], CFG["binary_col"]] if c in df.columns]
NUM_COLS = [c for c in df.columns if c not in CAT_COLS + DROP_COLS]

# Stable class mapping fitted once on the complete target vocabulary.
target_cat = df[CFG["target_col"]].astype("category")
CLASS_NAMES = list(target_cat.cat.categories)
class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
y = df[CFG["target_col"]].map(class_to_idx).to_numpy(dtype=np.int64)
N_CLASSES = len(CLASS_NAMES)

n = len(df)
n_test = int(n * CFG["test_size"])
n_pool = n - n_test
n_val = int(n_pool * CFG["val_size"])

train_df = df.iloc[: n_pool - n_val].reset_index(drop=True)
val_df   = df.iloc[n_pool - n_val : n_pool].reset_index(drop=True)
test_df  = df.iloc[n_pool:].reset_index(drop=True)

y_train = y[: n_pool - n_val]
y_val   = y[n_pool - n_val : n_pool]
y_test  = y[n_pool:]

print(f"Rows: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}")
print(f"Numeric features: {len(NUM_COLS)}")
print(f"Categorical features: {CAT_COLS}")
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")\n\n# ==============================================================================\nclass VocabEncoder:
    """Categorical encoder with reserved UNK=0 for unseen categories."""
    def __init__(self):
        self.class_to_idx = {}

    def fit(self, values):
        values = pd.Series(values).astype(str)
        self.class_to_idx = {v: i + 1 for i, v in enumerate(sorted(values.unique()))}
        return self

    def transform(self, values):
        values = pd.Series(values).astype(str)
        return values.map(self.class_to_idx).fillna(0).astype(np.int64).to_numpy()

    @property
    def num_embeddings(self):
        return len(self.class_to_idx) + 1


# Fit preprocessing only on the training partition
scaler = RobustScaler().fit(train_df[NUM_COLS].to_numpy())
cat_encoders = {c: VocabEncoder().fit(train_df[c]) for c in CAT_COLS}
VOCAB_SIZES = {c: cat_encoders[c].num_embeddings for c in CAT_COLS}

def encode_partition(frame):
    x_num = scaler.transform(frame[NUM_COLS].to_numpy()).astype(np.float32)
    if CAT_COLS:
        x_cat = np.stack([cat_encoders[c].transform(frame[c]) for c in CAT_COLS], axis=1)
    else:
        x_cat = np.empty((len(frame), 0), dtype=np.int64)
    return x_num, x_cat

X_train_num, X_train_cat = encode_partition(train_df)
X_val_num, X_val_cat = encode_partition(val_df)
X_test_num, X_test_cat = encode_partition(test_df)

assert np.isfinite(X_train_num).all()
assert np.isfinite(X_val_num).all()
assert np.isfinite(X_test_num).all()

print("Vocabulary sizes:", VOCAB_SIZES)\n\n# ==============================================================================\nclass SlidingWindowFlowDataset(Dataset):
    def __init__(self, x_num, x_cat, y, window_size):
        if len(y) < window_size:
            raise ValueError("Partition is shorter than window_size.")
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.window_size = window_size
        self.last_indices = np.arange(window_size - 1, len(y))

    def __len__(self):
        return len(self.last_indices)

    def __getitem__(self, idx):
        last = self.last_indices[idx]
        first = last - self.window_size + 1
        return (
            self.x_num[first:last + 1],
            self.x_cat[first:last + 1],
            self.y[last],
        )

    def labels(self):
        return self.y[self.last_indices].numpy()


train_ds = SlidingWindowFlowDataset(
    X_train_num, X_train_cat, y_train, CFG["window_size"]
)
val_ds = SlidingWindowFlowDataset(
    X_val_num, X_val_cat, y_val, CFG["window_size"]
)
test_ds = SlidingWindowFlowDataset(
    X_test_num, X_test_cat, y_test, CFG["window_size"]
)

# sqrt inverse-frequency sampling, as in the original research pipeline
train_labels = train_ds.labels()
counts = np.bincount(train_labels, minlength=N_CLASSES).astype(np.float64)
counts[counts == 0] = 1.0
class_weights = 1.0 / np.sqrt(counts)
sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
    torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False)

print(f"Windows: train={len(train_ds):,}, val={len(val_ds):,}, test={len(test_ds):,}")\n\n# ==============================================================================\nclass SharedCategoricalEmbedder(nn.Module):
    def __init__(self, vocab_sizes, d_token):
        super().__init__()
        self.cols = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(vocab_sizes[col], d_token) for col in self.cols
        })
        self.flat_dim = len(self.cols) * d_token

    def tokens(self, x_cat):
        if not self.cols:
            shape = (*x_cat.shape[:-1], 0, 1)
            return x_cat.new_empty(shape, dtype=torch.float32)
        return torch.stack(
            [self.embeddings[c](x_cat[..., i]) for i, c in enumerate(self.cols)],
            dim=-2,
        )

    def flat(self, x_cat):
        if not self.cols:
            return x_cat.new_empty((*x_cat.shape[:-1], 0), dtype=torch.float32)
        return torch.cat(
            [self.embeddings[c](x_cat[..., i]) for i, c in enumerate(self.cols)],
            dim=-1,
        )


class NumericTokenizer(nn.Module):
    def __init__(self, n_num, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num, d_token))
        self.bias = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1.0, 1.0)

    def forward(self, x_num):
        return x_num.unsqueeze(-1) * self.weight + self.bias


class FeatureAttentionEncoder(nn.Module):
    """FT-Transformer-style attention across features for each flow."""
    def __init__(self, d_token, n_heads, n_layers, dropout, ff_mult=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, num_tokens, cat_tokens):
        b, w = num_tokens.shape[:2]
        cls = self.cls_token.expand(b, w, 1, -1)
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=2)
        t, d = tokens.shape[2], tokens.shape[3]
        x = self.encoder(tokens.reshape(b * w, t, d))
        x = x.reshape(b, w, t, d)
        return x[:, :, 0, :]


class FlowEmbedding(nn.Module):
    def __init__(self, n_num, cat_embedder, d_model):
        super().__init__()
        self.numeric_proj = nn.Linear(n_num, d_model)
        self.cat_embedder = cat_embedder
        self.merge = nn.Sequential(
            nn.Linear(d_model + cat_embedder.flat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x_num, x_cat):
        num_h = self.numeric_proj(x_num)
        cat_h = self.cat_embedder.flat(x_cat)
        return self.merge(torch.cat([num_h, cat_h], dim=-1))


class Chomp1d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size] if self.size > 0 else x


class CausalTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel, dilation=dilation, padding=pad)),
            Chomp1d(pad), nn.GELU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.GELU()

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return self.act(self.net(x) + residual)


class TCNBranch(nn.Module):
    def __init__(self, in_dim, channels, kernel, dropout):
        super().__init__()
        layers = []
        current = in_dim
        self.dilations = [2 ** i for i in range(len(channels))]
        for out_ch, dilation in zip(channels, self.dilations):
            layers.append(CausalTemporalBlock(current, out_ch, kernel, dilation, dropout))
            current = out_ch
        self.net = nn.Sequential(*layers)
        self.out_dim = current
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(self.dilations)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, dim, hidden, dropout):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h_feature, h_tcn):
        interaction = h_feature * h_tcn
        gate = self.gate_net(
            torch.cat([h_feature, h_tcn, interaction], dim=-1)
        )
        fused = gate * h_feature + (1.0 - gate) * h_tcn
        return self.norm(fused), gate\n\n# ==============================================================================\nclass HybridFeatureTCN(nn.Module):
    def __init__(self, n_num, cat_vocab_sizes, n_classes, cfg):
        super().__init__()

        self.cat_embedder = SharedCategoricalEmbedder(
            cat_vocab_sizes, cfg["d_token"]
        )

        # Branch A: feature attention
        self.numeric_tokenizer = NumericTokenizer(n_num, cfg["d_token"])
        self.feature_attention = FeatureAttentionEncoder(
            cfg["d_token"],
            cfg["n_attn_heads"],
            cfg["n_attn_layers"],
            cfg["dropout"],
        )
        self.feature_projection = nn.Linear(
            cfg["d_token"], cfg["fusion_dim"]
        )

        # Branch B: temporal TCN
        self.flow_embedding = FlowEmbedding(
            n_num, self.cat_embedder, cfg["tcn_in_dim"]
        )
        self.tcn = TCNBranch(
            cfg["tcn_in_dim"],
            cfg["tcn_channels"],
            cfg["tcn_kernel"],
            cfg["dropout"],
        )
        self.tcn_projection = nn.Linear(
            self.tcn.out_dim, cfg["fusion_dim"]
        )

        # Fusion + classifier
        self.fusion = AdaptiveGatedFusion(
            cfg["fusion_dim"], cfg["fusion_dim"], cfg["dropout"]
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg["fusion_dim"], cfg["fusion_dim"] // 2),
            nn.LayerNorm(cfg["fusion_dim"] // 2),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["fusion_dim"] // 2, n_classes),
        )

    def forward(self, x_num, x_cat):
        # Feature-attention branch
        num_tokens = self.numeric_tokenizer(x_num)
        cat_tokens = self.cat_embedder.tokens(x_cat)
        feature_seq = self.feature_attention(num_tokens, cat_tokens)
        feature_vec = self.feature_projection(feature_seq.mean(dim=1))

        # Temporal branch
        flow_seq = self.flow_embedding(x_num, x_cat)
        temporal_seq = self.tcn(flow_seq)
        temporal_vec = self.tcn_projection(temporal_seq[:, -1, :])

        # Adaptive fusion
        fused, gate = self.fusion(feature_vec, temporal_vec)
        logits = self.classifier(fused)
        return logits, fused, gate


model = HybridFeatureTCN(
    len(NUM_COLS), VOCAB_SIZES, N_CLASSES, CFG
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")
print(f"TCN receptive field: {model.tcn.receptive_field}")
assert model.tcn.receptive_field >= CFG["window_size"]

# Forward-pass sanity check
x_num, x_cat, _ = next(iter(train_loader))
with torch.no_grad():
    logits, fused, gate = model(x_num[:8].to(DEVICE), x_cat[:8].to(DEVICE))
print("Logits:", logits.shape, "| Fused:", fused.shape, "| Gate:", gate.shape)\n\n# ==============================================================================\nclass FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        return (-(1.0 - pt).pow(self.gamma) * log_pt).mean()


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(training):
        for x_num, x_cat, y_batch in loader:
            x_num = x_num.to(DEVICE)
            x_cat = x_cat.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits, _, _ = model(x_num, x_cat)
            loss = criterion(logits, y_batch)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip_norm"])
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_true.append(y_batch.detach().cpu().numpy())
            all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


criterion = FocalLoss(CFG["focal_gamma"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG["lr"],
    weight_decay=CFG["weight_decay"],
)

history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_state = None
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, CFG["epochs"] + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer)
    val_metrics = run_epoch(model, val_loader, criterion)

    history["train_loss"].append(train_metrics["loss"])
    history["val_loss"].append(val_metrics["loss"])
    history["train_f1"].append(train_metrics["macro_f1"])
    history["val_f1"].append(val_metrics["macro_f1"])

    improved = val_metrics["loss"] < best_val_loss - CFG["min_delta"]
    if improved:
        best_val_loss = val_metrics["loss"]
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch {epoch:02d} | "
        f"train loss={train_metrics['loss']:.4f}, F1={train_metrics['macro_f1']:.4f} | "
        f"val loss={val_metrics['loss']:.4f}, F1={val_metrics['macro_f1']:.4f}"
        + (" *" if improved else "")
    )

    if patience_counter >= CFG["patience"]:
        print("Early stopping.")
        break

if best_state is not None:
    model.load_state_dict(best_state)\n\n# ==============================================================================\n@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_true, all_pred, all_prob, all_gate = [], [], [], []

    for x_num, x_cat, y_batch in loader:
        logits, _, gate = model(x_num.to(DEVICE), x_cat.to(DEVICE))
        prob = torch.softmax(logits, dim=1)

        all_true.append(y_batch.numpy())
        all_pred.append(prob.argmax(dim=1).cpu().numpy())
        all_prob.append(prob.cpu().numpy())
        all_gate.append(gate.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
        np.concatenate(all_gate),
    )


y_true, y_pred, y_prob, gate_values = predict(model, test_loader)

metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
    "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    "Weighted F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
}

for name, value in metrics.items():
    print(f"{name:16s}: {value:.4f}")

print("\nClassification report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=range(N_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)\n\n# ==============================================================================\nepochs = np.arange(1, len(history["train_loss"]) + 1)

plt.figure(figsize=(7, 4))
plt.plot(epochs, history["train_loss"], label="Train")
plt.plot(epochs, history["val_loss"], label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Focal Loss")
plt.title("Training History")
plt.legend()
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_true, y_pred, labels=range(N_CLASSES))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Held-Out Chronological Test Set")
plt.tight_layout()
plt.show()\n\n# ==============================================================================\ncheckpoint = {
    "model_state_dict": model.state_dict(),
    "config": CFG,
    "numeric_features": NUM_COLS,
    "categorical_features": CAT_COLS,
    "class_names": CLASS_NAMES,
    "vocabularies": {
        c: cat_encoders[c].class_to_idx for c in CAT_COLS
    },
    "scaler_center": getattr(scaler, "center_", None),
    "scaler_scale": getattr(scaler, "scale_", None),
}

torch.save(checkpoint, "hybrid_feature_tcn.pt")
print("Saved: hybrid_feature_tcn.pt")