# run_pandora_ttdfiot_s_random.py
# "As-is" conversion of the user's new code for the TTDFIOTIDS2025 dataset.
# This version randomly selects a single attack class to hide.
# t-SNE 'n_iter' argument has been corrected to 'max_iter'.

# --- 1. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer # Using QuantileTransformer for advanced scaling
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import argparse
import os
from itertools import cycle

def set_global_seed(seed=42):
    """
    Ensures fully deterministic and reproducible behavior
    across Python, NumPy, and PyTorch (CPU + GPU).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)  # Optional: make CPU ops deterministic

    print(f"[INFO] Global seed fixed at {seed} — deterministic mode enabled.")

# Set seed at the start
set_global_seed(42)

# -------------------------
# Optional deterministic DataLoader behavior
# -------------------------
def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)

warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

# --- 2. DATA LOADING & PREPARATION ---
def load_and_prepare_dataset(t_file_path, label_column, quantile_n=1000):
    try:
        df = pd.read_csv(t_file_path)
        print(f"Successfully loaded data from: {t_file_path}")
        
        # --- Data Cleaning and Scaling ---
        all_features = [col for col in df.columns if col != label_column]
        df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Remove infinite values if any
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Use QuantileTransformer to handle non-Gaussian distributions and outliers.
        scaler = QuantileTransformer(n_quantiles=quantile_n, output_distribution='normal', random_state=42)
        df[all_features] = scaler.fit_transform(df[all_features])

        print("\nData prepared and scaled using QuantileTransformer.")
        return df, all_features

    except (FileNotFoundError, KeyError) as e:
        print(f"---! ERROR !--- An error occurred: {e}. Please check file path and column name.")
        return pd.DataFrame(), []

# --- 3. FEATURE GROUPING ---
def split_features(all_features):
    if not all_features:
        print("No features to split.")
        return {}
        
    print(f"\nUsing all {len(all_features)} features for the model.")

    print("\nLogically splitting features into Temporal and Volumetric groups...")
    temporal_keywords = ['duration', 'rate', 'srate', 'drate', 'iat','idle','active']
    
    temporal_features = [f for f in all_features if any(keyword in f.lower() for keyword in temporal_keywords)]
    volumetric_features = [f for f in all_features if f not in temporal_features]
    
    # Fallback mechanism in case the logical split fails.
    if not temporal_features or not volumetric_features:
        print("Warning: Logical split resulted in an empty feature group. Falling back to a random split.")
        random.shuffle(all_features)
        split_point = len(all_features) // 2
        temporal_features = all_features[:split_point]
        volumetric_features = all_features[split_point:]

    feature_groups = {'temporal': temporal_features, 'volumetric': volumetric_features}
    print(f"Temporal Modality Features ({len(temporal_features)}): {feature_groups['temporal']}")
    print(f"Volumetric Modality Features ({len(volumetric_features)}): {feature_groups['volumetric']}")
    return feature_groups

# --- 4. AI ARCHITECTURE - CORE & ATTENTION COMPONENTS ---

class FeatureAttention(nn.Module):
    """
    Learns a weight for each input feature to select the most important ones.
    """
    def __init__(self, num_features):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(num_features))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        attention_scores = self.softmax(self.attention_weights)
        x_attended = x * attention_scores
        return x_attended

class SimplifiedMamba(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=4, padding=3, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_conv = self.conv1d(x_t)[:, :, :x.shape[1]]
        return self.out_proj(x_conv.transpose(1, 2))

class Expert(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2 * d_model, d_model)
        )
    def forward(self, x): return self.net(x)

class MoE(nn.Module):
    def __init__(self, d_model, num_experts=4, dropout_rate=0.1):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model, dropout_rate) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)
        outputs = torch.stack([e(x) for e in self.experts], dim=-1)
        return torch.einsum('bse,bsde->bsd', scores, outputs)

class MambaMoEBlock(nn.Module):
    def __init__(self, d_model, num_experts=4, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = SimplifiedMamba(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.moe = MoE(d_model, num_experts, dropout_rate)

    def forward(self, x):
        x = x + self.mamba(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.cross_attn_A = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn_B = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d_model * 2, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model * 2))
        self.ln = nn.LayerNorm(d_model * 2)

    def forward(self, z_A, z_B):
        z_A_seq, z_B_seq = z_A.unsqueeze(1), z_B.unsqueeze(1)
        z_A_prime, _ = self.cross_attn_A(query=z_A_seq, key=z_B_seq, value=z_B_seq)
        z_B_prime, _ = self.cross_attn_B(query=z_B_seq, key=z_A_seq, value=z_A_seq)
        fused = torch.cat([z_A_prime.squeeze(1), z_B_prime.squeeze(1)], dim=1)
        return self.ln(fused + self.ffn(fused))

print("\nCore AI and Feature Attention components defined.")

# --- 5. AI ARCHITECTURE - THE FULL PROBABILISTIC MODEL (with Triplet Loss & Attention) ---

class ProbabilisticEncoder(nn.Module):
    def __init__(self, num_features, d_model, num_blocks, num_experts, dropout_rate=0.1):
        super().__init__()
        self.feature_attention = FeatureAttention(num_features)
        self.embedding = nn.Linear(num_features, d_model)
        self.encoder_blocks = nn.Sequential(*[MambaMoEBlock(d_model, num_experts, dropout_rate) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

    def forward(self, x):
        x_attended = self.feature_attention(x)
        x_embedded = self.embedding(x_attended).unsqueeze(1)
        x_encoded = self.encoder_blocks(x_embedded).mean(dim=1)
        x_encoded = self.dropout(x_encoded)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        return mu, logvar

class SotaIDS_Probabilistic(nn.Module):
    def __init__(self, num_temporal_features, num_volumetric_features, d_model, num_blocks, num_experts, dropout_rate, n_heads, triplet_loss_weight):
        super().__init__()
        self.d_model = d_model
        self.temporal_encoder = ProbabilisticEncoder(num_temporal_features, d_model, num_blocks, num_experts, dropout_rate)
        self.volumetric_encoder = ProbabilisticEncoder(num_volumetric_features, d_model, num_blocks, num_experts, dropout_rate)
        self.fusion = CrossAttentionFusion(d_model, n_heads)
        self.final_fc_mu = nn.Linear(d_model * 2, d_model * 2)
        self.final_fc_logvar = nn.Linear(d_model * 2, d_model * 2)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        self.triplet_loss_weight = triplet_loss_weight

    def forward_encoder(self, x_temporal, x_volumetric):
        mu_temp, logvar_temp = self.temporal_encoder(x_temporal)
        mu_vol, logvar_vol = self.volumetric_encoder(x_volumetric)
        fused_mu_intermediate = self.fusion(mu_temp, mu_vol)
        mu_final = self.final_fc_mu(fused_mu_intermediate)
        logvar_final = logvar_temp + logvar_vol
        return mu_final, logvar_final

    def forward(self, support_temporal, support_volumetric, support_labels, query_temporal, query_volumetric, query_labels_for_loss):
        n_way = len(torch.unique(support_labels))
        k_shot = support_temporal.shape[0] // n_way
        
        support_mu, support_logvar = self.forward_encoder(support_temporal, support_volumetric)
        query_mu, query_logvar = self.forward_encoder(query_temporal, query_volumetric)
        
        support_mu_reshaped = support_mu.view(n_way, k_shot, -1)
        support_var_reshaped = torch.exp(support_logvar).view(n_way, k_shot, -1)
        
        proto_mu = support_mu_reshaped.mean(dim=1)
        proto_var = support_var_reshaped.mean(dim=1)
        
        distances = self.wasserstein_distance(query_mu, torch.exp(query_logvar), proto_mu, proto_var)
        log_p_y = (-distances).log_softmax(dim=1)
        loss_cls = -log_p_y.gather(1, query_labels_for_loss.view(-1, 1)).squeeze().mean()
        
        anchors = query_mu
        positives = proto_mu[query_labels_for_loss]
        dist_matrix = torch.cdist(anchors, proto_mu)
        mask = torch.zeros_like(dist_matrix)
        mask.scatter_(1, query_labels_for_loss.unsqueeze(1), float('inf'))
        dist_matrix_masked = dist_matrix + mask
        hard_negative_indices = torch.argmin(dist_matrix_masked, dim=1)
        negatives = proto_mu[hard_negative_indices]
        loss_triplet = self.triplet_loss(anchors, positives, negatives)
        
        total_loss = loss_cls + (self.triplet_loss_weight * loss_triplet)
        
        _, preds = torch.min(distances, 1)
        acc = (preds == query_labels_for_loss).float().mean()
        
        return total_loss, acc

    def wasserstein_distance(self, mu1, var1, mu2, var2):
        mu1, var1 = mu1.unsqueeze(1), var1.unsqueeze(1)
        mu2, var2 = mu2.unsqueeze(0), var2.unsqueeze(0)

        # Handle potential NaNs or Infs from exp(logvar)
        var1 = torch.nan_to_num(var1, nan=1.0, posinf=1.0, neginf=1e-6)
        var2 = torch.nan_to_num(var2, nan=1.0, posinf=1.0, neginf=1e-6)

        term1 = torch.sum((mu1 - mu2)**2, dim=2)
        term2 = torch.sum(var1 + var2 - 2 * (var1 * var2).sqrt(), dim=2)
        return term1 + term2

print("Full Probabilistic SOTA IDS model defined with combined Triplet Loss and Feature Attention.")

# --- 6. META-LEARNING - EPISODIC DATALOADER ---

class EpisodicDataset(Dataset):
    def __init__(self, df, feature_groups, label_col):
        self.df = df
        self.temporal_cols = feature_groups['temporal']
        self.volumetric_cols = feature_groups['volumetric']
        self.label_col = label_col
        self.df[self.label_col] = self.df[self.label_col].astype('category')
        self.label_map = dict(enumerate(self.df[self.label_col].cat.categories))
        self.label_codes = self.df[self.label_col].cat.codes
        self.labels = self.label_codes.unique()
        self.data_by_class = {label: self.label_codes[self.label_codes == label].index for label in self.labels}

    def __len__(self): return 5000 

    def __getitem__(self, index):
        N_WAY, K_SHOT, N_QUERY = 5, 5, 10
        valid_labels = [lbl for lbl, idx in self.data_by_class.items() if len(idx) >= K_SHOT]
        if len(valid_labels) < N_WAY:
             # Fallback if not enough classes with K_SHOT samples
             valid_labels = list(self.data_by_class.keys())
        
        selected_classes = random.sample(valid_labels, N_WAY) if len(valid_labels) >= N_WAY else np.random.choice(valid_labels, N_WAY, replace=True)

        s_idx, q_idx, s_labels, q_labels = [], [], [], []
        for i, cls in enumerate(selected_classes):
            c_idx = self.data_by_class[cls]
            num_samples_needed = K_SHOT + N_QUERY
            replace = len(c_idx) < num_samples_needed
            
            samples_idx = np.random.choice(c_idx, size=num_samples_needed, replace=replace)
            s_idx.extend(samples_idx[:K_SHOT])
            q_idx.extend(samples_idx[K_SHOT:])
            s_labels.extend([i] * K_SHOT)
            q_labels.extend([i] * N_QUERY)

        s_df, q_df = self.df.loc[s_idx], self.df.loc[q_idx]
        
        s_temporal = torch.tensor(s_df[self.temporal_cols].values, dtype=torch.float32)
        s_volumetric = torch.tensor(s_df[self.volumetric_cols].values, dtype=torch.float32)
        q_temporal = torch.tensor(q_df[self.temporal_cols].values, dtype=torch.float32)
        q_volumetric = torch.tensor(q_df[self.volumetric_cols].values, dtype=torch.float32)
        
        s_labels_t = torch.tensor(s_labels, dtype=torch.long)
        q_labels_t = torch.tensor(q_labels, dtype=torch.long)
        
        perm = torch.randperm(len(q_labels_t))
        q_temporal, q_volumetric, q_labels_t = q_temporal[perm], q_volumetric[perm], q_labels_t[perm]

        return s_temporal, s_volumetric, s_labels_t, q_temporal, q_volumetric, q_labels_t

class StandardDataset(Dataset):
    def __init__(self, df, feature_groups, label_col):
        self.df = df
        self.temporal_cols = feature_groups['temporal']
        self.volumetric_cols = feature_groups['volumetric']
        self.label_col = label_col
        self.df[self.label_col] = self.df[self.label_col].astype('category')
        self.labels = torch.tensor(self.df[self.label_col].cat.codes.values, dtype=torch.long)
        self.temporal_data = torch.tensor(self.df[self.temporal_cols].values.astype(np.float32))
        self.volumetric_data = torch.tensor(self.df[self.volumetric_cols].values.astype(np.float32))

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        return self.temporal_data[idx], self.volumetric_data[idx], self.labels[idx]

print("\nEpisodic and Standard Dataloaders defined.")

# --- 7. META-LEARNING - TRAINING & VALIDATION FUNCTIONS ---

def run_epoch(model, dataloader, optimizer, is_training, episodes_per_epoch, device):
    if is_training: model.train()
    else: model.eval()

    total_loss, total_acc = 0, 0
    if len(dataloader) == 0: return 0.0, 0.0
        
    epoch_iterator = tqdm(dataloader, desc="Training" if is_training else "Validation", total=episodes_per_epoch, leave=False)
    
    with torch.set_grad_enabled(is_training):
        for i, (s_temp, s_vol, s_labels, q_temp, q_vol, q_labels) in enumerate(epoch_iterator):
            if i >= episodes_per_epoch: break
                
            s_temp, s_vol, s_labels = s_temp.squeeze(0).to(device), s_vol.squeeze(0).to(device), s_labels.squeeze(0).to(device)
            q_temp, q_vol, q_labels = q_temp.squeeze(0).to(device), q_vol.squeeze(0).to(device), q_labels.squeeze(0).to(device)
            
            if is_training: optimizer.zero_grad()
            
            loss, acc = model(s_temp, s_vol, s_labels, q_temp, q_vol, q_labels)
            
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            epoch_iterator.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})
            
    avg_loss = total_loss / episodes_per_epoch if episodes_per_epoch > 0 else 0
    avg_acc = total_acc / episodes_per_epoch if episodes_per_epoch > 0 else 0
    return avg_loss, avg_acc

print("\nTraining and validation functions defined.")

# --- 8. EVALUATION & VISUALIZATION FUNCTIONS ---

def plot_auc_roc_curve(model, test_loader, support_df, feature_groups, label_map, device, label_column, title="", save_path="roc_auc_plot.png"):
    """
    Computes and plots the AUC ROC curve for each class in a multi-class setting.
    """
    if model is None: return
    model.eval()
    y_true_b, y_scores = [], []
    print(f"\n--- Generating AUC ROC Curve for {title} ---")

    # Prepare support set prototypes
    support_df[label_column] = support_df[label_column].astype('category')
    support_df['label_code'] = support_df[label_column].cat.codes
    support_labels = torch.tensor(support_df['label_code'].values, dtype=torch.long)
    unique_labels, support_labels_mapped = torch.unique(support_labels, return_inverse=True)
    n_way = len(unique_labels)
    s_temp = torch.tensor(support_df[feature_groups['temporal']].values, dtype=torch.float32).to(device)
    s_vol = torch.tensor(support_df[feature_groups['volumetric']].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        support_mu, support_logvar = model.forward_encoder(s_temp, s_vol)

    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_logvar = torch.zeros(n_way, support_logvar.shape[1], device=device)
    for i in range(n_way):
        indices = (support_labels_mapped == i)
        if torch.any(indices):
            proto_mu[i] = support_mu[indices].mean(dim=0)
            proto_logvar[i] = support_logvar[indices].mean(dim=0)
        else:
            print(f"Warning: No support samples for prototype index {i} in AUC ROC eval.")

    # Get predictions and true labels
    for batch_temp, batch_vol, batch_labels in tqdm(test_loader, desc=f"Generating Scores ({title})"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, torch.exp(proto_logvar))
            # Convert distances to scores (lower distance = higher score)
            scores = -distances
        y_scores.extend(scores.cpu().numpy())
        y_true_b.extend(batch_labels.cpu().numpy())

    # Binarize the labels for multi-class ROC
    classes = sorted(label_map.keys())
    y_true_binarized = label_binarize(y_true_b, classes=classes)
    y_scores = np.array(y_scores)
    
    # Ensure y_scores has the same number of classes as y_true_binarized
    if y_true_binarized.shape[1] != y_scores.shape[1]:
        print(f"Warning: Mismatch in binarized labels ({y_true_binarized.shape[1]}) and scores ({y_scores.shape[1]}). Adjusting scores array.")
        # This can happen if the test set doesn't contain all classes from the training set map.
        # We need to align scores to the full class list.
        # This is a complex alignment problem; for now, we'll plot for classes we have scores for.
        n_way = min(y_true_binarized.shape[1], y_scores.shape[1])
        y_true_binarized = y_true_binarized[:, :n_way]
        y_scores = y_scores[:, :n_way]


    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_way):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta'])
    for i, color in zip(range(n_way), colors):
        class_name = label_map.get(unique_labels[i].item(), f"Class {i}")
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_name} (area = {roc_auc[i]:.6f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multi-class Receiver Operating Characteristic for {title}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"AUC ROC plot saved to {save_path}")
    plt.show()


def evaluate_known_attacks(model, test_loader, support_df, feature_groups, label_map, device, label_column, title="", save_dir="."):
    if model is None: return
    model.eval()
    y_true, y_pred = [], []
    print(f"\n--- Evaluating model performance on the test set ({title}) ---")
    
    if support_df.empty:
        print(f"Warning: Support DataFrame for evaluation ({title}) is empty. Cannot create prototypes.")
        return

    support_df[label_column] = support_df[label_column].astype('category')
    support_df['label_code'] = support_df[label_column].cat.codes
    support_labels = torch.tensor(support_df['label_code'].values, dtype=torch.long)
    unique_labels, support_labels_mapped = torch.unique(support_labels, return_inverse=True)
    n_way = len(unique_labels)
    s_temp = torch.tensor(support_df[feature_groups['temporal']].values, dtype=torch.float32).to(device)
    s_vol = torch.tensor(support_df[feature_groups['volumetric']].values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        support_mu, support_logvar = model.forward_encoder(s_temp, s_vol)
    
    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_logvar = torch.zeros(n_way, support_logvar.shape[1], device=device)
    for i in range(n_way):
        indices = (support_labels_mapped == i)
        if torch.any(indices):
            proto_mu[i] = support_mu[indices].mean(dim=0)
            proto_logvar[i] = support_logvar[indices].mean(dim=0)
        else:
            print(f"Warning: No support samples for prototype index {i} in eval ({title}).")

    for batch_temp, batch_vol, batch_labels in tqdm(test_loader, desc=f"Evaluating ({title})"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, torch.exp(proto_logvar))
        
        _, predicted_class_indices = torch.min(distances, dim=1)
        y_true.extend([label_map.get(l.item(), "Unknown") for l in batch_labels])
        predicted_orig_codes = unique_labels[predicted_class_indices].cpu().numpy()
        y_pred.extend([label_map.get(code, "Unknown") for code in predicted_orig_codes])
        
    print(f"\n--- Classification Report ({title}) ---")
    all_labels_present = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, zero_division=0, labels=all_labels_present, target_names=all_labels_present, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # --- SAVE TO CSV CODE ADDED HERE ---
    if title:
        # Sanitize title for filename
        safe_title = title.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower()
        filename = f"report_{safe_title}.csv"
        save_path = os.path.join(save_dir, filename)
        report_df.to_csv(save_path)
        print(f"Saved classification report to {save_path}")
    # -----------------------------------

def evaluate_zero_shot_performance(model, unseen_df, support_df, feature_groups, label_map, device, label_column, is_adapted=False, unseen_class_label="", save_dir="."):
    """
    Evaluates the model on a batch of unseen data for zero-shot novelty detection or few-shot classification.
    """
    if model is None or unseen_df.empty:
        print("Model not available or unseen data is empty.")
        return
    
    model.eval()
    y_true, y_pred, novelty_detected, novelty_scores = [], [], [], []
    
    # Prepare support set prototypes (either with or without the unseen class)
    support_df[label_column] = support_df[label_column].astype('category')
    support_df['label_code'] = support_df[label_column].cat.codes
    support_labels = torch.tensor(support_df['label_code'].values, dtype=torch.long)
    unique_labels, support_labels_mapped = torch.unique(support_labels, return_inverse=True)
    n_way = len(unique_labels)
    
    s_temp = torch.tensor(support_df[feature_groups['temporal']].values, dtype=torch.float32).to(device)
    s_vol = torch.tensor(support_df[feature_groups['volumetric']].values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        support_mu, support_logvar = model.forward_encoder(s_temp, s_vol)
    
    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_logvar = torch.zeros(n_way, support_logvar.shape[1], device=device)
    for i in range(n_way):
        indices = (support_labels_mapped == i)
        if torch.any(indices):
            proto_mu[i] = support_mu[indices].mean(dim=0)
            proto_logvar[i] = support_logvar[indices].mean(dim=0)
        else:
            print(f"Warning: No support samples for prototype index {i} in {'Adapted' if is_adapted else 'Zero-Shot'} eval.")

    # Prepare unseen test data
    unseen_dataset = StandardDataset(unseen_df, feature_groups, label_column)
    unseen_loader = DataLoader(unseen_dataset, batch_size=256)
    
    # Evaluate
    for batch_temp, batch_vol, _ in tqdm(unseen_loader, desc=f"Evaluating {'Adapted' if is_adapted else 'Zero-Shot'} Model"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, torch.exp(proto_logvar))
            
        min_distance, predicted_class_indices = torch.min(distances, dim=1)
        
        y_true.extend([unseen_class_label] * len(batch_temp))
        predicted_orig_codes = unique_labels[predicted_class_indices].cpu().numpy()
        predicted_labels = [label_map.get(code, "Unknown") for code in predicted_orig_codes]
        y_pred.extend(predicted_labels)
        novelty_detected.extend([label != 'Benign' for label in predicted_labels if label is not None])
        novelty_scores.extend(min_distance.cpu().numpy())

    # Report Results
    if is_adapted:
        print(f"\n--- Few-Shot Classification Report (on Unseen Class: {unseen_class_label}) ---")
        all_labels_present = sorted(list(set(y_true + y_pred)))
        report = classification_report(y_true, y_pred, zero_division=0, labels=all_labels_present, target_names=all_labels_present)
        print(report)
    else:
        results_df = pd.DataFrame({
            'True Label': y_true,
            'Predicted Nearest Class': y_pred,
            'Novelty Score (Distance)': novelty_scores,
            'Detected as Novelty': novelty_detected
        })
        
        # --- SAVE TO CSV CODE ADDED HERE ---
        if unseen_class_label:
            filename = f"zero_shot_results_{unseen_class_label}.csv"
            save_path = os.path.join(save_dir, filename)
            results_df.to_csv(save_path, index=False)
            print(f"Saved zero-shot results to {save_path}")
        # -----------------------------------

        print(f"\n--- Pure Zero-Shot Novelty Detection Report (on Unseen Class: {unseen_class_label}) ---")
        print("Sample of Zero-Shot Results:")
        print(results_df.head())
        
        print("\nNovelty Score (Distance) Statistics for the Unseen Class:")
        print(results_df['Novelty Score (Distance)'].describe())
        
        detection_accuracy = results_df['Detected as Novelty'].mean()
        print(f"\nNovelty Detection Accuracy: {detection_accuracy:.2%}")
        print("(This is the % of unseen attacks whose nearest class was not 'Benign')")
        
def fine_tune_for_adaptation(model, adaptation_df, feature_groups, device, num_shots, label_column, hyperparams):
    """
    Briefly fine-tunes the model on a small set of new data.
    """
    print(f"\n--- Simulating Adaptation: Fine-tuning on {num_shots} shots of the new class ---")
    
    adapt_dataset = EpisodicDataset(adaptation_df, feature_groups, label_column)
    adapt_loader = DataLoader(adapt_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'] / 10) # Use a smaller LR for fine-tuning
    
    model.train()
    # Fine-tune for a small number of episodes
    run_epoch(model, adapt_loader, optimizer, is_training=True, episodes_per_epoch=100, device=device)
    
    print("Adaptation complete.")
    return model

# --- 9. MAIN EXECUTION ---

def main():
    parser = argparse.ArgumentParser(description="Reproducible SOTA IDS experiment on TTDFIOTIDS2025.")
    parser.add_argument("--dataset_path", type=str, default="/home/beamformer/NDSS_2025/TTDF_IoT_IDS_2025_Ready_Again.csv", help="Path to TTDFIOTIDS2025 CSV.")
    parser.add_argument("--label_column", type=str, default="Label", help="Label column name.")
    # --unseen_classes argument is removed, as it's now handled randomly.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save models and artifacts.")
    parser.add_argument("--quantile_n", type=int, default=1000, help="n_quantiles for QuantileTransformer")
    # hyperparams exposed as args
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--triplet_loss_weight", type=float, default=0.75)
    args = parser.parse_args()

    # This global is used by the eval functions.
    global label_column
    label_column = args.label_column
    
    os.makedirs(args.save_dir, exist_ok=True)

    # load dataset
    df, all_features = load_and_prepare_dataset(args.dataset_path, label_column, quantile_n=args.quantile_n)
    if df.empty:
        print("Empty dataframe loaded — exiting.")
        return

    # --- CLASS MAPPING REMOVED ---
    print("\nProceeding with all original classes from the dataset.")

    feature_groups = split_features(all_features)
    if not feature_groups:
        print("\nSkipping feature processing (feature split failed).")
        return

    final_df = df[all_features + [label_column]]
    print("\nCreated initial dataframe with all features.")

    # Data cleaning: remove classes with too few samples for episodic training
    print("\nCleaning data: Removing classes with fewer than 10 samples...")
    class_counts = final_df[label_column].value_counts()
    classes_to_keep = class_counts[class_counts >= 10].index
    original_rows = len(final_df)
    final_df = final_df[final_df[label_column].isin(classes_to_keep)]
    print(f"Removed {original_rows - len(final_df)} rows belonging to very rare classes.")

    print("\nFinal class distribution:")
    print(final_df[label_column].value_counts())

    # if no data after cleaning
    if final_df.empty:
        print("No data available after cleaning; exiting.")
        return

    # --- Data Splitting (Random Unseen Class Logic) ---
    all_class_labels = final_df[label_column].unique()
    attack_labels = [label for label in all_class_labels if "benign" not in str(label).lower()]
    if not attack_labels:
        # Fallback if 'benign' string isn't present
        attack_labels = [l for l in all_class_labels if l != 'Benign' and l != 'BENIGN']
    
    if not attack_labels:
        print("Error: No attack labels found to select for hiding. Exiting.")
        return
        
    UNSEEN_CLASS_LABEL = random.choice(attack_labels)
    print(f"\n==================== Running experiment hiding RANDOMLY selected class: '{UNSEEN_CLASS_LABEL}' ====================")

    seen_df = final_df[final_df[label_column] != UNSEEN_CLASS_LABEL]
    unseen_df = final_df[final_df[label_column] == UNSEEN_CLASS_LABEL]

    if seen_df.empty:
        print(f"Error: No data left after excluding unseen class. Exiting.")
        return

    if unseen_df.empty:
        print(f"Warning: The unseen class '{UNSEEN_CLASS_LABEL}' has no data. ZSL/Adaptation will be skipped.")

    try:
        train_val_df, test_df = train_test_split(seen_df, test_size=0.2, stratify=seen_df[label_column], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_column], random_state=42)
    except ValueError as e:
        print(f"\nStratification failed: {e}. Splitting without stratification.")
        train_val_df, test_df = train_test_split(seen_df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    train_dataset = EpisodicDataset(train_df, feature_groups, label_column)
    val_dataset = EpisodicDataset(val_df, feature_groups, label_column)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Training on device: {device} ---")

    hyperparams = {
        'd_model': args.d_model,
        'num_blocks': args.num_blocks,
        'num_experts': args.num_experts,
        'dropout_rate': args.dropout_rate,
        'n_heads': args.n_heads,
        'LR': args.lr,
        'triplet_loss_weight': args.triplet_loss_weight
    }
    print(f"Using fixed hyperparameters: {hyperparams}")

    # Model init
    model = SotaIDS_Probabilistic(
        num_temporal_features=len(feature_groups['temporal']),
        num_volumetric_features=len(feature_groups['volumetric']),
        d_model=hyperparams['d_model'],
        num_blocks=hyperparams['num_blocks'],
        num_experts=hyperparams['num_experts'],
        dropout_rate=hyperparams['dropout_rate'],
        n_heads=hyperparams['n_heads'],
        triplet_loss_weight=hyperparams['triplet_loss_weight']
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    EPOCHS = args.epochs
    EPISODES_PER_TRAIN_EPOCH = args.train_episodes
    EPISODES_PER_VAL_EPOCH = args.val_episodes
    PATIENCE = args.patience
    
    # Define save paths based on save_dir and the randomly chosen class
    model_save_path = os.path.join(args.save_dir, f"best_model_ttdfiot_hiding_{UNSEEN_CLASS_LABEL}.pth")
    history_save_path = os.path.join(args.save_dir, f"training_history_ttdfiot_hiding_{UNSEEN_CLASS_LABEL}.csv")
    plot_save_path = os.path.join(args.save_dir, f"training_plots_ttdfiot_hiding_{UNSEEN_CLASS_LABEL}.png")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, optimizer, is_training=True, episodes_per_epoch=EPISODES_PER_TRAIN_EPOCH, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, None, is_training=False, episodes_per_epoch=EPISODES_PER_VAL_EPOCH, device=device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Time: {epoch_duration:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation loss improved. Checkpoint saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\nMeta-training finished.")
    print("\nLoading best model checkpoint for final evaluation...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print(f"Warning: Saved checkpoint {model_save_path} not found. Using current model state for evaluation.")

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_save_path, index=False)
    print(f"Training history saved successfully to {history_save_path}.")

    # --- 9. TRAINING VISUALIZATION ---
    if history['train_loss']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Over Epochs'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.grid(True)

        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy Over Epochs'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy')
        ax2.legend(); ax2.grid(True)

        plt.savefig(plot_save_path)
        print(f"\nTraining plots saved to {plot_save_path}")
        plt.show()

    # --- 10. FEATURE IMPORTANCE ANALYSIS ---
    if model:
        print("\n--- Feature Importance Analysis ---")
        
        temporal_weights = model.temporal_encoder.feature_attention.softmax(model.temporal_encoder.feature_attention.attention_weights).cpu().detach().numpy()
        volumetric_weights = model.volumetric_encoder.feature_attention.softmax(model.volumetric_encoder.feature_attention.attention_weights).cpu().detach().numpy()

        temporal_importance = pd.DataFrame({'feature': feature_groups['temporal'], 'importance': temporal_weights}).sort_values('importance', ascending=False)
        volumetric_importance = pd.DataFrame({'feature': feature_groups['volumetric'], 'importance': volumetric_weights}).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Temporal Features:")
        print(temporal_importance.head(10))

        print("\nTop 10 Most Important Volumetric Features:")
        print(volumetric_importance.head(10))

        # Plotting
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=temporal_importance.head(15))
        plt.title('Top 15 Temporal Feature Importances')
        temp_feat_path = os.path.join(args.save_dir, f"feature_importance_temporal_hiding_{UNSEEN_CLASS_LABEL}.png")
        plt.savefig(temp_feat_path, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=volumetric_importance.head(15))
        plt.title('Top 15 Volumetric Feature Importances')
        vol_feat_path = os.path.join(args.save_dir, f"feature_importance_volumetric_hiding_{UNSEEN_CLASS_LABEL}.png")
        plt.savefig(vol_feat_path, bbox_inches='tight')
        plt.show()

    # --- 11. COMPREHENSIVE EVALUATION ---
    
    # --- Part 1: Evaluation on Seen Attacks (Test Set) ---
    support_set_for_eval = train_df.groupby(label_column).head(10)
    test_dataset_standard = StandardDataset(test_df, feature_groups, label_column)
    test_loader = DataLoader(test_dataset_standard, batch_size=args.eval_batch_size)
    evaluate_known_attacks(model, test_loader, support_set_for_eval, feature_groups, train_dataset.label_map, device, label_column, title="Seen Attacks", save_dir=args.save_dir)

    # --- Part 2: AUC ROC Curve Visualization ---
    auc_roc_save_path = os.path.join(args.save_dir, f"auc_roc_plot_hiding_{UNSEEN_CLASS_LABEL}.png")
    plot_auc_roc_curve(model, test_loader, support_set_for_eval, feature_groups, train_dataset.label_map, device, label_column, "TTDFIOTIDS2025 Seen Attacks", auc_roc_save_path)

    # --- Part 3: Zero-Shot & Adaptation Evaluation ---
    if len(unseen_df) > 10:
        # Reserve 10 samples for adaptation experiments, use the rest for testing
        try:
            unseen_test_df, unseen_adapt_df_pool = train_test_split(unseen_df, test_size=10, stratify=unseen_df[label_column], random_state=42)
        except ValueError:
             print("Warning: Could not stratify unseen adaptation pool split. Splitting without stratification.")
             unseen_test_df, unseen_adapt_df_pool = train_test_split(unseen_df, test_size=10, random_state=42)

        # Take a smaller subset of the unseen test data for the pure ZSL evaluation
        try:
            unseen_zsl_test_df, _ = train_test_split(unseen_test_df, test_size=0.8, stratify=unseen_test_df[label_column], random_state=42)
        except ValueError:
            unseen_zsl_test_df, _ = train_test_split(unseen_test_df, test_size=0.8, random_state=42)

        # STAGE 1: PURE ZERO-SHOT EVALUATION
        zsl_support_df = train_df.groupby(label_column).head(10) 
        evaluate_zero_shot_performance(model, unseen_zsl_test_df, zsl_support_df, feature_groups, train_dataset.label_map, device, label_column, is_adapted=False, unseen_class_label=UNSEEN_CLASS_LABEL, save_dir=args.save_dir)
        
        # STAGE 2: FEW-SHOT ADAPTATION & RE-EVALUATION LOOP
        shots_to_evaluate = [1, 5, 10]
        for num_shots in shots_to_evaluate:
            if num_shots > len(unseen_adapt_df_pool):
                print(f"Skipping {num_shots}-shot: not enough samples in adaptation pool ({len(unseen_adapt_df_pool)}).")
                continue

            print(f"\n==================== STARTING {num_shots}-SHOT ADAPTATION EXPERIMENT ====================")
            
            # Start with a fresh copy of the pre-trained model for each experiment
            adapted_model = copy.deepcopy(model)
            
            # Sample the adaptation shots from the pool of 10
            unseen_adapt_df = unseen_adapt_df_pool.sample(n=num_shots, random_state=42)

            # Create the training set for fine-tuning
            adaptation_training_df = pd.concat([train_df.sample(n=2000, random_state=42), unseen_adapt_df])
            
            # Fine-tune the model
            adapted_model = fine_tune_for_adaptation(adapted_model, adaptation_training_df, feature_groups, device, num_shots, label_column, hyperparams)
            
            # Evaluate the adapted model on a combined test set of both seen and unseen attack classes
            print(f"\n--- Evaluating Adapted Model ({num_shots}-shot) on Combined Seen and Unseen Test Data ---")
            combined_test_df = pd.concat([test_df, unseen_test_df])
            
            # The support set for evaluation now includes the adaptation shots
            adapted_support_df = pd.concat([train_df, unseen_adapt_df]).groupby(label_column).head(10)
            
            # Create a new label map that includes all classes (seen and unseen)
            full_label_map = dict(enumerate(pd.concat([train_df, unseen_df])[label_column].astype('category').cat.categories))
            
            combined_dataset = StandardDataset(combined_test_df, feature_groups, label_column)
            combined_loader = DataLoader(combined_dataset, batch_size=args.eval_batch_size)
            
            evaluate_known_attacks(adapted_model, combined_loader, adapted_support_df, feature_groups, full_label_map, device, label_column, title=f"Adapted Model ({num_shots}-shot) on All Attacks", save_dir=args.save_dir)

    else:
        print("\nSkipping Zero-Shot & Adaptation Evaluation: Not enough samples in the unseen class.")

    print(f"\n==================== EXPERIMENT FOR HIDING '{UNSEEN_CLASS_LABEL}' COMPLETE ====================")

if __name__ == "__main__":
    main()