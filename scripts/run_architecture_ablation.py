# run_architecture_ablation.py
#
# Converts the PANDORA (Mamba-MoE vs. Transformer) ablation study
# notebook into a reusable Python script parameterized with argparse.
# This script is intended to be run by the shell scripts for
# different datasets (CICIDS2017, UNSW-NB15, BoT-IoT).

# --- 1. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, label_binarize
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.manifold import TSNE
import warnings
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import os
import math
import argparse # Added for parameterization

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

# Deterministic DataLoader worker
def seed_worker(worker_id):
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)


warnings.filterwarnings('ignore')
print("Libraries imported successfully.")

# --- 2. DATA LOADING & PREPARATION (Function) ---
def load_and_prepare_dataset(dataset_path, label_column, quantile_n=1000):
    try:
        df = pd.read_csv(dataset_path)
        print(f"Successfully loaded data from: {dataset_path}")

        # --- Data Cleaning and Scaling ---
        all_features = [col for col in df.columns if col != label_column]
        df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Remove infinite values if any
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Use QuantileTransformer for advanced scaling
        scaler = QuantileTransformer(n_quantiles=quantile_n, output_distribution='normal', random_state=42)
        df[all_features] = scaler.fit_transform(df[all_features])

        print("\nData prepared and scaled using QuantileTransformer.")
        return df

    except (FileNotFoundError, KeyError) as e:
        print(f"---! ERROR !--- An error occurred: {e}. Please check the file path and column names.")
        return pd.DataFrame()

# --- 3. FEATURE ENGINEERING (Function) ---
def engineer_features(df, label_column):
    if df.empty:
        print("\nSkipping feature processing (input DataFrame is empty).")
        return pd.DataFrame(), {}

    # Define all features
    all_features = [col for col in df.columns if col != label_column]
    print(f"\nUsing all {len(all_features)} features for the model.")

    print("\nLogically splitting features into Temporal and Volumetric groups...")
    temporal_keywords = ['duration', 'rate', 'srate', 'drate', 'iat','idle','active']

    temporal_features = [f for f in all_features if any(keyword in f.lower() for keyword in temporal_keywords)]
    volumetric_features = [f for f in all_features if f not in temporal_features]

    if not temporal_features or not volumetric_features:
        print("Warning: Logical split resulted in an empty feature group. Falling back to a random split.")
        all_features_copy = list(all_features)
        random.shuffle(all_features_copy)
        split_point = len(all_features_copy) // 2
        temporal_features = all_features_copy[:split_point]
        volumetric_features = all_features_copy[split_point:]

    feature_groups = {'temporal': temporal_features, 'volumetric': volumetric_features}
    print(f"Temporal Modality Features ({len(temporal_features)}): {feature_groups['temporal']}")
    print(f"Volumetric Modality Features ({len(volumetric_features)}): {feature_groups['volumetric']}")

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
    
    return final_df, feature_groups


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

# --- 5. AI ARCHITECTURE - PANDORA FOR ABLATION STUDY ---

# --- Encoder Core Blocks (Mamba-MoE vs. Transformer) ---

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.ln2(x + self.dropout(ffn_output))
        return x

# --- Probabilistic Encoders (Mamba vs. NEW Conventional Transformer) ---

class ProbabilisticEncoder(nn.Module): # Mamba-MoE based
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TabularTransformerProbabilisticEncoder(nn.Module): # Conventional Transformer for Tabular Data
    def __init__(self, num_features, d_model, n_heads, d_ff, num_blocks, dropout_rate=0.1):
        super().__init__()
        self.feature_embedder = nn.Linear(1, d_model) # Embed each feature value individually
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Learnable CLS token
        self.pos_encoder = PositionalEncoding(d_model, max_len=num_features + 2) # For CLS token + features
        self.encoder_blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout_rate) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_mu = nn.Linear(d_model, d_model)
        self.fc_logvar = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, num_features)
        x = x.unsqueeze(-1) # -> (batch_size, num_features, 1) to treat each feature as a token
        x_embedded = self.feature_embedder(x) # -> (batch_size, num_features, d_model)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # -> (batch_size, 1, d_model)
        x_with_cls = torch.cat((cls_tokens, x_embedded), dim=1) # -> (batch_size, num_features + 1, d_model)
        x_pos_encoded = self.pos_encoder(x_with_cls)
        x_encoded_seq = x_pos_encoded
        for block in self.encoder_blocks:
            x_encoded_seq = block(x_encoded_seq)
        cls_output = x_encoded_seq[:, 0, :] # Use only the output of the CLS token
        cls_output = self.dropout(cls_output)
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)
        return mu, logvar

# --- Full Model Architectures ---
class SotaIDS_Probabilistic(nn.Module): # PANDORA (Mamba-MoE)
    def __init__(self, num_temporal_features, num_volumetric_features, d_model, num_blocks, num_experts,
                 dropout_rate, n_heads, triplet_loss_weight):
        super().__init__()
        self.temporal_encoder = ProbabilisticEncoder(num_temporal_features, d_model, num_blocks, num_experts, dropout_rate)
        self.volumetric_encoder = ProbabilisticEncoder(num_volumetric_features, d_model, num_blocks, num_experts, dropout_rate)
        self.common_init(d_model, n_heads, triplet_loss_weight)

    def common_init(self, d_model, n_heads, triplet_loss_weight):
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
        logvar_final = self.final_fc_logvar(self.fusion(logvar_temp, logvar_vol))
        return mu_final, logvar_final

    def wasserstein_distance(self, mu1, var1, mu2, var2):
        mu1, var1 = mu1.unsqueeze(1), var1.unsqueeze(1); mu2, var2 = mu2.unsqueeze(0), var2.unsqueeze(0)
        term1 = torch.sum((mu1 - mu2)**2, dim=2); term2 = torch.sum(var1 + var2 - 2 * (var1 * var2).sqrt(), dim=2)
        return term1 + term2

    def forward(self, support_temporal, support_volumetric, support_labels, query_temporal, query_volumetric, query_labels_for_loss):
        n_way = len(torch.unique(support_labels)); k_shot = support_temporal.shape[0] // n_way
        support_mu, support_logvar = self.forward_encoder(support_temporal, support_volumetric)
        query_mu, query_logvar = self.forward_encoder(query_temporal, query_volumetric)
        proto_mu = support_mu.view(n_way, k_shot, -1).mean(dim=1)
        proto_var = torch.exp(support_logvar).view(n_way, k_shot, -1).mean(dim=1)
        distances = self.wasserstein_distance(query_mu, torch.exp(query_logvar), proto_mu, proto_var)
        log_p_y = (-distances).log_softmax(dim=1)
        loss_cls = -log_p_y.gather(1, query_labels_for_loss.view(-1, 1)).squeeze().mean()
        anchors, positives = query_mu, proto_mu[query_labels_for_loss]
        dist_matrix = torch.cdist(anchors, proto_mu)
        mask = torch.ones_like(dist_matrix).scatter_(1, query_labels_for_loss.unsqueeze(1), float('inf'))
        hard_negative_indices = torch.argmin(dist_matrix * mask, dim=1)
        negatives = proto_mu[hard_negative_indices]
        loss_triplet = self.triplet_loss(anchors, positives, negatives)
        total_loss = loss_cls + (self.triplet_loss_weight * loss_triplet)
        _, preds = torch.min(distances, 1)
        acc = (preds == query_labels_for_loss).float().mean()
        return total_loss, acc, log_p_y.exp()

class SotaIDS_Transformer(SotaIDS_Probabilistic): # PANDORA (Transformer)
    def __init__(self, num_temporal_features, num_volumetric_features, d_model, n_heads, d_ff, num_blocks,
                 dropout_rate, triplet_loss_weight):
        # Call parent init but overwrite encoders immediately after
        super().__init__(num_temporal_features, num_volumetric_features, d_model, num_blocks, 1,
                         dropout_rate, n_heads, triplet_loss_weight)
        
        # Overwrite encoders with the conventional Tabular Transformer
        self.temporal_encoder = TabularTransformerProbabilisticEncoder(
            num_temporal_features, d_model, n_heads, d_ff, num_blocks, dropout_rate
        )
        self.volumetric_encoder = TabularTransformerProbabilisticEncoder(
            num_volumetric_features, d_model, n_heads, d_ff, num_blocks, dropout_rate
        )

print("PANDORA Mamba-MoE and Conventional Transformer models defined.")

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

# --- 7. TRAINING, EVALUATION & EFFICIENCY FUNCTIONS ---

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
            loss, acc, _ = model(s_temp, s_vol, s_labels, q_temp, q_vol, q_labels)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item(); total_acc += acc.item()
            epoch_iterator.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})
    return (total_loss / episodes_per_epoch, total_acc / episodes_per_epoch) if episodes_per_epoch > 0 else (0,0)


def evaluate_model(model, test_loader, support_df, feature_groups, full_label_map, device, label_column, eval_batch_size):
    model.eval()
    y_true_codes, y_pred_codes, y_prob_list = [], [], []
    is_probabilistic = isinstance(model, SotaIDS_Probabilistic)

    support_df[label_column] = support_df[label_column].astype('category')
    support_label_codes = torch.tensor(support_df[label_column].cat.codes.values, dtype=torch.long)
    unique_support_codes = torch.unique(support_label_codes)
    n_way, support_categories = len(unique_support_codes), support_df[label_column].cat.categories

    # --- MODIFICATION: Process support set in batches to avoid OOM ---
    print("Generating prototypes from the support set in batches...")
    support_dataset = StandardDataset(support_df, feature_groups, label_column)
    support_loader = DataLoader(support_dataset, batch_size=eval_batch_size, shuffle=False)
    all_support_mu, all_support_logvar = [], []
    with torch.no_grad():
        for s_temp_batch, s_vol_batch, _ in tqdm(support_loader, desc="Generating Prototypes"):
            s_temp_batch, s_vol_batch = s_temp_batch.to(device), s_vol_batch.to(device)
            if is_probabilistic:
                mu, logvar = model.forward_encoder(s_temp_batch, s_vol_batch)
                all_support_mu.append(mu.cpu()); all_support_logvar.append(logvar.cpu())
            else: # Fallback
                mu = model.forward_encoder(s_temp_batch, s_vol_batch)
                all_support_mu.append(mu.cpu())
    support_mu = torch.cat(all_support_mu, dim=0).to(device)
    if is_probabilistic:
        support_logvar = torch.cat(all_support_logvar, dim=0).to(device)
    # --- END MODIFICATION ---

    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_var = torch.zeros(n_way, support_mu.shape[1], device=device) if is_probabilistic else None
    for i, label_code in enumerate(unique_support_codes):
        proto_mu[i] = support_mu[support_label_codes == label_code].mean(dim=0)
        if is_probabilistic:
            proto_var[i] = torch.exp(support_logvar[support_label_codes == label_code]).mean(dim=0)

    for batch_temp, batch_vol, batch_labels in tqdm(test_loader, desc="Final Evaluation"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, proto_var)
            probs = (-distances).softmax(dim=1)
            _, predicted_indices = torch.max(probs, dim=1)
        y_true_codes.extend(batch_labels.cpu().numpy())
        y_pred_codes.extend(unique_support_codes[predicted_indices.cpu()].cpu().numpy())
        y_prob_list.append(probs.cpu().numpy())

    y_prob = np.concatenate(y_prob_list, axis=0)
    y_true_names = [full_label_map[code] for code in y_true_codes]
    y_pred_names = [full_label_map[code] for code in y_pred_codes]
    macro_f1 = f1_score(y_true_names, y_pred_names, average='macro', zero_division=0)
    all_class_codes = sorted(full_label_map.keys())
    y_true_bin = label_binarize(y_true_codes, classes=all_class_codes)
    aligned_probs = np.zeros((y_prob.shape[0], len(all_class_codes)))
    support_code_to_full_code_map = {i: support_categories[i] for i in range(len(support_categories))}
    full_code_to_idx_map = {code: i for i, code in enumerate(all_class_codes)}
    for i, support_code in enumerate(unique_support_codes):
        cat_name = support_code_to_full_code_map[support_code.item()]
        full_code = [k for k,v in full_label_map.items() if v == cat_name][0]
        aligned_idx = full_code_to_idx_map[full_code]
        aligned_probs[:, aligned_idx] = y_prob[:, i]
    auc_roc = roc_auc_score(y_true_bin, aligned_probs, average='macro', multi_class='ovr') if y_true_bin.shape[1] > 1 and len(np.unique(y_true_codes)) > 1 else 0.0
    print(f"\n--- Evaluation Metrics ---\nAUC-ROC Score: {auc_roc:.4f}")
    return macro_f1, auc_roc

def measure_efficiency(model, test_loader, feature_groups, device):
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_temp = torch.randn(1, len(feature_groups['temporal'])).to(device)
    dummy_vol = torch.randn(1, len(feature_groups['volumetric'])).to(device)
    for _ in range(20): # Warm-up
        _ = model.forward_encoder(dummy_temp, dummy_vol)
    total_time, total_samples = 0, 0
    with torch.no_grad():
        for batch_temp, batch_vol, _ in tqdm(test_loader, desc="Measuring Inference Time"):
            batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
            start_time = time.time()
            _ = model.forward_encoder(batch_temp, batch_vol)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            total_time += (end_time - start_time)
            total_samples += batch_temp.size(0)
    avg_inference_time_ms = (total_time / total_samples) * 1000
    return params, avg_inference_time_ms

print("\nTraining, evaluation, and efficiency functions defined.")

# --- 8. ABLATION STUDY EXECUTION (Main Function) ---
def main():
    parser = argparse.ArgumentParser(description="PANDORA Architecture Ablation (Mamba vs. Transformer)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset CSV file.")
    parser.add_argument("--label_column", type=str, default="Label", help="Label column name.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save models and results.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--quantile_n", type=int, default=1000, help="n_quantiles for QuantileTransformer")
    # Hyperparameters
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128, help="Feed-forward dim for Transformer")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--triplet_loss_weight", type=float, default=0.75)
    args = parser.parse_args()

    # Set seed for reproducibility
    set_global_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and process data
    df = load_and_prepare_dataset(args.dataset_path, args.label_column, args.quantile_n)
    final_df, feature_groups = engineer_features(df, args.label_column)
    
    if final_df.empty:
        print("\nSkipping training as no features were selected or data was not loaded.")
        return

    ablation_configs = {
        "PANDORA (Mamba-MoE)": {'model_type': 'mamba_moe'},
        "PANDORA (Transformer)": {'model_type': 'transformer'}
    }

    results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing Device: {device}")

    final_df[args.label_column] = final_df[args.label_column].astype('category')
    full_label_map = dict(enumerate(final_df[args.label_column].cat.categories))
    
    try:
        train_val_df, test_df = train_test_split(final_df, test_size=0.2, stratify=final_df[args.label_column], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[args.label_column], random_state=42)
    except ValueError as e:
        print(f"\nStratification failed: {e}. Splitting without stratification.")
        train_val_df, test_df = train_test_split(final_df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    for name, config in ablation_configs.items():
        print(f"\n{'='*25} Running Ablation: {name} {'='*25}")

        if config['model_type'] == 'mamba_moe':
            model = SotaIDS_Probabilistic(
                num_temporal_features=len(feature_groups['temporal']), num_volumetric_features=len(feature_groups['volumetric']),
                d_model=args.d_model, num_blocks=args.num_blocks, num_experts=args.num_experts,
                dropout_rate=args.dropout_rate, n_heads=args.n_heads, triplet_loss_weight=args.triplet_loss_weight
            ).to(device)
        else: # Transformer
            model = SotaIDS_Transformer(
                num_temporal_features=len(feature_groups['temporal']), num_volumetric_features=len(feature_groups['volumetric']),
                d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, num_blocks=args.num_blocks,
                dropout_rate=args.dropout_rate, triplet_loss_weight=args.triplet_loss_weight
            ).to(device)

        train_dataset = EpisodicDataset(train_df, feature_groups, args.label_column)
        val_dataset = EpisodicDataset(val_df, feature_groups, args.label_column)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, worker_init_fn=seed_worker)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_val_loss = float('inf'); patience_counter = 0
        model_save_path = os.path.join(args.save_dir, f'best_model_{name.replace(" ", "_")}.pth')

        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, optimizer, True, args.train_episodes, device)
            val_loss, val_acc = run_epoch(model, val_loader, None, False, args.val_episodes, device)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Time: {time.time() - epoch_start_time:.2f}s")
            if val_loss < best_val_loss:
                best_val_loss = val_loss; patience_counter = 0; torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
            if patience_counter >= args.patience: print("Early stopping triggered."); break
        
        print(f"\nLoading best model checkpoint for {name} from {model_save_path}...")
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        else:
            print(f"Warning: Model checkpoint not found at {model_save_path}. Evaluating with last state.")
        
        test_dataset_standard = StandardDataset(test_df, feature_groups, args.label_column)
        test_loader_standard = DataLoader(test_dataset_standard, batch_size=args.eval_batch_size, shuffle=False)
        
        params, inference_time = measure_efficiency(model, test_loader_standard, feature_groups, device)
        macro_f1, auc_roc = evaluate_model(model, test_loader_standard, train_df, feature_groups, full_label_map, device, args.label_column, args.eval_batch_size)
        
        results.append({
            "Configuration": name,
            "Parameters": f"{params:,}",
            "Inference Time (ms/sample)": f"{inference_time:.4f}",
            "Macro F1": f"{macro_f1:.4f}",
            "AUC-ROC": f"{auc_roc:.4f}"
        })

    results_df = pd.DataFrame(results)
    print("\n\n" + "="*30)
    print(f"--- Architecture Ablation Study Results (Dataset: {args.dataset_path}) ---")
    print("="*30)
    print(results_df.to_markdown(index=False))

    # Save results to CSV
    results_csv_path = os.path.join(args.save_dir, "architecture_ablation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")


if __name__ == "__main__":
    main()