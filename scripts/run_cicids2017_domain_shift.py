# run_cicids2017_domain_shift.py
# Direct conversion of CICIDS2017_VS_PTNIDS_S1_Domain_Shift.ipynb -> python
# Preserves logic from notebook (K_SHOT=10, all-class training, specific eval functions)
# while using the sample script's style (argparse, seeding, functions).

# --- 1. LIBRARY IMPORTS ---
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import warnings
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import argparse
import os

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
    torch.set_num_threads(1)

    print(f"[INFO] Global seed fixed at {seed} — deterministic mode enabled.")

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
def load_and_prepare_dataset(cic_file_path, label_column, quantile_n=1000):
    try:
        df = pd.read_csv(cic_file_path)
        print(f"Successfully loaded data from: {cic_file_path}")

        all_features = [col for col in df.columns if col != label_column]
        df[all_features] = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        scaler = QuantileTransformer(n_quantiles=quantile_n, output_distribution='normal', random_state=42)
        df[all_features] = scaler.fit_transform(df[all_features])

        print("\nData prepared and scaled using QuantileTransformer.")
    except (FileNotFoundError, KeyError) as e:
        print(f"---! ERROR !--- An error occurred: {e}. Please check file path and column name.")
        df = pd.DataFrame()
        all_features = []

    return df, all_features

# --- CLASS MAPPING (CICIDS2017) ---
def apply_class_mapping(df, label_column):
    if df.empty:
        return df
    CLASS_MAPPING = {
        'Benign': 'Benign',
        'Bot': 'Bot',
        'Ddos': 'DDoS',
        'Dos_Goldeneye': 'DoS',
        'Dos_Hulk': 'DoS',
        'Dos_Slowhttptest': 'DoS',
        'Dos_Slowloris': 'DoS',
        'Ftp_Patator': 'Brute-Force',
        'Ssh_Patator': 'Brute-Force',
        'Web_Attack_Brute_Force': 'Web Attack',
        'Web_Attack_Sql_Injection': 'Web Attack',
        'Web_Attack_Xss': 'Web Attack',
        'Heartbleed': 'Heartbleed',
        'Infiltration': 'Infiltration',
        'Portscan': 'PortScan'
    }

    df[label_column] = df[label_column].map(CLASS_MAPPING)

    original_rows = len(df)
    df.dropna(subset=[label_column], inplace=True)
    if len(df) < original_rows:
        print(f"\nRemoved {original_rows - len(df)} rows with labels not in CLASS_MAPPING.")

    print("\nApplied class mapping to broader categories.")
    return df

# --- 3. FEATURE GROUPING ---
def split_features(df, all_features, label_column):
    # From Notebook Cell 3: Drop problematic features
    features_to_drop = ['Fwd Header Length.1', 'Flow Duration']
    df = df.drop(columns=[f for f in features_to_drop if f in df.columns])
    all_features = [col for col in df.columns if col != label_column]

    print(f"\nUsing all {len(all_features)} features for the model.")
    print("\nLogically splitting features into Temporal and Volumetric groups...")
    temporal_keywords = ['duration', 'rate', 'srate', 'drate', 'iat','idle','active']
    
    temporal_features = [f for f in all_features if any(keyword in f.lower() for keyword in temporal_keywords)]
    volumetric_features = [f for f in all_features if f not in temporal_features]
    
    if not temporal_features or not volumetric_features:
        print("Warning: Logical split resulted in an empty feature group. Falling back to a random split.")
        temp_all = list(all_features)
        random.shuffle(temp_all)
        split_point = len(temp_all) // 2
        temporal_features = temp_all[:split_point]
        volumetric_features = temp_all[split_point:]

    feature_groups = {'temporal': temporal_features, 'volumetric': volumetric_features}
    print(f"Temporal Modality Features ({len(temporal_features)}): {feature_groups['temporal']}")
    print(f"Volumetric Modality Features ({len(volumetric_features)}): {feature_groups['volumetric']}")

    final_df = df[all_features + [label_column]]
    print("\nCreated initial dataframe with all features.")

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


# --- 5. AI ARCHITECTURE - THE FULL PROBABILISTIC MODEL ---
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
        term1 = torch.sum((mu1 - mu2)**2, dim=2)
        term2 = torch.sum(var1 + var2 - 2 * (var1 * var2).sqrt(), dim=2)
        return term1 + term2

print("Full Probabilistic SOTA IDS model defined with combined Triplet Loss and Feature Attention.")

# --- 6. STANDARD & EPISODIC DATALOADERS ---
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
        
        # --- From Notebook Cell 6 ---
        self.n_way = 5
        self.k_shot = 10    # Increased from 5 to 10 for more training samples per episode
        self.n_query = 10   # Kept at 10 as requested
        # -------------------------------------------

    def __len__(self):
        return 5000 # A large number to allow for many episodes

    def __getitem__(self, index):
        valid_labels = [lbl for lbl, idx in self.data_by_class.items() if len(idx) >= self.k_shot]
        
        if not valid_labels:
            raise ValueError(f"Cannot create an episode. No classes have at least {self.k_shot} samples.")

        effective_n_way = min(self.n_way, len(valid_labels))
        selected_classes = random.sample(valid_labels, effective_n_way)

        s_idx, q_idx, s_labels, q_labels = [], [], [], []
        for i, cls in enumerate(selected_classes):
            c_idx = self.data_by_class[cls]
            num_samples_needed = self.k_shot + self.n_query
            replace = len(c_idx) < num_samples_needed
            samples_idx = np.random.choice(c_idx, size=num_samples_needed, replace=replace)
            
            s_idx.extend(samples_idx[:self.k_shot])
            q_idx.extend(samples_idx[self.k_shot:])
            s_labels.extend([i] * self.k_shot)
            q_labels.extend([i] * self.n_query)

        s_df, q_df = self.df.loc[s_idx], self.df.loc[q_idx]
        
        s_temporal = torch.tensor(s_df[self.temporal_cols].values, dtype=torch.float32)
        s_volumetric = torch.tensor(s_df[self.volumetric_cols].values, dtype=torch.float32)
        q_temporal = torch.tensor(q_df[self.temporal_cols].values, dtype=torch.float32)
        q_volumetric = torch.tensor(q_df[self.volumetric_cols].values, dtype=torch.float32)
        
        s_labels_t = torch.tensor(s_labels, dtype=torch.long)
        q_labels_t = torch.tensor(q_labels, dtype=torch.long)
        
        if len(q_labels_t) > 0:
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

# --- 7. TRAINING & VALIDATION FUNCTIONS ---
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

# --- 9. TRAINING VISUALIZATION ---
def plot_training_history(history, save_path):
    if 'train_loss' in history and history['train_loss']:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Over Epochs'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss')
        ax1.legend(); ax1.grid(True)

        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Accuracy Over Epochs'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy')
        ax2.legend(); ax2.grid(True)

        plt.savefig(save_path)
        print(f"\nTraining plots saved to {save_path}")
        plt.show()
    else:
        print("\nNo history found to plot.")

# --- 10. FEATURE IMPORTANCE ANALYSIS ---
def plot_feature_importance(model, feature_groups, save_dir, prefix):
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

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=temporal_importance.head(15))
        plt.title('Top 15 Temporal Feature Importances')
        plt.savefig(os.path.join(save_dir, f"{prefix}_temporal_features.png"))
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=volumetric_importance.head(15))
        plt.title('Top 15 Volumetric Feature Importances')
        plt.savefig(os.path.join(save_dir, f"{prefix}_volumetric_features.png"))
        plt.show()

# --- 11. FINAL EVALUATION ON ALL ATTACK CLASSES ---
def evaluate_all_attacks(model, test_loader, support_df, feature_groups, label_map, device, label_column, title=""):
    if model is None: return
    model.eval()
    y_true, y_pred = [], []
    print(f"\n--- Evaluating model performance on the test set ({title}) ---")
    
    support_df[label_column] = support_df[label_column].astype('category')
    support_df['label_code'] = support_df[label_column].cat.codes
    support_labels = torch.tensor(support_df['label_code'].values, dtype=torch.long)
    
    unique_labels = torch.unique(support_labels)
    n_way = len(unique_labels)
    
    s_temp = torch.tensor(support_df[feature_groups['temporal']].values, dtype=torch.float32).to(device)
    s_vol = torch.tensor(support_df[feature_groups['volumetric']].values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        support_mu, support_logvar = model.forward_encoder(s_temp, s_vol)
    
    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_logvar = torch.zeros(n_way, support_logvar.shape[1], device=device)
    for i in range(n_way):
        proto_mu[i] = support_mu[support_labels == unique_labels[i]].mean(dim=0)
        proto_logvar[i] = support_logvar[support_labels == unique_labels[i]].mean(dim=0)
    
    for batch_temp, batch_vol, batch_labels in tqdm(test_loader, desc=f"Evaluating ({title})"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, torch.exp(proto_logvar))
        _, predicted_class_indices = torch.min(distances, dim=1)
        
        y_true.extend([label_map[l.item()] for l in batch_labels])
        y_pred.extend([label_map[unique_labels[p_idx.item()].item()] for p_idx in predicted_class_indices])
    
    print(f"\n--- Classification Report ({title}) ---")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

# --- 12. FEW-SHOT ADAPTATION & EVALUATION FUNCTIONS ---
def run_adaptation(model, support_df, feature_groups, device, n_shot, label_column):
    """Fine-tunes a model on a small support set."""
    print(f"\n--- Starting {n_shot}-Shot Adaptation Loop ---")
    
    adapted_model = copy.deepcopy(model).to(device)
    
    for param in adapted_model.temporal_encoder.parameters():
        param.requires_grad = False
    for param in adapted_model.volumetric_encoder.parameters():
        param.requires_grad = False
        
    adapted_model.train()
    adapted_model.fusion.train()
    adapted_model.final_fc_mu.train()
    adapted_model.final_fc_logvar.train()

    params_to_tune = list(adapted_model.fusion.parameters()) + \
                     list(adapted_model.final_fc_mu.parameters()) + \
                     list(adapted_model.final_fc_logvar.parameters())
                     
    optimizer = optim.Adam(params_to_tune, lr=1e-5) 
    
    adaptation_dataset = EpisodicDataset(support_df, feature_groups, label_column)
    
    n_way_adaptation = support_df[label_column].nunique()
    if n_way_adaptation == 0:
        print("Warning: No classes in the support set. Skipping adaptation.")
        return adapted_model

    adaptation_dataset.n_way = min(5, n_way_adaptation)
    adaptation_dataset.k_shot = n_shot
    adaptation_dataset.n_query = max(1, n_shot) 

    adaptation_loader = DataLoader(adaptation_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
    
    ADAPTATION_EPISODES = 100
    run_epoch(adapted_model, adaptation_loader, optimizer, is_training=True, episodes_per_epoch=ADAPTATION_EPISODES, device=device)
    
    print(f"--- Adaptation Finished ---")
    return adapted_model


def evaluate_adapted_model(model, query_loader, support_df, feature_groups, label_map, device, label_column, title=""):
    """Evaluates a model's performance using prototypes from a given support set."""
    if model is None: return
    model.eval()
    y_true, y_pred = [], []
    print(f"\n--- Evaluating model performance on {title} ---")
    
    support_df[label_column] = support_df[label_column].astype('category')
    support_df['label_code'] = support_df[label_column].cat.codes
    support_labels = torch.tensor(support_df['label_code'].values, dtype=torch.long)
    
    unique_labels_in_support = torch.unique(support_labels)
    n_way = len(unique_labels_in_support)
    
    s_temp = torch.tensor(support_df[feature_groups['temporal']].values, dtype=torch.float32).to(device)
    s_vol = torch.tensor(support_df[feature_groups['volumetric']].values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        support_mu, support_logvar = model.forward_encoder(s_temp, s_vol)
    
    proto_mu = torch.zeros(n_way, support_mu.shape[1], device=device)
    proto_logvar = torch.zeros(n_way, support_logvar.shape[1], device=device)
    for i, label_code in enumerate(unique_labels_in_support):
        proto_mu[i] = support_mu[support_labels == label_code].mean(dim=0)
        proto_logvar[i] = support_logvar[support_labels == label_code].mean(dim=0)

    support_categories = support_df[label_column].cat.categories
    proto_label_names = [support_categories[code.item()] for code in unique_labels_in_support]

    full_name_to_code = {v: k for k, v in label_map.items()}

    for batch_temp, batch_vol, batch_labels_codes in tqdm(query_loader, desc=f"Evaluating ({title})"):
        batch_temp, batch_vol = batch_temp.to(device), batch_vol.to(device)
        with torch.no_grad():
            new_mu, new_logvar = model.forward_encoder(batch_temp, batch_vol)
            distances = model.wasserstein_distance(new_mu, torch.exp(new_logvar), proto_mu, torch.exp(proto_logvar))
        
        _, predicted_proto_indices = torch.min(distances, dim=1)
        
        y_true.extend([label_map[l_code.item()] for l_code in batch_labels_codes])
        y_pred.extend([proto_label_names[p_idx.item()] for p_idx in predicted_proto_indices])
    
    print(f"\n--- Classification Report ({title}) ---")
    all_present_labels = sorted(list(set(y_true) | set(y_pred)))
    report = classification_report(y_true, y_pred, labels=all_present_labels, zero_division=0)
    print(report)


def run_domain_shift_adaptation_experiment(base_model, cross_dataset_path, training_feature_groups, full_label_map, device, label_column, all_categories):
    """Loads a new dataset, processes it IDENTICALLY to the training set, and runs few-shot adaptation."""
    print("\n" + "="*50)
    print("--- Testing Cross-Dataset Generalization (Domain Shift) ---")
    print("--- Processing CICIDS2018 IDENTICALLY to CICIDS2017 ---")
    print("="*50)

    if not os.path.exists(cross_dataset_path):
        print(f"Cross-dataset file not found at '{cross_dataset_path}'. Skipping test.")
        return

    try:
        cross_df_orig = pd.read_csv(cross_dataset_path)
        print(f"Successfully loaded cross-dataset from: {cross_dataset_path}\n")
    except Exception as e:
        print(f"Error loading cross-dataset: {e}")
        return
        
    cross_df = cross_df_orig.copy()
    original_features = training_feature_groups['temporal'] + training_feature_groups['volumetric']
    feature_cols_in_cross = [col for col in original_features if col in cross_df.columns]
    
    cross_df[feature_cols_in_cross] = cross_df[feature_cols_in_cross].apply(pd.to_numeric, errors='coerce').fillna(0)
    cross_df.replace([np.inf, -np.inf], 0, inplace=True)
    cross_df.dropna(subset=[label_column], inplace=True)

    print("Fitting a new QuantileTransformer to the CICIDS2018 data...")
    cross_scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    cross_df[feature_cols_in_cross] = cross_scaler.fit_transform(cross_df[feature_cols_in_cross])
    print("CICIDS2018 data has been cleaned and re-scaled based on its own distribution.\n")
    
    # Apply the same categories from the training dataset
    cross_df[label_column] = pd.Categorical(cross_df[label_column], categories=all_categories, ordered=True)
    
    for n_shot in [1, 5]:
        print(f"\n{'='*20} Starting {n_shot}-Shot Evaluation {'='*20}")
        
        class_counts = cross_df[label_column].value_counts()
        classes_to_include = class_counts[class_counts >= n_shot].index
        df_for_shot = cross_df[cross_df[label_column].isin(classes_to_include)]

        if df_for_shot.empty:
            print(f"Not enough samples to conduct {n_shot}-shot learning. Skipping.")
            continue
            
        support_df = df_for_shot.groupby(label_column, group_keys=False, observed=True).apply(lambda x: x.sample(n_shot))
        query_indices = df_for_shot.index.difference(support_df.index)
        query_df = df_for_shot.loc[query_indices]

        print(f"Created {n_shot}-shot support set with {len(support_df)} samples.")
        print(f"Created query set with {len(query_df)} samples.")

        adapted_model = run_adaptation(base_model, support_df, training_feature_groups, device, n_shot, label_column)

        query_dataset = StandardDataset(query_df, training_feature_groups, label_column)
        query_loader = DataLoader(query_dataset, batch_size=256)
        
        evaluate_adapted_model(adapted_model, query_loader, support_df, training_feature_groups, full_label_map, device, label_column, title=f"CICIDS2018 Adapted on {n_shot}-Shot (Re-Scaled)")


# --- 8. PUTTING IT ALL TOGETHER & TRAINING ---
def main():
    parser = argparse.ArgumentParser(description="Reproducible conversion of CICIDS2017 Domain Shift notebook to .py")
    parser.add_argument("--dataset_path", type=str, default="/home/avinash-awasthi/Downloads/NDSS_2025/CICIDS2017/CICIDS2017_Ready.csv", help="Path to CICIDS2017 CSV.")
    parser.add_argument("--cross_dataset_path", type=str, default="/home/avinash-awasthi/Downloads/NDSS_2025/CICIDS2018_Domain_Shift-Ready_Again.csv", help="Path to CICIDS2018 CSV for domain shift.")
    parser.add_argument("--label_column", type=str, default="Label", help="Label column name.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_episodes", type=int, default=500)
    parser.add_argument("--val_episodes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save models and artifacts.")
    parser.add_argument("--quantile_n", type=int, default=1000, help="n_quantiles for QuantileTransformer")
    # hyperparams
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--triplet_loss_weight", type=float, default=0.75)
    args = parser.parse_args()

    global label_column
    label_column = args.label_column
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- Cell 2 ---
    df, all_features = load_and_prepare_dataset(args.dataset_path, label_column, args.quantile_n)
    if df.empty:
        return
    df = apply_class_mapping(df, label_column)

    # --- Cell 3 ---
    if df.empty:
        print("\nSkipping feature processing (input DataFrame is empty).")
        return
    final_df, feature_groups = split_features(df, all_features, label_column)
    
    if final_df.empty or not feature_groups:
         print("\nSkipping training as no features were selected or data was not loaded.")
         return
         
    # --- Cell 8 (Training) ---
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

    # Logic from Notebook Cell 8: Train on ALL classes
    print("\nTraining on all classes without few-shot or ZSL adaptation.")
    train_val_df, test_df = train_test_split(final_df, test_size=0.2, stratify=final_df[label_column], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_column], random_state=42)

    train_dataset = EpisodicDataset(train_df, feature_groups, label_column)
    val_dataset = EpisodicDataset(val_df, feature_groups, label_column)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
    
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
    model_save_path = os.path.join(args.save_dir, 'best_model_full_training.pth')

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, is_training=True, episodes_per_epoch=args.train_episodes, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, None, is_training=False, episodes_per_epoch=args.val_episodes, device=device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | Time: {epoch_duration:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation loss improved. Checkpoint saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
        
        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break
            
    print("Full training finished.")
    print("\nLoading best model checkpoint for final evaluation...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("Warning: No best model was saved. Using model from last epoch.")

    history_csv_path = os.path.join(args.save_dir, 'training_history_full_training.csv')
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved successfully to {history_csv_path}")

    # --- Cell 9 ---
    plot_save_path = os.path.join(args.save_dir, 'training_plots_full_training.png')
    plot_training_history(history, plot_save_path)
    
    # --- Cell 10 ---
    plot_feature_importance(model, feature_groups, args.save_dir, "cicids2017")
    
    # --- Cell 11 ---
    test_dataset_standard = StandardDataset(test_df, feature_groups, label_column)
    test_loader = DataLoader(test_dataset_standard, batch_size=args.eval_batch_size)
    
    # Create a unified label map for the entire dataset
    all_categories = final_df[label_column].astype('category').cat.categories
    full_label_map = dict(enumerate(all_categories))

    evaluate_all_attacks(model, test_loader, train_df, feature_groups, full_label_map, device, label_column, title="CICIDS2017 Test Set")
    
    # --- Cell 13 ---
    run_domain_shift_adaptation_experiment(
        model, 
        args.cross_dataset_path, 
        feature_groups, 
        full_label_map, 
        device,
        label_column,
        all_categories # Pass the category list for consistent mapping
    )

if __name__ == "__main__":
    main()