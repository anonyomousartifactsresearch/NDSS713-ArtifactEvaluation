# run_pandora_ablation_s1.py
# "As-is" conversion of the user's ablation study script.
# This script trains models with varying hyperparameters to test sensitivity.

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
def load_and_prepare_dataset(cic_file_path, label_column, quantile_n=1000):
    try:
        df = pd.read_csv(cic_file_path)
        print(f"Successfully loaded data from: {cic_file_path}")
        
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

# --- NEW: CLASS MAPPING TO MAJOR CATEGORIES ---
def apply_class_mapping(df, label_column):
    if df.empty:
        return df
        
    CLASS_MAPPING = {
        'Benign': 'Benign',
        'Bot': 'Bot',
        'Ddos': 'DDoS',
        'DDoS': 'DDoS', # Adding duplicate for robustness
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
        'Portscan': 'PortScan',
        'PortScan': 'PortScan' # Adding duplicate for robustness
    }
    
    # Apply the mapping
    df[label_column] = df[label_column].map(CLASS_MAPPING)
    
    # Drop any rows where the label might not have been in the mapping
    original_rows = len(df)
    df.dropna(subset=[label_column], inplace=True)
    if len(df) < original_rows:
        print(f"\nRemoved {original_rows - len(df)} rows with labels not in CLASS_MAPPING.")

    print("\nApplied class mapping to broader categories.")
    return df

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
        temp_all_features = list(all_features) # Create a copy
        random.shuffle(temp_all_features)
        split_point = len(temp_all_features) // 2
        temporal_features = temp_all_features[:split_point]
        volumetric_features = temp_all_features[split_point:]

    feature_groups = {'temporal': temporal_features, 'volumetric': volumetric_features}
    print(f"Temporal Modality Features ({len(temporal_features)}): {feature_groups['temporal']}")
    print(f"Volumetric Modality Features ({len(volumetric_features)}): {feature_groups['volumetric']}")
    return feature_groups

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

# --- 5. AI ARCHITECTURE - ABLATION MODEL ---

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

class SotaIDS_Probabilistic_Ablation(nn.Module):
    """
    Modified version of the model to accept a variable triplet_margin
    for the ablation study.
    """
    def __init__(self, num_temporal_features, num_volumetric_features, d_model, num_blocks, num_experts, dropout_rate, n_heads, triplet_loss_weight, triplet_margin):
        super().__init__()
        self.d_model = d_model
        self.temporal_encoder = ProbabilisticEncoder(num_temporal_features, d_model, num_blocks, num_experts, dropout_rate)
        self.volumetric_encoder = ProbabilisticEncoder(num_volumetric_features, d_model, num_blocks, num_experts, dropout_rate)
        self.fusion = CrossAttentionFusion(d_model, n_heads)
        self.final_fc_mu = nn.Linear(d_model * 2, d_model * 2)
        self.final_fc_logvar = nn.Linear(d_model * 2, d_model * 2)
        # Pass the margin parameter to the TripletMarginLoss
        self.triplet_loss = nn.TripletMarginLoss(margin=triplet_margin, p=2)
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

print("Full Probabilistic SOTA IDS (Ablation) model defined.")


# --- 6. DATALOADERS ---

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

# StandardDataset is not used by the ablation study, but included for completeness
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


# --- 8. MAIN SCRIPT ---

def main():
    parser = argparse.ArgumentParser(description="Reproducible SOTA IDS Ablation Study on CICIDS2017.")
    parser.add_argument("--dataset_path", type=str, default="data/CICIDS2017_Ready.csv", help="Path to CICIDS2017 CSV.")
    parser.add_argument("--label_column", type=str, default="Label", help="Label column name.")
    parser.add_argument("--unseen_class", type=str, default="DDoS", help="Single unseen class to hide for creating train/val sets.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save models and artifacts.")
    parser.add_argument("--quantile_n", type=int, default=1000, help="n_quantiles for QuantileTransformer")
    
    # Ablation-specific training parameters
    parser.add_argument("--sens_epochs", type=int, default=15, help="Number of epochs for each ablation run.")
    parser.add_argument("--sens_train_episodes", type=int, default=150, help="Training episodes per epoch.")
    parser.add_argument("--sens_val_episodes", type=int, default=50, help="Validation episodes per epoch.")

    # Base hyperparams exposed as args
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--triplet_loss_weight", type=float, default=0.75)
    parser.add_argument("--triplet_margin", type=float, default=1.0)
    args = parser.parse_args()

    # This global is used by the dataloaders
    global label_column
    label_column = args.label_column
    
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. DATA LOADING & PREPARATION ---
    df, all_features = load_and_prepare_dataset(args.dataset_path, label_column, quantile_n=args.quantile_n)
    if df.empty:
        print("Empty dataframe loaded — exiting.")
        return

    # --- CLASS MAPPING ---
    df = apply_class_mapping(df, label_column)

    # --- 3. FEATURE GROUPING ---
    feature_groups = split_features(all_features)
    if not feature_groups:
        print("\nSkipping feature processing (feature split failed).")
        return

    final_df = df[all_features + [label_column]]
    print("\nCreated initial dataframe with all features.")

    # --- Data cleaning ---
    print("\nCleaning data: Removing classes with fewer than 10 samples...")
    class_counts = final_df[label_column].value_counts()
    classes_to_keep = class_counts[class_counts >= 10].index
    original_rows = len(final_df)
    final_df = final_df[final_df[label_column].isin(classes_to_keep)]
    print(f"Removed {original_rows - len(final_df)} rows belonging to very rare classes.")

    print("\nFinal class distribution:")
    print(final_df[label_column].value_counts())

    if final_df.empty:
        print("No data available after cleaning; exiting.")
        return

    # --- 8. DATA SPLITTING FOR EXPERIMENTS ---
    UNSEEN_CLASS_LABEL = args.unseen_class
    print(f"\nConfiguring Zero-Shot experiment: Hiding ATTACK class '{UNSEEN_CLASS_LABEL}' from training.")

    seen_df = final_df[final_df[label_column] != UNSEEN_CLASS_LABEL]
    unseen_df = final_df[final_df[label_column] == UNSEEN_CLASS_LABEL] # Not used in this script, but good practice

    if seen_df.empty:
        print(f"Error: No data left after excluding unseen class '{UNSEEN_CLASS_LABEL}'. Exiting.")
        return

    try:
        train_val_df, test_df = train_test_split(seen_df, test_size=0.2, stratify=seen_df[label_column], random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_column], random_state=42)
    except ValueError as e:
        print(f"\nStratification failed: {e}. Splitting without stratification.")
        train_val_df, test_df = train_test_split(seen_df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    # --- 12. HYPERPARAMETER ABLATION STUDY ---
    print("\n--- Starting Hyperparameter Ablation Study ---")

    # Base hyperparameters from args
    base_hyperparams = {
        'd_model': args.d_model,
        'num_blocks': args.num_blocks,
        'num_experts': args.num_experts,
        'dropout_rate': args.dropout_rate,
        'n_heads': args.n_heads,
        'LR': args.lr,
        'triplet_loss_weight': args.triplet_loss_weight, # Lambda (λ)
        'triplet_margin': args.triplet_margin       # Margin (m)
    }

    # Define the parameter ranges for the ablation study
    params_to_tune = {
        'd_model': [32, 64, 128],
        'num_experts': [2, 4, 8],
        'dropout_rate': [0.2, 0.35, 0.5],
        'triplet_loss_weight': [0.25, 0.5, 0.75, 1.0], # Lambda (λ)
        'triplet_margin': [0.5, 1.0, 1.5, 2.0]        # Margin (m)
    }

    # Store results, including epoch-by-epoch accuracy
    sensitivity_results = {}

    # Set training parameters for the study
    SENS_EPOCHS = args.sens_epochs
    SENS_EPISODES_TRAIN = args.sens_train_episodes
    SENS_EPISODES_VAL = args.sens_val_episodes

    # --- Check for Data and Run Study ---
    if not train_df.empty:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning ablation study on device: {device}")
        
        train_dataset_sens = EpisodicDataset(train_df, feature_groups, label_column)
        val_dataset_sens = EpisodicDataset(val_df, feature_groups, label_column)
        train_loader_sens = DataLoader(train_dataset_sens, batch_size=1, shuffle=True, worker_init_fn=seed_worker)
        val_loader_sens = DataLoader(val_dataset_sens, batch_size=1, shuffle=False, worker_init_fn=seed_worker)

        for param_name, param_values in params_to_tune.items():
            print(f"\n---> Testing sensitivity for parameter: '{param_name}' <--- ")
            
            # Store accuracy history for each value of the parameter
            accuracies_over_epochs = []

            for value in param_values:
                print(f"  -> Training with {param_name} = {value}")
                current_hyperparams = base_hyperparams.copy()
                current_hyperparams[param_name] = value

                model_params = current_hyperparams.copy()
                del model_params['LR'] # LR is for optimizer, not model

                model_sens = SotaIDS_Probabilistic_Ablation(
                    num_temporal_features=len(feature_groups['temporal']),
                    num_volumetric_features=len(feature_groups['volumetric']),
                    **model_params
                ).to(device)

                optimizer_sens = optim.Adam(model_sens.parameters(), lr=current_hyperparams['LR'])
                
                # Store validation accuracy for each epoch
                epoch_accuracies = []
                
                for epoch in range(SENS_EPOCHS):
                    _, _ = run_epoch(model_sens, train_loader_sens, optimizer_sens, is_training=True, episodes_per_epoch=SENS_EPISODES_TRAIN, device=device)
                    _, val_acc = run_epoch(model_sens, val_loader_sens, None, is_training=False, episodes_per_epoch=SENS_EPISODES_VAL, device=device)
                    epoch_accuracies.append(val_acc)
                
                print(f"    Best Validation Accuracy for value {value}: {max(epoch_accuracies):.4f}")
                accuracies_over_epochs.append(epoch_accuracies)

            sensitivity_results[param_name] = {'values': param_values, 'accuracies_over_epochs': accuracies_over_epochs}

        print("\n--- Ablation Study Complete ---")

        # --- Visualize Results ---
        if sensitivity_results:
            plot_titles = {
                'd_model': 'Model Dimension (d_model)',
                'num_experts': 'Number of Experts',
                'dropout_rate': 'Dropout Rate',
                'triplet_loss_weight': 'Lambda (λ) for Triplet Loss',
                'triplet_margin': 'Margin (m) for Triplet Loss'
            }

            # Generate a separate plot for each hyperparameter
            for param_name, results in sensitivity_results.items():
                plt.style.use('seaborn-v0_8-whitegrid')
                plt.rc('font', size=14)
                plt.figure(figsize=(12, 8))
                
                # Use a clear, high-contrast color palette
                colors = ['#d62728', '#2ca02c', '#1f77b4', '#9467bd', '#ff7f0e']
                
                for i, value in enumerate(results['values']):
                    accuracies = results['accuracies_over_epochs'][i]
                    plt.plot(range(1, SENS_EPOCHS + 1), accuracies, 
                             linestyle='--', marker='o', color=colors[i % len(colors)],
                             label=f'{param_name} = {value}')
                
                plt.title(f'Sensitivity to {plot_titles.get(param_name, param_name)}', fontsize=20, fontweight='bold')
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Validation Accuracy', fontsize=16)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.legend(title=f'{param_name} Values', fontsize=12)
                plt.ylim(bottom=0.5, top=1.0)
                
                plt.tight_layout()
                save_path = os.path.join(args.save_dir, f'hyperparameter_sensitivity_{param_name}.png')
                plt.savefig(save_path)
                print(f"Saved sensitivity plot to {save_path}")
                plt.show()

    else:
        print("\nSkipping ablation study as training data (`train_df`) is not available.")

if __name__ == "__main__":
    main()
