#!/bin/bash
# =============================================================
# Reproduce Pandora vs PTNIDS (CICIDS2017 - Scenario 1)
# This script ensures full reproducibility of the notebook run
# using the updated .py file that supports multiple unseen classes.
# =============================================================

# Activate your environment if required
# source ~/miniconda3/bin/activate pytorch_env

# Define paths and directories
DATA_PATH="data/CICIDS2017_Ready.csv"
SAVE_DIR="results/pandora_vs_ptnids_s1/"

# Define unseen classes (comma-separated, flexible)
# Example options:
# UNSEEN_CLASSES="DDoS"
# UNSEEN_CLASSES="DDoS,Bot"
# UNSEEN_CLASSES="DDoS,Bot,PortScan"
UNSEEN_CLASSES="DDoS"

# Training parameters (identical to notebook)
EPOCHS=100
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=15
EVAL_BATCH_SIZE=256

# Model hyperparameters (same as notebook)
D_MODEL=64
NUM_BLOCKS=1
NUM_EXPERTS=2
DROPOUT_RATE=0.5
N_HEADS=2
LR=0.0001
TRIPLET_LOSS_WEIGHT=0.75
QUANTILE_N=1000

# =============================================================
# Run the experiment
# =============================================================

python scripts/run_pandora_vs_ptnids_cicids2017_s1.py \
    --dataset_path "$DATA_PATH" \
    --label_column "Label" \
    --unseen_classes "$UNSEEN_CLASSES" \
    --epochs $EPOCHS \
    --train_episodes $TRAIN_EPISODES \
    --val_episodes $VAL_EPISODES \
    --patience $PATIENCE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --save_dir "$SAVE_DIR" \
    --d_model $D_MODEL \
    --num_blocks $NUM_BLOCKS \
    --num_experts $NUM_EXPERTS \
    --dropout_rate $DROPOUT_RATE \
    --n_heads $N_HEADS \
    --lr $LR \
    --triplet_loss_weight $TRIPLET_LOSS_WEIGHT \
    --quantile_n $QUANTILE_N

echo "============================================================="
echo "   All experiments completed successfully."
echo "   Results saved in: $SAVE_DIR"
echo "============================================================="
