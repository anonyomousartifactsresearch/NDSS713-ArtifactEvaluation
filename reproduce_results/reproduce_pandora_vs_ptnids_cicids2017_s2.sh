#!/bin/bash
# =============================================================
# Reproduce PANDORA (CICIDS2017 - Scenario 2)
# This script runs the experiment hiding a *group* of classes.
# =============================================================

# Activate your environment if required
# source ~/miniconda3/bin/activate pytorch_env

# Define paths and directories
# Using relative paths from S1 reference for better reproducibility
DATA_PATH="data/CICIDS2017_Ready.csv" 
# Or use your full path:
# DATA_PATH="/home/avinash-awasthi/Downloads/Conferences/NDSS_2026/CICIDS2017/CICIDS2017_Ready.csv"

SAVE_DIR="results/pandora_s2_cicids2017/"
SCRIPT_PATH="scripts/run_pandora_vs_ptnids_cicids2017_s2.py"

# Define unseen classes (comma-separated, flexible)
# This matches your new code's logic
UNSEEN_CLASSES="Web Attack,DoS"

# Training parameters (identical to your new code)
EPOCHS=100
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=15
EVAL_BATCH_SIZE=256

# Model hyperparameters (identical to your new code)
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

echo "Starting PANDORA S2 experiment..."
echo "Hiding classes: $UNSEEN_CLASSES"
echo "Saving results to: $SAVE_DIR"

python "$SCRIPT_PATH" \
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