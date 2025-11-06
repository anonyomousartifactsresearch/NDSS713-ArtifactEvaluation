#!/bin/bash
# =============================================================
# Reproduce PANDORA (CICIoT2023 - Random Unseen Class)
# -------------------------------------------------------------
# This script runs the experiment on the CICIoT2023 dataset,
# allowing the Python script to randomly select one attack class
# to hide.

DATA_PATH="data/CICIoT2023_Ready.csv"
# Or full path (uncomment below if needed)
# DATA_PATH="/home/beamformer/NDSS_2025/CICIoT2023_Ready.csv"

SAVE_DIR="results/pandora_ciciot2023_random/"
SCRIPT_PATH="scripts/run_pandora_ciciot2023.py"

# --- UNSEEN_CLASSES variable is NOT needed ---
# The Python script handles random unseen class selection internally.

# --- 3. Training parameters ---
EPOCHS=100
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=15
EVAL_BATCH_SIZE=256

# --- 4. Model hyperparameters ---
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

echo "============================================================="
echo "Starting PANDORA CICIoT2023 experiment..."
echo "NOTE: Python script will randomly select one attack class to hide."
echo "Saving results to: $SAVE_DIR"
echo "============================================================="

python "$SCRIPT_PATH" \
  --dataset_path "$DATA_PATH" \
  --label_column "Label" \
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
