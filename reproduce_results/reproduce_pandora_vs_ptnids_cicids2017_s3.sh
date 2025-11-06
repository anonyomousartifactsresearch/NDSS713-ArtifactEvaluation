#!/bin/bash
# =============================================================
# Reproduce PANDORA (CICIDS2017 - Scenario 3)
# This script runs the experiment hiding 'DDoS,PortScan,Bot'
# =============================================================

DATA_PATH="data/CICIDS2017_Ready.csv"
# Or full path (uncomment below if needed)
# DATA_PATH="/home/avinash-awasthi/Downloads/NDSS_2025/CICIDS2017/CICIDS2017_Ready.csv"

SAVE_DIR="results/pandora_s3_cicids2017/"
SCRIPT_PATH="scripts/run_pandora_vs_ptnids_cicids2017_s3.py"

# --- 3. Define unseen classes (comma-separated) ---
UNSEEN_CLASSES="DDoS,PortScan,Bot"

# --- 4. Training parameters ---
EPOCHS=100
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=15
EVAL_BATCH_SIZE=256

# --- 5. Model hyperparameters ---
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
echo "Starting PANDORA S3 experiment..."
echo "Hiding classes: $UNSEEN_CLASSES"
echo "Saving results to: $SAVE_DIR"
echo "============================================================="

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
