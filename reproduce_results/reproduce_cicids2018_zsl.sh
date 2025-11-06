#!/bin/bash
# =============================================================
# Reproduce CICIDS2018 ZSL 5-shot Notebook
# This script ensures full reproducibility of the notebook run.
# =============================================================

# Activate your environment if required
# source ~/miniconda3/bin/activate pytorch_env

# Define paths and directories
DATA_PATH="data/CICIDS2018_Domain_Shift-Ready_Again.csv"
SAVE_DIR="results/cicids2018_zsl_5_shot/"
SCRIPT_PATH="scripts/run_cicids2018_zsl.py" # Assumes script is in 'scripts' dir

# Training parameters (identical to notebook)
EPOCHS=100
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=15
EVAL_BATCH_SIZE=256
QUANTILE_N=1000

# Model hyperparameters (same as notebook)
D_MODEL=64
NUM_BLOCKS=1
NUM_EXPERTS=2
DROPOUT_RATE=0.5
N_HEADS=2
LR=0.0001
TRIPLET_LOSS_WEIGHT=0.75

# =============================================================
# Run the experiment
# =============================================================

echo "Running CICIDS2018 Training..."
mkdir -p $SAVE_DIR

python $SCRIPT_PATH \
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
echo "   CICIDS2018 experiment completed successfully."
echo "   Results saved in: $SAVE_DIR"
echo "============================================================="