#!/bin/bash
# =============================================================
# Run PANDORA Ablation Study (CICIDS2017)
# =============================================================

# Activate your environment if required
# source ~/miniconda3/bin/activate pytorch_env

# Define paths and directories
DATA_PATH="data/CICIoT2023_Ready.csv"
SAVE_DIR="results/loss_ablation_ciciot2023/"
SCRIPT_PATH="scripts/run_loss_ablation.py" #<-- Assumes script is in 'scripts/'

# Training parameters
EPOCHS=5
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=3
EVAL_BATCH_SIZE=256
QUANTILE_N=1000

# Model hyperparameters (same as reference)
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

echo "Starting Ablation Study for CICIoT2023..."
echo "Saving results to $SAVE_DIR"

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
echo "   Ablation study for CICIoT2023 complete."
echo "   Results saved in: $SAVE_DIR"
echo "============================================================="