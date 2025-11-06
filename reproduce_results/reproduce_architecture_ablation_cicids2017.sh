#!/bin/bash
# =============================================================
# Run PANDORA Architecture Ablation (Mamba vs. Transformer)
# Dataset: CICIDS2017
# =============================================================

# Activate your environment if required
# source ~/miniconda3/bin/activate pytorch_env

# Define paths and directories
DATA_PATH="data/CICIDS2017_Ready.csv"
SAVE_DIR="results/arch_ablation_cicids2017/"
SCRIPT_PATH="scripts/run_architecture_ablation.py" #<-- Path to the new .py script

# Training parameters (from notebook)
EPOCHS=5
TRAIN_EPISODES=500
VAL_EPISODES=100
PATIENCE=3
EVAL_BATCH_SIZE=256
QUANTILE_N=1000

# Model hyperparameters (from notebook)
D_MODEL=64
NUM_BLOCKS=1
NUM_EXPERTS=2
DROPOUT_RATE=0.5
N_HEADS=2
D_FF=128 #<-- New parameter for Transformer
LR=0.0001
TRIPLET_LOSS_WEIGHT=0.75

# =============================================================
# Run the experiment
# =============================================================

echo "Starting Architecture Ablation for CICIDS2017..."
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
    --d_ff $D_FF \
    --lr $LR \
    --triplet_loss_weight $TRIPLET_LOSS_WEIGHT \
    --quantile_n $QUANTILE_N

echo "============================================================="
echo "   Architecture Ablation for CICIDS2017 complete."
echo "   Results saved in: $SAVE_DIR"
echo "============================================================="