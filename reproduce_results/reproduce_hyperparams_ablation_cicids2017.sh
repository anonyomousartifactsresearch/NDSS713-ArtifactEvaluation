#!/bin/bash
# ============================================
# Run Ablation Study for Pandora (S1)
# ============================================

# Activate virtual environment (optional)
# source venv/bin/activate

# Experiment name
EXP_NAME="pandora_ablation_cicids2017"
LOG_DIR="logs"
RESULTS_DIR="results/pandora_ablation"

# Create directories if not exist
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Log file (no timestamp)
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

echo "============================================"
echo "Running Ablation Study: ${EXP_NAME}"
echo "Start Time: $(date)"
echo "Logs will be saved to: ${LOG_FILE}"
echo "============================================"

# Run the Python script
python3 scripts/run_pandora_ablation_cicids2017.py | tee "$LOG_FILE"

echo "============================================"
echo "Ablation Study Completed!"
echo "End Time: $(date)"
echo "Results saved in: ${RESULTS_DIR}"
echo "Log file: ${LOG_FILE}"
echo "============================================"
