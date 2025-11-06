# PANDORA-AE: Artifact Evaluation

> **PANDORA:** Lightweight Adversarial Defense for Edge IoT using Uncertainty-Aware Metric Learning  
> Artifact Evaluation (AE) package for the PANDORA paper (NDSS 2026 submission).  
> *This README was prepared using the project's artifact documentation (AE.docx).*

---

## Table of contents
1. [Overview](#overview)  
2. [Highlights / Claims](#highlights--claims)  
3. [Getting started](#getting-started)  
   - [Recommended: Docker (quick)](#recommended-docker-quick)  
   - [Build from scratch (manual)](#build-from-scratch-manual)  
4. [Repository structure](#repository-structure)  
5. [How to reproduce key experiments](#how-to-reproduce-key-experiments)  
6. [Expected outputs / results layout](#expected-outputs--results-layout)  
7. [Notes & dataset info](#notes--dataset-info)  
8. [Contact / citation](#contact--citation)  
9. [License](#license)

---

# Overview

PANDORA-AE is the official artifact package that contains source code, required datasets (CSV-ready variants via Git LFS), pre-configured Docker image, and reproduction scripts to re-run the experiments described in the PANDORA paper. The artifact reproduces main claims (SOTA comparison, ablations, few-shot/zero-shot results, hyperparameter sensitivity) and produces the tables/figures reported in the manuscript.

---

# Highlights / Claims

The artifact enables reproduction of the following main claims from the paper:

- **SOTA comparison**: PANDORA vs PTN-IDS on CICIDS2017 under few-shot settings (Table III).  
- **Loss ablation**: Components of the PMSD loss (Wasserstein, Triplet, Euclidean) (Table VII).  
- **Architecture ablation**: Mamba-MoE encoder vs Transformer encoder (Table VIII).  
- **Main results**: Full training & evaluation on `CICIoT2023` and `TTDFIOTIDS2025` including known/zero-shot/few-shot experiments.  
- **Hyperparameter sensitivity**: Sensitivity plots for CICIDS2017 experiments.  
These claims and their reproduction steps are documented and automated in the `reproduce_results/` scripts.

---

# Getting started

> Two setup options are provided. **Docker** is recommended for speed and reproducibility.

## Recommended: Docker (quick)

1. Pull the pre-built image (GHCR):

```bash
docker pull ghcr.io/anonyomousartifactsresearch/pandora-ae:1.0
```

2. Create a local `results/` folder and run:

```bash
mkdir -p $(pwd)/results

# If you have an NVIDIA GPU
docker run -it --name pandora-ae   -v $(pwd)/results:/app/results   --gpus all   ghcr.io/anonyomousartifactsresearch/pandora-ae:1.0

# If no GPU, omit --gpus all (Recommended):
docker run -it --name pandora-ae -v $(pwd)/results:/app/results ghcr.io/anonyomousartifactsresearch/pandora-ae:1.0
```

Inside the container you'll be at `/app` (or similar). Use the `reproduce_results/` scripts to run experiments. All outputs will appear in your host `results/` folder thanks to the volume mount.

## Build from scratch (manual)

> Use this if you prefer a local environment without Docker.

1. Install Git LFS and clone the repo:

```bash
# Install git-lfs first (platform-specific)
git clone https://github.com/anonyomousartifactsresearch/NDSS713-ArtifactEvaluation.git
cd PANDORA-AE
git lfs pull
```

2. Option A — Build Docker image locally:

```bash
docker build -t pandora-ae:local .
mkdir -p $(pwd)/results
docker run -it --name pandora-ae -v $(pwd)/results:/app/results --gpus all pandora-ae:local
```

2. Option B — Local Python environment (no Docker):

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the reproduction scripts from `reproduce_results/`.

---

# Repository structure (top-level)

```
PANDORA-AE/
├── .gitattributes          # Git LFS config
├── .gitignore
├── Dockerfile
├── Pandora_NDSS2026.pdf    # Paper PDF
├── README.md               # (this file)
├── requirements.txt
├── data/                   # Ready CSVs (managed via Git LFS)
│   ├── CICIDS2017_Ready.csv
│   ├── CICIoT2023_Ready.csv
│   └── TTDF_IoT_IDS_2025_Ready_Again.csv
|   └── CICIDS2018_Domain_Shift-Ready_Again.csv.csv

├── reproduce_results/      # Shell scripts that run experiments end-to-end
│   ├── reproduce_pandora_vs_ptnids_cicids2017_s1.sh
│   ├── reproduce_loss_ablation_cicids2017.sh
│   └── ...
├── results/                # (empty) produced outputs are saved here
├── scripts/                # Core Python scripts to train/eval/plot
│   ├── run_architecture_ablation.py
│   ├── run_loss_ablation.py
│   └── run_pandora_vs_ptnids_cicids2017_s3.py
└── docs/                   # Optional experimental notes, config examples
```

(See repository root for the full listing.)

---

# How to reproduce key experiments

> All commands below assume you're either inside the Docker container or have activated the Python virtualenv and are at the repo root.

## 0) Run all the results
```bash
bash run_all.sh
```

## 1) SOTA comparison (CICIDS2017)
Scenario 1 (Hide DDoS):
```bash
bash reproduce_results/reproduce_pandora_vs_ptnids_cicids2017_s1.sh
```

Scenario 2 (Hide Web Attack, DoS):
```bash
bash reproduce_results/reproduce_pandora_vs_ptnids_cicids2017_s2.sh
```

Scenario 3 (Hide DDoS, PortScan, Bot):
```bash
bash reproduce_results/reproduce_pandora_vs_ptnids_cicids2017_s3.sh
```

## 2) Main results (CICIoT2023 & TTDFIOTIDS2025)
```bash
bash reproduce_results/reproduce_pandora_ciciot2023_s_random.sh
bash reproduce_results/reproduce_pandora_ttdfiot_s_random.sh
```

## 3) Ablation studies (loss & architecture)
Loss ablation on CICIDS2017:
```bash
bash reproduce_results/reproduce_loss_ablation_cicids2017.sh
```

Architecture ablation on CICIoT2023:
```bash
bash reproduce_results/reproduce_architecture_ablation_ciciot2023.sh
```

## 4) Hyperparameter sensitivity (CICIDS2017)
```bash
bash reproduce_results/reproduce_hyperparams_ablation_cicids2017.sh
```

## 5) Domain Shift Results
Few-Shot Adaptation for CICIDS2017:
```bash
bash reproduce_results/reproduce_cicids2017_domain_shift.sh
```
Zero Shot Learning for CICIDS2018:
```bash
reproduce_results/reproduce_cicids2018_zsl.sh
```

Each script will write logs, training artifacts, CSVs, and plots to `results/<experiment_folder>/`.

---

# Expected outputs / results layout

After running the reproduce scripts, `results/` will contain per-experiment subfolders:

```
results/
├── pandora_s2_cicids2017/
│   ├── best_model_s2_attention_triplet.pth
│   ├── training_history_s2_attention_triplet.csv
│   └── training_plots_s2_attention_triplet.png
├── loss_ablation_ciciot2023/
│   ├── ablation_results.csv
│   └── ...
└── arch_ablation_cicids2017/
    ├── architecture_ablation_results.csv
    └── best_model_... .pth
```

All plots are saved as `.png` files and numeric results (per-run metrics) are saved as CSVs for easy table reproduction. If you used the Docker volume mount (`-v`), these files are available on your host after the container exits.

---

# Notes & dataset info

- The `data/` folder contains *ready-to-use* CSVs (smaller, preprocessed variants) stored via **Git LFS**. The raw PCAP archives are not required for AE reproduction.  
- If cloning manually, run `git lfs pull` after cloning to fetch the large files.  
- Scripts automatically detect CUDA/GPU; if absent they fall back to CPU.

---

# Contact / citation

If you use this artifact in your work, cite the PANDORA paper (see `Pandora_NDSS2026.pdf` in repo). For issues or questions, open an issue on the repository or contact the authors (see paper header). This README and reproduction instructions were prepared from the artifact's AE documentation (AE.docx).

---

# License

See the repository `LICENSE` file for licensing details. If no license file is present, contact the authors before reusing code or data.

---

## Quick checklist before running experiments
- [ ] Docker installed (recommended) OR Python 3.12 environment ready  
- [ ] `git lfs` installed and `git lfs pull` executed (if building locally)  
- [ ] `results/` folder mount or local folder exists and is writable  
