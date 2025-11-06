PANDORA-AE/
├── .gitattributes          <-- (I will generate this next) Tells Git to use LFS for CSVs.
├── .gitignore              <-- (I will generate this next) Ignores cache and results.
├── ARTIFACT_APPENDIX.tex   <-- The 2-page document for the AE chairs.
├── Dockerfile              <-- Our Python 3.12 environment recipe.
├── README.md               <-- The main "How-to" guide for the evaluator.
├── requirements.txt        <-- Our *minimal* list of Python libraries.
|
├── data/                   <-- Folder for your datasets.
│   ├── CICIDS2017_Ready.csv
│   ├── CICIoT2023_Ready.csv
│   └── TTDF_IoT_IDS_2025_Ready_Again.csv
|
├── reproduce_results/      <-- The "push-button" shell scripts.
│   ├── reproduce_all.sh    <-- The master script to run everything.
│   ├── reproduce_cicids_ablation.sh
│   ├── reproduce_cicids_vs_ptnids_s1.sh
│   ├── reproduce_cicids_vs_ptnids_s2.sh
│   ├── reproduce_cicids_vs_ptnids_s3.sh
│   ├── reproduce_ciciot2023.sh
│   ├── reproduce_dataset_evaluation.sh
│   ├── reproduce_imbalanced_scenario.sh
│   ├── reproduce_loss_ablation.sh
│   ├── reproduce_mamba_vs_transformer.sh
│   └── reproduce_ttdfiotids2025.sh
|
├── results/                <-- This folder will be EMPTY in the repo.
│   └── .gitkeep            <-- An empty file so Git creates the folder.
|
└── scripts/                <-- Your 10 converted Python scripts.
    ├── run_cicids_ablation.py
    ├── run_cicids_vs_ptnids_s1.py
    ├── run_cicids_vs_ptnids_s2.py
    ├── run_cicids_vs_ptnids_s3.py
    ├── run_ciciot2023.py
    ├── run_dataset_evaluation.py
    ├── run_imbalanced_scenario.py
    ├── run_loss_ablation.py
    ├── run_mamba_vs_transformer.py
    └── run_ttdfiotids2025.py
