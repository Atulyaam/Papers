# Final Project Tree

```text
IDS-UNSW-NB15/
│
├── .git/
├── .venv/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── audit/
│
├── configs/
│   ├── project_config.yaml
│   └── data_schema.yaml
│
├── experiments/
│   ├── pre_registration.json
│   └── experiment_registry.json
│
├── notebooks/
│
├── src/
│   ├── preprocessing/
│   ├── feature_selection/
│   ├── models/
│   │   ├── base_models/
│   │   ├── stacking/
│   │   └── autoencoder/
│   ├── fusion/
│   ├── explainability/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│
├── tests/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── shap/
│   ├── logs/
│   └── checkpoints/
│
├── PROJECT_PLAN.md
├── DEVELOPMENT_PROCESS.md
├── ARCHITECTURE.md
├── EXPERIMENT_PROTOCOL.md
├── LEAKAGE_AND_DATA_ISOLATION.md
├── PRE_REGISTRATION.md
├── LOGGING_CHECKPOINTS_PROVENANCE.md
├── SEEDS_AND_REPRODUCIBILITY.md
├── DATA_CONTRACT.md
├── SHAP_AND_EXPLAINABILITY.md
├── EXPERIMENT_REGISTRY.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Directory Rules

- `data/raw`: immutable source files
- `data/processed`: derived data
- `data/splits`: split/protection artifacts
- `data/audit`: deterministic audit outputs
- `src`: reusable project logic
- `notebooks`: exploration only
- `tests`: automated tests
- `results`: generated outputs
- `experiments`: protocol/registry records
- `configs`: explicit configuration
