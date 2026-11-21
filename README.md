# Urban Dengue Risk Intelligence (Operation-Mosha)

A ward-level weekly dengue outbreak risk nowcasting and surveillance system for 5 major cities in Bangladesh (Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj).

## Architecture & System Design
For full system architecture, mathematical formulation of the two-stage hurdle model, feature lag engineering, and validation strategy, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Project Structure
```
Operation-Mosha/
├── app.py                     # Streamlit interactive dashboard (Ibrahim)
├── requirements.txt           # Core Python dependencies
├── README.md                  # Project overview and runbook
├── ARCHITECTURE.md            # Detailed system design & contracts
├── .gitignore                 # Repository hygiene
├── data/
│   ├── raw/
│   │   └── dengue.csv         # Raw weekly observations
│   └── processed/
│       └── cleaned.csv        # Validated, clean observations
├── src/
│   ├── __init__.py
│   ├── clean_data.py          # Deterministic data validation (Sadrul)
│   ├── features.py            # Ward-level lag engineering (Rajen)
│   ├── train.py               # Hurdle model training (Rajen)
│   └── predict.py             # Inference interface: score_rows (Rajen)
├── models/
│   └── model_bundle.joblib    # Serialized model artifact
├── reports/
│   ├── data_quality.md        # Data quality audit report (Sadrul)
│   └── validation.json        # Out-of-sample metrics & baseline comparison
└── tests/
    ├── test_clean_data.py     # Data validation tests (Sadrul)
    └── test_features.py       # Feature leakage & lag tests (Rajen)
```

## Team Responsibilities
* **Rajen (Technical Lead):** Architecture, temporal feature engineering, hurdle model, evaluation, inference contract (`score_rows`).
* **Sadrul (Data Engineering):** Data cleaning, physical validation, integrity checks, quality audit report.
* **Ibrahim (UI & Presentation):** Streamlit dashboard, scenario simulation controls, documentation, demonstration recording.

## Setup & Localhost Execution
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data validation
python src/clean_data.py

# 3. Train model & generate validation report
python src/train.py

# 4. Launch interactive dashboard
streamlit run app.py
```
