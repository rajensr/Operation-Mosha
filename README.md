# Urban Dengue Risk Intelligence (Operation-Mosha)

A ward-level weekly dengue outbreak risk nowcasting and surveillance decision-support system for 5 major cities in Bangladesh (**Chattogram, Dhaka North, Dhaka South, Gazipur, Narayanganj**).

Built for the **EDU AI Engineering Hackathon 2026**.

---

## 🏛️ Architecture & System Design
For full system architecture, mathematical formulation of the two-stage hurdle model, feature lag engineering, and validation strategy, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📁 Project Structure
```text
Operation-Mosha/
├── app.py                     # Streamlit interactive dashboard (Ibrahim)
├── requirements.txt           # Core Python dependencies
├── README.md                  # Project overview and runbook (Ibrahim)
├── ARCHITECTURE.md            # Detailed system design & contracts (Rajen)
├── .gitignore                 # Repository hygiene
├── data/
│   ├── raw/
│   │   └── dengue.csv         # Raw weekly observations (31,320 rows)
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

---

## 👥 Team Responsibilities
* **Rajen (Technical Lead):** System architecture, temporal feature engineering, two-stage hurdle model, baseline benchmarking, and the inference contract (`score_rows`).
* **Sadrul (Data Engineering):** Deterministic data cleaning, physical bounds validation, temporal continuity, and comprehensive data audit reporting.
* **Ibrahim (UI & Presentation):** Streamlit dashboard development, interactive weather simulation controls, documentation, and demonstration script.

---

## 🚀 Setup & Localhost Execution

### Prerequisites
- Python 3.11.x
- Git

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/rajensr/Operation-Mosha.git
cd Operation-Mosha

# Create virtual environment and install dependencies
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Cleaning & Audit
```bash
python src/clean_data.py
# Runs audit and exports data/processed/cleaned.csv and reports/data_quality.md
```

### 3. Run Automated Tests
```bash
pytest -v
# Executes 17 unit tests covering schema, physical bounds, lag boundaries, and inference contracts
```

### 4. Train Hurdle Model & Benchmark Baselines
```bash
python src/train.py
# Fits Stage 1 classifier and Stage 2 regressor, saves models/model_bundle.joblib, and exports reports/validation.json
```

### 5. Launch Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 💻 What the Dashboard Does
1. **Surveillance Snapshot:** Displays current risk tier (**Low, Medium, High, Critical**), expected dengue cases, and outbreak probability ($P(\text{cases} > 0)$) for any selected ward and observation week.
2. **Weather Scenario Simulation:** Real-time sliders for rainfall, average temperature, and relative humidity. Modifying any slider recomputes the feature pipeline and updates risk estimates immediately.
3. **Longitudinal Trends:** Dual-axis interactive time series chart showing 10 years of weekly rainfall vs. recorded dengue cases with vertical week markers.
4. **City Ward Cross-Section Comparison:** Horizontal bar charts comparing expected caseload across all wards within the selected municipality for that specific week.
5. **Driver Attribution:** Displays preceding 1- to 4-week rainfall lags ($t-1, t-2, t-3, t-4$) and 4-week cumulative rainfall volume.
6. **Empirical Validation Telemetry:** Live tabbed viewer of out-of-sample metrics from `reports/validation.json` comparing model performance to the All-Zero baseline.

---

## ⚠️ Model Limitations
- **Risk Nowcast vs. Long-Range Forecast:** The model estimates the risk of the current week using weather observed through that week and prior weeks ($t-1$ through $t-4$). It does not perform unsupported multi-month forecasts.
- **Microclimate Station Coverage:** Environmental indices are aggregated at the ward level; micro-focal stagnant water pools within individual households are unobserved.
- **Initial Observation Weeks:** The first 4 observation weeks of each ward lack a 4-week lag history and were safely truncated during training to prevent synthetic data imputation.

---

## 🤖 Full Disclosure of AI Tools Used
In strict compliance with the Hackathon Transparency Guidelines:
- **Coding Assistant:** Antigravity AI (powered by Google DeepMind Gemini models) and Codex.
- **AI Contributions:** Used to scaffold boilerplate code, assist with unit test cases, generate initial Streamlit chart layouts, and draft markdown documentation.
- **Human Review:** All mathematical formulations, feature lag boundaries, zero-inflation handling logic, and physical data bounds were verified, tested, and validated by team members.
- **External APIs:** Zero external cloud APIs or paid LLM inference endpoints are used at runtime. The entire application runs 100% offline on localhost.

---

## 🎬 3-Minute Video Demonstration Script (for Ibrahim)
*Target duration: 2:30 - 2:45 continuous uncut screen recording with voiceover.*

1. **[0:00 - 0:30] Introduction & Problem:**
   - *"Welcome to Urban Dengue Risk Intelligence (Operation-Mosha). In Bangladesh, dengue transmission is heavily seasonal and zero-inflated—over 58% of ward-weeks record zero cases. Standard models predict zero everywhere and miss outbreaks."*
2. **[0:30 - 1:00] Solution & Architecture:**
   - *"We implemented a two-stage hurdle model in scikit-learn: Stage 1 predicts outbreak probability, and Stage 2 estimates conditional severity. Features are strictly isolated by ward with 1- to 4-week rainfall lags to prevent data leakage."*
3. **[1:00 - 1:45] Live Dashboard Walkthrough:**
   - Select Dhaka South, Ward 01, peak monsoon week.
   - Show Risk Tier badge (High/Critical) and expected caseload.
   - Move rainfall slider from 20 mm to 150 mm: show expected cases and probability increase dynamically.
4. **[1:45 - 2:15] Comparative & Driver Analytics:**
   - Show the longitudinal trend chart with preceding rainfall peaks.
   - Highlight the 1-4 week rainfall lag metrics in the driver attribution section.
   - Show the ward comparison chart ranking municipal risk across the city.
5. **[2:15 - 2:45] Validation & Reproducibility:**
   - Switch to the Validation Telemetry tab.
   - Highlight that while the All-Zero baseline gets 0% outbreak recall, our model achieves **84.1% test recall** and **0.882 F1** on held-out 2025 test data.
   - Conclude: *"All 17 tests pass, and the application runs completely self-contained on localhost without any external API dependencies. Thank you!"*
