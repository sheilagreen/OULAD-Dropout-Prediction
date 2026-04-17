# Predicting Student Dropout in Online Courses Using Early-Warning Learning Analytics

**DATA 6545: Data Science and MLOps — Final Project**
**Fairfield University | Dolan School of Business | Spring 2026**
**Gabriela, Sheila and Sanjog**

---

## Overview

Online education platforms face persistently high dropout rates, reducing course completion, learner satisfaction, and long-term revenue. This project develops an **early-warning predictive model** that estimates the probability a student will withdraw from an online course using only information available within the **first 30 days** of enrollment.

The model is trained on the [Open University Learning Analytics Dataset (OULAD)](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad), which contains **32,593 student-course enrollment records** spanning demographics, registration timing, VLE interaction logs, and assessment submissions.

## Key Results

| Model | Accuracy | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Baseline Logistic Regression | .735 | .630 | .566 | .596 | .775 |
| **Tuned Balanced Random Forest** | **.776** | **.609** | **.649** | **.628** | **.802** |
| Tuned Gradient Boosting | .788 | .510 | .727 | .600 | .789 |

The **Tuned Balanced Random Forest** is the recommended model, achieving the best balance of recall and precision (highest F1 of .628) with strong overall discrimination (AUC = .802).

At an operating threshold of **0.30**, the model captures ~66.6% of actual withdrawals while flagging a manageable number of students for outreach.

## Repository Structure

```
OULAD-Dropout-Prediction/
├── data/                          # Dataset files (CSV)
│   └── README.md                  # Data download instructions
├── notebooks/
│   └── Final_Modelling_V1.ipynb   # Main modeling notebook (EDA → MLflow → SHAP)
├── src/
│   └── app.py                     # Flask API for real-time scoring
├── models/
│   └── README.md                  # Model artifact descriptions
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container for API deployment
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sheilagreen/OULAD-Dropout-Prediction.git
cd OULAD-Dropout-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the OULAD dataset from [Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) and place the CSV files in the `data/` directory:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`
- `studentAssessment.csv`

### 5. Run the Notebook

Open and run `notebooks/Final_Modelling_V1.ipynb` in Jupyter or Google Colab. The notebook covers the full pipeline:

1. Data loading and integration
2. Data cleaning and target creation
3. Feature engineering (Day 30 early-warning window)
4. Exploratory data analysis
5. Model training and comparison (6 configurations)
6. MLflow experiment tracking
7. Hyperparameter tuning
8. Threshold analysis and intervention simulation
9. Rolling window analysis
10. SHAP model explainability

### 6. Launch the Flask API (Optional)

```bash
cd src
python app.py
```

The API expects all 36 model features as JSON input (including pre-encoded one-hot categorical variables). Use the following example to test a single prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "studied_credits": 60,
    "date_registration": -5,
    "num_of_prev_attempts": 0,
    "total_clicks": 80,
    "active_days": 5,
    "avg_clicks_day": 16.0,
    "last_activity": 20,
    "avg_score": 30.0,
    "num_submitted": 1,
    "num_unsubmitted": 4,
    "highest_education_HE Qualification": 0,
    "highest_education_Lower Than A Level": 1,
    "highest_education_No Formal quals": 0,
    "highest_education_Post Graduate Qualification": 0,
    "region_East Midlands Region": 0,
    "region_Ireland": 0,
    "region_London Region": 1,
    "region_North Region": 0,
    "region_North Western Region": 0,
    "region_Scotland": 0,
    "region_South East Region": 0,
    "region_South Region": 0,
    "region_South West Region": 0,
    "region_Wales": 0,
    "region_West Midlands Region": 0,
    "region_Yorkshire Region": 0,
    "imd_band_10-20": 0,
    "imd_band_20-30%": 1,
    "imd_band_30-40%": 0,
    "imd_band_40-50%": 0,
    "imd_band_50-60%": 0,
    "imd_band_60-70%": 0,
    "imd_band_70-80%": 0,
    "imd_band_80-90%": 0,
    "imd_band_90-100%": 0,
    "disability_Y": 0
  }'
```

Expected response:

```json
{
  "dropout_probability": 0.72,
  "risk_tier": "High",
  "threshold_used": 0.3
}
```

**Note:** The current API expects features in their pre-encoded form. A production version would wrap preprocessing and the model in a single `sklearn.pipeline.Pipeline` so that callers can submit raw categorical values (e.g., `"region": "London Region"`). This is noted as future work in the final report.

### 7. Docker Deployment (Optional)

```bash
docker build -t dropout-predictor .
docker run -p 5000:5000 dropout-predictor
```

## Methodology

- **Prediction window:** Day 30 (early-warning constraint)
- **Target:** Binary classification (Withdrawn = 1, all other outcomes = 0)
- **Feature set:** 36 features after one-hot encoding (enrollment, demographic, VLE engagement, assessment performance)
- **Models evaluated:** Logistic Regression, Random Forest, Gradient Boosting (default and tuned variants)
- **Experiment tracking:** MLflow with 6 logged runs
- **Explainability:** SHAP values for feature importance and directionality
- **Evaluation focus:** Recall for dropout class (minimizing missed at-risk students)

## MLflow Experiment Tracking

All model runs are logged under the MLflow experiment `OULAD-Dropout-Prediction`, including parameters, metrics, and tags for each experimental configuration.

### Accessing the Logs

Due to size, MLflow logs are distributed as a Release asset rather than committed to the repository. To view them:

1. Download `mlruns.zip` from the [latest release](https://github.com/sheilagreen/OULAD-Dropout-Prediction/releases/latest)
2. Unzip in the project root:

```bash
unzip mlruns.zip
```

3. Launch the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

4. Open `http://localhost:5000` in your browser to explore parameters, metrics, and run comparisons.

Alternatively, rerunning `notebooks/Final_Modelling_V1.ipynb` end to end will regenerate the logs in a fresh `mlruns/` directory.

## Tech Stack

- Python 3.10+
- pandas, NumPy, scikit-learn
- MLflow (experiment tracking)
- SHAP (model explainability)
- Flask (API deployment)
- Docker (containerization)
- Matplotlib, Seaborn, Plotly (visualization)

## Dataset Citation

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics Dataset. *Scientific Data*, 4, 170171.

## License

This project is for academic purposes as part of DATA 6545 at Fairfield University.

