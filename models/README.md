# Model Artifacts

This directory contains the trained model and supporting artifacts required by the Flask API (`src/app.py`) for real-time dropout risk scoring.

## Files

| File | Description |
|------|-------------|
| `early_warning_rf_balanced.pkl` | Trained Tuned Balanced Random Forest classifier (deployment version) |
| `model_features.pkl` | Ordered list of feature column names expected by the model at inference time |
| `scaler.pkl` | Fitted StandardScaler (retained from the logistic regression pipeline for completeness; not required by the Random Forest) |

## Model Details

- **Algorithm:** Random Forest Classifier with `class_weight="balanced"`
- **Tuned hyperparameters:** `n_estimators=100`, `max_depth=10`, `min_samples_split=5`, `min_samples_leaf=4`, `max_features=None`
- **Training data:** 26,038 records (80% stratified train/test split)
- **Feature count:** 36 features after one-hot encoding
- **Prediction window:** Day 30 (early-warning constraint — only features observable within the first 30 days of enrollment)
- **Target:** Binary classification (Withdrawn = 1, all other outcomes = 0)

## Deployment Note

The final deployment model uses `n_estimators=100` rather than the originally tuned `n_estimators=200`. Halving the ensemble size reduces the serialized artifact from roughly 50 MB to a size compatible with standard GitHub file limits and produces a leaner model for API deployment, with negligible impact on predictive performance (F1 change under 1%). This tradeoff is documented in the final report as a practical production consideration.

## Regenerating the Artifacts

To regenerate these files from scratch, run the full `notebooks/Final_Modelling_V1.ipynb` notebook end to end. The deployment section at the bottom of the notebook retrains the model with the reduced ensemble size and saves all three artifacts to this directory.

## Loading the Model

```python
import joblib

model = joblib.load("models/early_warning_rf_balanced.pkl")
feature_names = joblib.load("models/model_features.pkl")

# Example prediction
import numpy as np
X_new = np.array([[...]])  # shape: (1, 36), columns in order of feature_names
risk_probability = model.predict_proba(X_new)[:, 1]
```
