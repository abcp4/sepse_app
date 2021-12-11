import streamlit as st
import shap
import joblib
import xgboost as xgb

def predict_data(
    age: int, 
    sector: str, 
    temperature: float, 
    respiratory_frequency: float, 
    systolic_blood_pressure: float, 
    diastolic_blood_pressure: float,
    mean_arterial_pressure: float, 
    oxygen_saturation: float,
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    f5: float,
    ):
    sector_encoded = label_encoder.transform([sector])
    features = [
        age,
        sector_encoded,
        temperature,
        respiratory_frequency,
        systolic_blood_pressure,
        diastolic_blood_pressure,
        mean_arterial_pressure,
        oxygen_saturation,
        f1,
        f2,
        f3,
        f4,
        f5,
    ]
    
    features = scaler.transform([features])
    print(features.shape)
    y_pred = model.predict(features)
    shap_values = explainer.shap_values(features)
    return y_pred, shap_values, explainer

### Carregar modelos do matplotlib
model = joblib.load("xgb_model.joblib")
label_encoder = joblib.load("le.joblib")
scaler = joblib.load("scaler.joblib")

## Explainer
explainer = shap.TreeExplainer(model)

from binning import binning
sample = [87, "UTIG", 36.0, 18.0, 128.0, 75.0, 93.0, 91.0]
extra_features=binning(sample)
sample.extend(list(extra_features.flatten()))
print(sample)
y_pred, shap_values, explainer = predict_data(*sample)
print(y_pred)
print(shap_values)

# shap.plots.force(explainer.expected_value, shap_values[0,:], sample)
shap.summary_plot(shap_values, sample)

