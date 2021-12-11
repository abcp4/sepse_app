# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of showing geographic data."""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import streamlit.components.v1 as components
import streamlit as st
import shap
import joblib
import xgboost as xgb
from binning import binning
import matplotlib.pyplot as plt

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

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
    return y_pred, shap_values


# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# st.title('App para predição de deterioração clínica do paciente')

st.header("App para predição de deterioração clínica do paciente")
st.markdown("""Essa demonstração faz uso do streamlit, xgboost e shap para explicabilidade do modelo.
""")

dataset_name = st.selectbox('Selecione o modelo', ['moons', 'blobs', 'circles', 'linear']) # select your dataset

st.sidebar.title("Coloque as informações do paciente") # seleect your dataset params
setor  = st.sidebar.selectbox('Setor', [
                            "UTIG",
                            "1AP2",
                            "4AP2",
                            "UTIC",
                            "UTIP",
                            "3AP1",
                            "3AP2",
                            "4AP1",
                            "1AP1",
                            "2AP2",
                            "UIP",
                            "3AP3",
                            "1AP2 - 126",
                            "2AP1",
                            "3AP3 - EPI",
                            "SEMI-CO",
                            ]) # data points

idade  = st.sidebar.number_input('Idade',value=87,step=1)
temp  = st.sidebar.number_input('Temperatura',value=36.)
resp  = st.sidebar.number_input('Frequência respiratória', value=18.)
sisto  = st.sidebar.number_input('Pressão Sistólica', value=128.)
diasto  = st.sidebar.number_input('Pressão Diastólica', value=75.)
media  = st.sidebar.number_input('Pressão Média', value=93.)
o2  = st.sidebar.number_input('Saturação O2', value=91.)
submit = st.sidebar.button('Fazer Predição')

### Carregar modelos do matplotlib
model = joblib.load("xgb_model.joblib")
label_encoder = joblib.load("le.joblib")
scaler = joblib.load("scaler.joblib")
## Explainer
explainer = shap.TreeExplainer(model)
labels=['Melhorado','Obito']
column_names=[  'Idade','Setor','Temperatura','Frequência respiratória',
                'Pressão Sistólica','Pressão Diastólica',
                'Pressão Média','Saturação O2',
                'Bin_Temperatura','Bin_Respiração',
                'Bin_Sistólica','Bin_Média','Bin_o2'
            ]

if submit:
    sample=[idade,setor,temp,resp,sisto,diasto,media,o2]
    # sample = [87, "UTIG", 36.0, 18.0, 128.0, 75.0, 93.0, 91.0]
    extra_features=binning(sample,column_names)
    sample.extend(list(extra_features.flatten()))
    print(sample)
    y_pred, shap_values = predict_data(*sample)
    print(y_pred)
    print(shap_values)
    
    # st.write("Predição: ", labels[int(y_pred[0])])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sample={column_names[k]:[sample[k]] for k in range(len(sample))}
    sample=pd.DataFrame.from_dict(sample)
    p =shap.force_plot(explainer.expected_value, shap_values[0,:], sample)
    # shap.summary_plot(shap_values, sample)
    # st.pyplot(fig)
    st.subheader(f'Predição: {labels[int(y_pred[0])]}')
    st_shap(p)
