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
import altair as alt
import pydeck as pdk
import streamlit.components.v1 as components
import streamlit as st
import shap
import requests
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, validator
from binning import binning
import numpy as np
import xgboost

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def predict_data(age: int,
                sector: str,
                temperature: float,
                respiratory_frequency: float,
                systolic_blood_pressure: float,
                diastolic_blood_pressure: float,
                mean_arterial_pressure: float,
                oxygen_saturation: float
                ):
    """Informe os dados do paciente que deseja obter uma predição."""
    if(sector != 'Nenhum'):
        sector_encoded = label_encoder.transform([sector])
        features = [
            age,
            sector_encoded.item(),
            temperature,
            respiratory_frequency,
            systolic_blood_pressure,
            diastolic_blood_pressure,
            mean_arterial_pressure,
            oxygen_saturation,
        ]
    else:
        features = [
            age,
            0,
            temperature,
            respiratory_frequency,
            systolic_blood_pressure,
            diastolic_blood_pressure,
            mean_arterial_pressure,
            oxygen_saturation,
        ]

    extra_features=binning(features)
    features.extend(list(extra_features.flatten()))
    
    features_np=np.asarray(features,dtype=np.float64)
    if(sector != 'Nenhum'):
        features_np=scaler.transform([features_np])
        y_pred = model.predict(features_np)
        return {"predict": y_pred[0],'features': features}
    else:
        print(features_np)
        features_np=np.delete(features_np, 1)
        print(features_np)
        features_np=nosection_scaler.transform([features_np])
        y_pred = nosection_model.predict(features_np)
        features=[features[0]]+features[2:]
        print(features)
        return {"predict": y_pred[0],'features': features}
    
# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")

# st.title('App para predição de deterioração clínica do paciente')

st.header("App para predição de deterioração clínica do paciente")
st.markdown("""Essa demonstração faz uso do streamlit, xgboost e shap para explicabilidade do modelo.
""")

modelo_name = st.selectbox('Selecione o modelo', ['Modelo com setor', 'Modelo sem setor']) # select your dataset

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

### Carregar modelos d
model = joblib.load("models/xgb_model.joblib")
label_encoder = joblib.load("models/le.joblib")
scaler = joblib.load("models/scaler.joblib")

nosection_model = joblib.load("models/nosection_xgb_model.joblib")
nosection_scaler = joblib.load("models/nosection_scaler.joblib")

## Explainer
explainer = shap.TreeExplainer(model)
nosection_explainer = shap.TreeExplainer(nosection_model)
labels=['Melhorado','Obito']
column_names=[  'Idade','Setor','Temperatura','Frequência respiratória',
                'Pressão Sistólica','Pressão Diastólica',
                'Pressão Média','Saturação O2',
                'Bin_Temperatura','Bin_Respiração',
                'Bin_Sistólica','Bin_Média','Bin_o2'
            ]
column_names_nosector= [column_names[0]]+column_names[2:]



if submit:    
    if modelo_name == 'Modelo sem setor':
        setor = 'Nenhum'
    data=predict_data(idade,setor,temp,resp,sisto,diasto,media,o2)
    y_pred=data['predict']
    features=data['features']
    if modelo_name != 'Modelo sem setor':
        shap_values = explainer.shap_values(features)
        features={column_names[k]:[features[k]] for k in range(len(features))}
        features=pd.DataFrame.from_dict(features)
    else:
        shap_values = nosection_explainer.shap_values(features)
        features={column_names_nosector[k]:[features[k]] for k in range(len(features))}
        features=pd.DataFrame.from_dict(features)
    
    # st.write("Predição: ", labels[int(y_pred[0])])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    p =shap.force_plot(explainer.expected_value, shap_values[0,:], features)
    # shap.summary_plot(shap_values, sample)
    # st.pyplot(fig)
    st.subheader(f'Predição: {labels[int(y_pred)]}')
    st_shap(p)
