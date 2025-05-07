from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # Para salvar e carregar o modelo

# Inicializando o Flask
app = Flask(__name__)

# Carregando o modelo treinado
model = joblib.load('models/meu_modelo_de_arritmia.pkl') 

# Escalonador para as entradas
scaler = joblib.load('models/scaler.pkl')  
