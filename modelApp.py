from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Carrega modelo e scaler
model = joblib.load('models/meu_modelo_de_arritmia.pkl')
scaler = joblib.load('models/escalador.pkl')

@app.route('/')
def home():
    return jsonify({'mensagem': 'API de predição de arritmia OK'})

@app.route('/prever', methods=['POST'])
def prever():
    try:
        # Dados JSON do corpo da requisição
        dados = request.json
        if not dados:
            return jsonify({'erro': 'Nenhum dado recebido'}), 400

        # Converte para DataFrame
        df = pd.DataFrame([dados])

        # Aplica escalonamento
        dados_escalonados = scaler.transform(df)

        # Faz predição
        predicao = model.predict(dados_escalonados)

        # Retorna resultado
        return jsonify({'predicao': int(predicao[0])})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
