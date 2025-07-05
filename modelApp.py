from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('models/meu_modelo_de_arritmia.pkl')
scaler = joblib.load('models/escalador.pkl')

@app.route('/')
def home():
    return jsonify({'mensagem': 'API de predição de arritmia OK'})

@app.route('/prever', methods=['POST'])
def prever():
    try:
        dados = request.json
        if not dados:
            return jsonify({'erro': 'Nenhum dado recebido'}), 400

        # Converte os dados recebidos para DataFrame
        df = pd.DataFrame([dados])

        # Aplica o escalonamento
        dados_escalonados = scaler.transform(df)

        # Faz predição
        predicao = model.predict(dados_escalonados)

        return jsonify({'predicao': str(predicao[0])})

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
