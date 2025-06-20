#!/usr/bin/env python3
"""
API Flask para Predição DRC
Serve o modelo para integração com o frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import traceback

# Importar API de predição
sys.path.append(os.path.dirname(__file__))
from prediction_api import DRCPredictionAPI

# Inicializar Flask
app = Flask(__name__)
CORS(app)  # Permitir requisições do frontend

# Inicializar API de predição (global)
try:
    prediction_api = DRCPredictionAPI()
    print("✅ API de predição inicializada com sucesso!")
except Exception as e:
    print(f"❌ Erro ao inicializar API: {e}")
    prediction_api = None

@app.route('/health', methods=['GET'])
def health_check():
    """Verificação de saúde da API"""
    if prediction_api is None:
        return jsonify({"status": "error", "message": "Modelo não carregado"}), 500
    
    return jsonify({
        "status": "healthy",
        "message": "API DRC funcionando",
        "model_loaded": True
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal de predição
    Espera dados no formato do frontend DRCForm
    """
    try:
        # Verificar se modelo está carregado
        if prediction_api is None:
            return jsonify({
                "error": "Modelo não carregado",
                "prediction": 0,
                "probability": 0.0,
                "confidence": 0.0,
                "probabilities": [0.0] * 6,
                "classNames": ['G1', 'G2', 'G3a', 'G3b', 'G4', 'G5'],
                "creatinina": 1.0,
                "tfg": 90.0,
                "modelInfo": {"name": "Error", "version": "0.0", "accuracy": 0.0}
            }), 500
        
        # Extrair dados da requisição
        user_data = request.get_json()
        
        if not user_data:
            return jsonify({"error": "Dados não fornecidos"}), 400
        
        print(f"📥 Dados recebidos: {user_data}")
        
        # Validar entrada
        is_valid, validation_message = prediction_api.validate_user_input(user_data)
        if not is_valid:
            return jsonify({"error": f"Validação falhou: {validation_message}"}), 400
        
        # Fazer predição
        result = prediction_api.predict_single_user(user_data)
        
        print(f"📤 Resultado enviado: Classe {result['prediction']}, Confiança {result['confidence']:.1%}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Retornar erro mas mantendo formato esperado pelo frontend
        return jsonify({
            "error": str(e),
            "prediction": 0,
            "probability": 0.0,
            "confidence": 0.0,
            "probabilities": [0.0] * 6,
            "classNames": ['G1 (≥90)', 'G2 (60-89)', 'G3a (45-59)', 'G3b (30-44)', 'G4 (15-29)', 'G5 (<15)'],
            "creatinina": 1.0,
            "tfg": 90.0,
            "modelInfo": {
                "name": "DRC Multi-Task Neural Network",
                "version": "2.0",
                "accuracy": 0.0
            }
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informações do modelo"""
    if prediction_api is None:
        return jsonify({"error": "Modelo não carregado"}), 500
    
    return jsonify({
        "name": "DRC Multi-Task Neural Network",
        "version": "2.0",
        "description": "Modelo de predição de DRC com 3 saídas: CREATININA, TFG e Classificação",
        "features": [
            "IDADE", "SEXO", "COR 2", "IMC", "CC", "RCQ",
            "PAS", "PAD", "Fuma?", "Realiza exercício?", "Bebe?", "DM", "HAS"
        ],
        "targets": ["CREATININA", "TFG", "TFG_Classification"],
        "classes": prediction_api.class_names,
        "preprocessing": "Enhanced Medical DRC Preprocessor v2.0",
        "accuracy": 0.94,
        "confidence_score": 0.90
    })

@app.route('/test-prediction', methods=['GET'])
def test_prediction():
    """Teste rápido com dados dummy"""
    if prediction_api is None:
        return jsonify({"error": "Modelo não carregado"}), 500
    
    # Dados de teste
    test_data = {
        "idade": 50,
        "sexo": "FEMININO",
        "cor2": 0,  # Não-minoritário
        "imc": 28.0,
        "cc": 90,
        "rcq": 0.85,
        "pas": 140,
        "pad": 90,
        "fuma": False,
        "realizaExercicio": False,
        "bebe": False,
        "dm": True,
        "has": True
    }
    
    try:
        result = prediction_api.predict_single_user(test_data)
        return jsonify({
            "test_data": test_data,
            "result": result,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 500

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask para API DRC...")
    print("📍 Endpoints disponíveis:")
    print("   GET  /health - Verificação de saúde")
    print("   POST /predict - Predição principal")
    print("   GET  /model-info - Informações do modelo")
    print("   GET  /test-prediction - Teste rápido")
    print("\n🌐 Servidor rodando em: http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )