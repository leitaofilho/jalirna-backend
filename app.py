#!/usr/bin/env python3
"""
Flask API Server para DRC Prediction
Serve a API neural para integração com o frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Adicionar diretório atual ao path (arquivos estão no mesmo diretório)
sys.path.append(os.path.dirname(__file__))

from prediction_api import DRCPredictionAPI

app = Flask(__name__)

# Configurar CORS para produção - LIBERADO PARA DEBUG
CORS(app, origins="*", supports_credentials=False)

# Inicializar API original
try:
    drc_api = DRCPredictionAPI()
    print("✅ API DRC inicializada com sucesso!")
except Exception as e:
    print(f"❌ Erro ao inicializar API: {e}")
    drc_api = None

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Verifica saúde da API"""
    if request.method == 'OPTIONS':
        return '', 200
        
    print(f"🔍 Health check chamado de: {request.headers.get('Origin', 'desconhecido')}")
    
    if drc_api is None:
        return jsonify({'status': 'error', 'message': 'API não inicializada'}), 500
    
    return jsonify({
        'status': 'healthy',
        'message': 'API DRC funcionando',
        'model_loaded': drc_api.model is not None,
        'preprocessor_loaded': drc_api.preprocessor is not None,
        'cors_test': 'OK'
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Retorna informações do modelo"""
    if drc_api is None:
        return jsonify({'error': 'API não inicializada'}), 500
    
    return jsonify({
        'name': 'DRC Multi-Task Neural Network',
        'version': '2.0',
        'accuracy': 0.89,
        'architecture': 'Multi-Task Neural Network',
        'features': 13,
        'classes': 6,
        'classNames': drc_api.class_names,
        'description': 'Modelo neural para predição de DRC com qualidade médica'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint principal de predição"""
    if drc_api is None:
        return jsonify({'error': 'API não inicializada'}), 500
    
    try:
        # Obter dados da requisição
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Dados não fornecidos'}), 400
        
        print(f"📥 Recebendo requisição: {data}")
        
        # Validar entrada (método original)
        is_valid, validation_message = drc_api.validate_user_input(data)
        if not is_valid:
            print(f"❌ Validação falhou: {validation_message}")
            return jsonify({'error': validation_message}), 400
        
        # Fazer predição (método original)
        result = drc_api.predict_single_user(data)
        
        print(f"✅ Predição realizada: {result['classNames'][result['prediction']]}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Página inicial"""
    return jsonify({
        'message': 'DRC Prediction API',
        'version': '2.0',
        'endpoints': [
            'GET /api/health - Verificar saúde',
            'GET /api/model-info - Informações do modelo',
            'POST /api/predict - Fazer predição'
        ]
    })

if __name__ == '__main__':
    # Configuração para deploy (Supabase/Heroku/etc)
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("🚀 Iniciando jaliRNA Backend...")
    print(f"   Porta: {port}")
    print(f"   Debug: {debug}")
    print(f"📍 Health: /api/health")
    print(f"📊 Model Info: /api/model-info") 
    print(f"🎯 Predict: POST /api/predict")
    
    app.run(
        host='0.0.0.0',  # Permitir conexões externas
        port=port,
        debug=debug
    )