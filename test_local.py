#!/usr/bin/env python3
"""
Teste local do backend jaliRNA
Verifica se o backend está funcionando corretamente
"""

import requests
import json

# Configuração
BASE_URL = 'http://localhost:5000'

def test_health():
    """Testa endpoint de health"""
    print("🔍 Testando health endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/api/health')
        if response.status_code == 200:
            print("✅ Health check: OK")
            print(f"   Resposta: {response.json()}")
        else:
            print(f"❌ Health check falhou: {response.status_code}")
            print(f"   Resposta: {response.text}")
    except Exception as e:
        print(f"❌ Erro no health check: {e}")

def test_model_info():
    """Testa endpoint de model info"""
    print("\n🔍 Testando model-info endpoint...")
    try:
        response = requests.get(f'{BASE_URL}/api/model-info')
        if response.status_code == 200:
            print("✅ Model info: OK")
            info = response.json()
            print(f"   Nome: {info.get('name', 'N/A')}")
            print(f"   Versão: {info.get('version', 'N/A')}")
        else:
            print(f"❌ Model info falhou: {response.status_code}")
            print(f"   Resposta: {response.text}")
    except Exception as e:
        print(f"❌ Erro no model info: {e}")

def test_prediction():
    """Testa endpoint de predição"""
    print("\n🔍 Testando prediction endpoint...")
    
    # Dados de teste
    test_data = {
        "IDADE": 45,
        "SEXO": "MASCULINO",
        "COR 2": 0,
        "IMC": 25.5,
        "CC": 85,
        "RCQ": 0.9,
        "PAS": 130,
        "PAD": 80,
        "Fuma?": "NÃO",
        "Realiza exercício?": "SIM",
        "Bebe?": "NÃO",
        "DM": "NÃO",
        "HAS": "SIM"
    }
    
    try:
        response = requests.post(
            f'{BASE_URL}/api/predict',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            print("✅ Predição: OK")
            result = response.json()
            print(f"   Predição: {result.get('prediction', 'N/A')}")
            print(f"   Probabilidade: {result.get('probability', 'N/A'):.3f}")
            print(f"   Creatinina: {result.get('creatinina', 'N/A'):.3f}")
            print(f"   TFG: {result.get('tfg', 'N/A'):.3f}")
            
            # Verificar se tem classe predita
            if 'classNames' in result and 'prediction' in result:
                class_name = result['classNames'][result['prediction']]
                print(f"   Classe: {class_name}")
        else:
            print(f"❌ Predição falhou: {response.status_code}")
            print(f"   Resposta: {response.text}")
    except Exception as e:
        print(f"❌ Erro na predição: {e}")

def main():
    """Executa todos os testes"""
    print("🚀 Testando jaliRNA Backend Local\n")
    
    test_health()
    test_model_info()
    test_prediction()
    
    print("\n✨ Testes concluídos!")

if __name__ == '__main__':
    main()