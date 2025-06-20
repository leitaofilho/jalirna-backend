#!/usr/bin/env python3
"""
Teste final antes do deploy - Verifica se tudo está funcionando
"""

import requests
import json
import time

def test_local_api():
    """Testa API local antes do deploy"""
    base_url = "http://localhost:5001"
    
    print("🧪 Testando API local antes do deploy...")
    
    # Teste 1: Health Check
    print("\n1. Health Check...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Status: {health.get('status')}")
            print(f"   ✅ Modelo: {health.get('model_loaded')}")
            print(f"   ✅ Preprocessador: {health.get('preprocessor_loaded')}")
        else:
            print(f"   ❌ Falha: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False
    
    # Teste 2: Model Info
    print("\n2. Model Info...")
    try:
        response = requests.get(f"{base_url}/api/model-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print(f"   ✅ Nome: {info.get('name')}")
            print(f"   ✅ Versão: {info.get('version')}")
            print(f"   ✅ Acurácia: {info.get('accuracy')}")
        else:
            print(f"   ❌ Falha: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Erro: {e}")
    
    # Teste 3: Predição Real
    print("\n3. Predição Teste...")
    test_data = {
        "idade": 45,
        "sexo": "MASCULINO",
        "cor2": 0,
        "imc": 25.5,
        "cc": 85,
        "rcq": 0.9,
        "pas": 130,
        "pad": 80,
        "fuma": False,
        "realizaExercicio": True,
        "bebe": False,
        "dm": False,
        "has": True
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Predição: {result.get('prediction')}")
            print(f"   ✅ Classe: {result.get('classNames', [])[result.get('prediction', 0)]}")
            print(f"   ✅ Confiança: {result.get('confidence', 0):.3f}")
            print(f"   ✅ Creatinina: {result.get('creatinina', 0):.3f}")
            print(f"   ✅ TFG: {result.get('tfg', 0):.1f}")
            print(f"   ✅ Modelo: {result.get('modelInfo', {}).get('name', 'N/A')}")
        else:
            print(f"   ❌ Falha: {response.status_code}")
            print(f"   ❌ Erro: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return False
    
    print("\n🎉 Todos os testes passaram! API pronta para deploy!")
    return True

def test_remote_api(base_url):
    """Testa API remota após deploy"""
    print(f"\n🌐 Testando API remota: {base_url}")
    
    # Health check remoto
    try:
        response = requests.get(f"{base_url}/api/health", timeout=30)
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ API remota funcionando!")
            print(f"   ✅ Status: {health.get('status')}")
            return True
        else:
            print(f"   ❌ API remota falhou: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Erro na API remota: {e}")
        return False

if __name__ == "__main__":
    print("🚀 jaliRNA - Teste de Deploy")
    print("=" * 50)
    
    # Testar local primeiro
    if test_local_api():
        print("\n📋 Próximos passos:")
        print("1. Criar repositório no GitHub")
        print("2. Fazer push do código")
        print("3. Configurar deploy no Supabase")
        print("4. Testar API remota")
        print("5. Configurar frontend Vercel")
        
        # Exemplo de teste remoto (descomente após deploy)
        # remote_url = "https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api"
        # test_remote_api(remote_url)
    else:
        print("\n❌ Corrija os problemas locais antes do deploy!")