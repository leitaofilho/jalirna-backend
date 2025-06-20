# 🧬 jaliRNA Backend - Docker Production

Sistema de predição DRC com PyTorch Neural Networks em container Docker otimizado.

## 🏗️ Arquitetura

- **Framework**: Flask + PyTorch
- **Containerização**: Docker multi-stage
- **Modelo**: Multi-task Neural Network
- **Deploy**: Railway/Supabase/Heroku compatible

## 🚀 Quick Start

### Local Development
```bash
# Testar com Docker
./test_docker.sh

# Ou Docker Compose
docker-compose up --build
```

### Production Deploy

#### Railway (Recomendado)
1. Push para GitHub
2. Railway detecta Dockerfile automaticamente
3. Deploy automático com zero-config

#### Supabase (com Docker)
1. Supabase > Functions > Docker deployment
2. Conectar GitHub repo
3. Build command: `docker build .`

## 📋 Endpoints

- `GET /api/health` - Health check
- `GET /api/model-info` - Model metadata  
- `POST /api/predict` - DRC prediction

## 🔧 Environment Variables

```bash
FLASK_ENV=production
PORT=8000
PYTHONPATH=/app
```

## 🧪 Testing

```bash
# Local test
curl http://localhost:5001/api/health

# Prediction test
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"idade": 45, "sexo": "MASCULINO", "cor2": 0, "imc": 25.5, "cc": 85, "rcq": 0.9, "pas": 130, "pad": 80, "fuma": false, "realizaExercicio": true, "bebe": false, "dm": false, "has": true}'
```

## 📦 Docker Features

- ✅ Multi-stage build otimizado
- ✅ Non-root user security
- ✅ Health checks integrados
- ✅ Dependências fixas
- ✅ Production-ready gunicorn
- ✅ Cache layers otimizados

## 🎯 Production URLs

Após deploy:
- **Railway**: `https://jalirna-backend-production.up.railway.app`
- **Supabase**: `https://project.supabase.co/functions/v1/jalirna`

## 📊 Performance

- **Cold start**: ~30s (PyTorch loading)
- **Response time**: <200ms
- **Memory**: ~512MB
- **CPU**: 1 core sufficient