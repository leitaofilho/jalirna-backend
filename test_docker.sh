#!/bin/bash
# jaliRNA Docker Test Script

set -e

echo "🐳 jaliRNA Docker Test"
echo "====================="

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Build da imagem
echo -e "${BLUE}📦 Building Docker image...${NC}"
docker build -t jalirna-backend .

# Verificar se build foi bem-sucedido
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker build successful!${NC}"
else
    echo -e "${RED}❌ Docker build failed!${NC}"
    exit 1
fi

# Executar container
echo -e "${BLUE}🚀 Starting container...${NC}"
docker run -d --name jalirna-test -p 5001:8000 jalirna-backend

# Aguardar inicialização
echo -e "${YELLOW}⏳ Waiting for API to start...${NC}"
sleep 30

# Testar health check
echo -e "${BLUE}🔍 Testing health endpoint...${NC}"
response=$(curl -s http://localhost:5001/api/health)

if echo "$response" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Health check passed!${NC}"
    echo "$response" | python3 -m json.tool
else
    echo -e "${RED}❌ Health check failed!${NC}"
    echo "Response: $response"
fi

# Testar predição
echo -e "${BLUE}🧪 Testing prediction...${NC}"
prediction=$(curl -s -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 45,
    "sexo": "MASCULINO",
    "cor2": 0,
    "imc": 25.5,
    "cc": 85,
    "rcq": 0.9,
    "pas": 130,
    "pad": 80,
    "fuma": false,
    "realizaExercicio": true,
    "bebe": false,
    "dm": false,
    "has": true
  }')

if echo "$prediction" | grep -q "prediction"; then
    echo -e "${GREEN}✅ Prediction test passed!${NC}"
    echo "$prediction" | python3 -m json.tool
else
    echo -e "${RED}❌ Prediction test failed!${NC}"
    echo "Response: $prediction"
fi

# Logs do container
echo -e "${BLUE}📋 Container logs:${NC}"
docker logs jalirna-test

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up...${NC}"
docker stop jalirna-test
docker rm jalirna-test

echo -e "${GREEN}🎉 Docker test completed!${NC}"