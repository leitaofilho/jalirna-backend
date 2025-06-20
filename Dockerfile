# jaliRNA Backend - Railway Stable Docker
FROM python:3.10-slim

# Metadados
LABEL maintainer="jaliRNA Team"
LABEL description="jaliRNA DRC Prediction API - PyTorch Neural Network"
LABEL version="2.0"

# Variáveis de ambiente para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=8000

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Criar diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (otimização cache)
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copiar código da aplicação
COPY . .

# Expor porta
EXPOSE $PORT

# Health check otimizado
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Comando de produção otimizado para Railway
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "--preload", "--max-requests", "1000", "app:app"]