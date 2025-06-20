# jaliRNA Backend - SOLUÇÃO DEFINITIVA RAILWAY
FROM python:3.11-slim

# Metadados
LABEL maintainer="jaliRNA Team"
LABEL description="jaliRNA DRC Prediction API - DEFINITIVO"
LABEL version="2.0-final"

# Variáveis de ambiente otimizadas
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PORT=8000 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependências do sistema essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Diretório de trabalho
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# INSTALAÇÃO FORÇADA - SOLUÇÃO DEFINITIVA
RUN pip install --upgrade pip==23.3.1 && \
    pip install --force-reinstall --no-deps numpy==1.26.2 && \
    pip install --force-reinstall --no-deps torch==2.3.1 && \
    pip install -r requirements.txt && \
    pip check

# Verificar instalação
RUN python -c "import numpy; import torch; print('✅ NumPy:', numpy.__version__); print('✅ PyTorch:', torch.__version__)"

# Copiar aplicação
COPY . .

# Expor porta
EXPOSE $PORT

# Health check robusto
HEALTHCHECK --interval=45s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Comando final otimizado
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "180", "--preload", "app:app"]