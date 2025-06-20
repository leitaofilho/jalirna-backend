# jaliRNA Backend - Docker Production Image
FROM python:3.12-slim

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
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd --create-home --shell /bin/bash jalirna
WORKDIR /app

# Copiar requirements primeiro (cache Docker)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Dar permissões corretas
RUN chown -R jalirna:jalirna /app
USER jalirna

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/health || exit 1

# Expor porta
EXPOSE $PORT

# Comando de produção
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "--preload", "app:app"]