# jaliRNA Backend - Supabase Deployment

Backend Flask independente para deploy no Supabase, separado do frontend Vercel.

## Arquitetura

- **Frontend**: Next.js no Vercel (apenas interface)
- **Backend**: Flask no Supabase (API + ML)
- **Comunicação**: HTTP REST API

## Setup Local

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar servidor
python app.py

# Testar endpoints
python test_local.py
```

## Endpoints

- `GET /api/health` - Health check
- `GET /api/model-info` - Informações do modelo
- `POST /api/predict` - Predição DRC

## Deploy Supabase

1. Fazer upload dos arquivos para o Supabase
2. Configurar variáveis de ambiente:
   ```
   FLASK_ENV=production
   PORT=8000
   ```
3. O Supabase executará automaticamente via Procfile

## Configuração Frontend

No frontend Vercel, definir a variável de ambiente:
```
NEXT_PUBLIC_DRC_API_URL=https://sua-url-supabase.com
```

## Vantagens desta Abordagem

- ✅ Backend completo com PyTorch no Supabase
- ✅ Frontend leve e rápido no Vercel
- ✅ Sem limitações de build time
- ✅ Escalabilidade independente
- ✅ Custos otimizados