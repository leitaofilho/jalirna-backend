# 🚀 Deploy jaliRNA Backend no Supabase

## Pré-requisitos
- Conta no Supabase (https://supabase.com/)
- Git configurado
- Backend funcionando localmente (testado ✅)

## 📋 Passo a Passo

### 1. Preparar Repositório Git
```bash
cd backend_supabase
git init
git add .
git commit -m "Initial jaliRNA backend for Supabase"
```

### 2. Criar Repositório no GitHub
1. Acesse https://github.com/new
2. Nome: `jalirna-backend` (ou outro nome)
3. Descrição: `jaliRNA DRC Prediction Backend - PyTorch Neural API`
4. Público ou Privado (sua escolha)
5. **NÃO** inicializar com README, .gitignore ou license
6. Criar repositório

### 3. Conectar Local com GitHub
```bash
# Substituir pela sua URL do repositório
git remote add origin https://github.com/SEU_USUARIO/jalirna-backend.git
git branch -M main
git push -u origin main
```

### 4. Deploy no Supabase

#### A. Acessar Supabase
1. Acesse https://supabase.com/
2. Faça login ou crie conta
3. Clique em "New Project"

#### B. Configurar Projeto
1. **Nome**: `jalirna-backend`
2. **Database Password**: (escolha uma senha forte)
3. **Region**: Escolha a região mais próxima
4. **Plan**: Free tier (suficiente para início)
5. Clique "Create new project"

#### C. Configurar Edge Functions
1. No dashboard, vá em "Edge Functions" (menu lateral)
2. Clique "Create function"
3. **Nome**: `jalirna-api`
4. **Template**: Python (Flask)

#### D. Conectar GitHub
1. Em "Edge Functions", clique "Deploy from GitHub"
2. Autorizar Supabase no GitHub
3. Selecionar repositório: `jalirna-backend`
4. **Branch**: `main`
5. **Root directory**: `/` (raiz)
6. **Build command**: `pip install -r requirements.txt`
7. **Start command**: `python app.py`

### 5. Configurar Variáveis de Ambiente
No dashboard Supabase > Settings > Environment Variables:

```
FLASK_ENV=production
PORT=8000
PYTHONPATH=/app
```

### 6. Verificar Deploy
Após o deploy (pode levar alguns minutos):

1. **URL da API**: `https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api`
2. **Testar Health**: `GET https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api/api/health`
3. **Testar Predição**: `POST https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api/api/predict`

## 🧪 Testes Pós-Deploy

### Health Check
```bash
curl https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api/api/health
```

Deve retornar:
```json
{
  "status": "healthy",
  "message": "API DRC funcionando", 
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### Predição Teste
```bash
curl -X POST https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api/api/predict \
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
  }'
```

## 📝 URLs Importantes

Após deploy, anote estas URLs:

- **Base URL**: `https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api`
- **Health**: `/api/health`
- **Model Info**: `/api/model-info`  
- **Predict**: `/api/predict`

## ⚡ Próximo Passo: Vercel

Com o backend funcionando no Supabase, configurar no Vercel:

```bash
# No frontend Vercel, adicionar variável de ambiente:
NEXT_PUBLIC_DRC_API_URL=https://SEU_PROJETO.supabase.co/functions/v1/jalirna-api
```

## 🚨 Troubleshooting

### Build Falha
- Verificar `requirements.txt`
- Verificar logs no dashboard Supabase
- Confirmar que PyTorch está instalando corretamente

### CORS Issues
- Verificar se URL do frontend Vercel está nas origens permitidas
- Adicionar domínio específico se necessário

### Modelo Não Carrega
- Verificar se arquivos `model_production/` estão no repositório
- Confirmar que `preprocessing/` foi incluído

## 🎯 Status Esperado
✅ Backend rodando no Supabase
✅ API PyTorch funcionando 
✅ Endpoints respondendo
✅ CORS configurado
✅ Pronto para conectar com frontend Vercel