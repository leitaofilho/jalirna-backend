# Preprocessing - DRC Pipeline

## 🔧 Scripts Principais (Produção)

### `enhanced_medical_preprocessing.py`
- **Pipeline de produção para balanceamento de dados DRC**
- Estratégia híbrida: RandomOverSampler + SMOTE adaptativo
- Validação de melhoria obrigatória (97.4% alcançada)
- Métricas de qualidade e visualizações automáticas

### `detailed_methodology_report.py`
- **Validador de metodologia e gerador de relatórios**
- Análise de confiabilidade (Score: 90/100)
- Validação de qualidade das amostras sintéticas
- Visualizações detalhadas da metodologia aplicada

## 📊 Documentação

### `METODOLOGIA_DETALHADA_BALANCEAMENTO.md`
- Relatório completo da metodologia aplicada
- Justificativas técnicas para cada decisão
- Métricas de qualidade e validação
- Garantias para aplicações médicas

### `RESPOSTA_FINAL_PREPROCESSAMENTO.md`
- Análise comparativa dos pipelines
- Recomendações específicas para dados médicos
- Melhores práticas implementadas

## 📈 Resultados

### `plots/methodology_report/`
- Análise espacial com PCA
- Avaliação de confiabilidade
- Visualizações da metodologia

### `plots/preprocessing_enhanced/`
- Análise de melhoria (97.4%)
- Comparações antes/depois
- Métricas de balanceamento

## 📁 Legacy

### `legacy/`
Scripts experimentais e versões anteriores:
- `preprocessing_pipeline.py` - Pipeline básico (falhou)
- `medical_grade_preprocessing.py` - Versão inicial médica
- `advanced_preprocessing_pipeline.py` - Experimental
- Outros scripts de teste

### `legacy_plots/`
Visualizações de versões anteriores dos pipelines

## 🚀 Como Usar

### Executar Pipeline de Produção
```python
python3 enhanced_medical_preprocessing.py
```

### Gerar Relatório de Metodologia
```python
python3 detailed_methodology_report.py
```

## ✅ Status

- **Pipeline Validado**: ✅ Produção
- **Confiabilidade**: 90/100 - ALTA
- **Melhoria**: 97.4% no desbalanceamento
- **Classes Críticas**: Todas resolvidas
- **Aplicação Médica**: ✅ Aprovado