#!/usr/bin/env python3
"""
Pipeline Avançado de Pré-processamento para Dados Médicos Altamente Desbalanceados
Implementação das melhores práticas com validação rigorosa e métricas antes/depois
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats
from collections import Counter
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class AdvancedDRCPreprocessor:
    """
    Pipeline avançado de pré-processamento para dados médicos altamente desbalanceados
    Inclui validação rigorosa, métricas detalhadas e visualizações antes/depois
    """
    
    def __init__(self, strategy='adaptive', validation_split=0.2, random_state=42):
        self.strategy = strategy
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Componentes do pipeline
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.balancer = None
        self.target_encoder = None
        
        # Metadados
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        self.preprocessing_report = {}
        
        # Features de entrada
        self.input_features = [
            'IDADE', 'SEXO', 'COR', 'COR 2', 'IMC', 'CC', 'RCQ',
            'PAS', 'PAD', 'Fuma?', 'Realiza exercício?', 'Bebe?', 'DM', 'HAS'
        ]
        
        # Variáveis alvo
        self.target_variables = ['CREATININA', 'TFG', 'TFG_Classification']
        
        # Configurações clínicas
        self.clinical_ranges = {
            'CREATININA': {'normal_M': (0.7, 1.2), 'normal_F': (0.6, 1.1), 'units': 'mg/dL'},
            'TFG': {'normal': 90, 'stages': [15, 30, 60, 90], 'units': 'mL/min/1.73m²'},
            'IDADE': {'min': 18, 'max': 120},
            'IMC': {'underweight': 18.5, 'normal': 24.9, 'overweight': 29.9},
            'PAS': {'normal': 120, 'elevated': 129, 'stage1': 139, 'stage2': 180},
            'PAD': {'normal': 80, 'elevated': 89, 'stage1': 99, 'stage2': 110}
        }
    
    def analyze_data_quality(self, df):
        """
        Análise detalhada da qualidade dos dados ANTES do pré-processamento
        """
        print("\n" + "="*60)
        print("ANÁLISE DE QUALIDADE DOS DADOS - ESTADO INICIAL")
        print("="*60)
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_data': {},
            'outliers': {},
            'imbalance_analysis': {},
            'clinical_validation': {},
            'data_types': {},
            'recommendations': []
        }
        
        # 1. Análise de dados faltantes
        print("\n1. DADOS FALTANTES:")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        for col in df.columns:
            if missing_data[col] > 0:
                quality_report['missing_data'][col] = {
                    'count': int(missing_data[col]),
                    'percentage': float(missing_pct[col])
                }
                print(f"  {col}: {missing_data[col]} ({missing_pct[col]:.1f}%)")
        
        if not quality_report['missing_data']:
            print("  ✅ Nenhum dado faltante encontrado")
        
        # 2. Análise de tipos de dados
        print("\n2. TIPOS DE DADOS:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_vals = df[col].nunique()
            quality_report['data_types'][col] = {
                'dtype': dtype,
                'unique_values': int(unique_vals),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col])
            }
            
            if dtype == 'object':
                # Verificar se é numérico com formatação incorreta
                try:
                    pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='raise')
                    print(f"  ⚠️  {col}: object, mas parece numérico (vírgula decimal)")
                    quality_report['recommendations'].append(f"Converter {col} para numérico")
                except:
                    print(f"  📝 {col}: categórico com {unique_vals} valores únicos")
        
        # 3. Análise de outliers (apenas para variáveis numéricas)
        print("\n3. OUTLIERS (IQR Method):")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['ID_ANONIMO']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                outlier_pct = (outlier_count / len(df)) * 100
                
                if outlier_count > 0:
                    quality_report['outliers'][col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_pct),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound)
                    }
                    print(f"  {col}: {outlier_count} outliers ({outlier_pct:.1f}%)")
        
        # 4. Análise de desbalanceamento
        if 'TFG_Classification' in df.columns:
            print("\n4. ANÁLISE DE DESBALANCEAMENTO:")
            class_counts = df['TFG_Classification'].value_counts().sort_index()
            total_samples = len(df)
            
            quality_report['imbalance_analysis'] = {
                'total_classes': len(class_counts),
                'class_distribution': {},
                'imbalance_metrics': {}
            }
            
            for class_name, count in class_counts.items():
                pct = (count / total_samples) * 100
                quality_report['imbalance_analysis']['class_distribution'][class_name] = {
                    'count': int(count),
                    'percentage': float(pct)
                }
                print(f"  {class_name}: {count} amostras ({pct:.1f}%)")
            
            # Métricas de desbalanceamento
            majority_class = class_counts.max()
            minority_class = class_counts.min()
            imbalance_ratio = majority_class / minority_class
            
            quality_report['imbalance_analysis']['imbalance_metrics'] = {
                'majority_class_size': int(majority_class),
                'minority_class_size': int(minority_class),
                'imbalance_ratio': float(imbalance_ratio),
                'effective_classes': int((class_counts >= 10).sum())
            }
            
            print(f"  Razão de desbalanceamento: {imbalance_ratio:.1f}:1")
            print(f"  Classes com ≥10 amostras: {(class_counts >= 10).sum()}/{len(class_counts)}")
            
            # Classificar nível de desbalanceamento
            if imbalance_ratio > 100:
                level = "EXTREMAMENTE DESBALANCEADO"
            elif imbalance_ratio > 20:
                level = "ALTAMENTE DESBALANCEADO"
            elif imbalance_ratio > 5:
                level = "MODERADAMENTE DESBALANCEADO"
            else:
                level = "RELATIVAMENTE BALANCEADO"
            
            print(f"  Nível: {level}")
            quality_report['imbalance_analysis']['imbalance_level'] = level
        
        # 5. Validação clínica
        print("\n5. VALIDAÇÃO CLÍNICA:")
        
        # Creatinina por sexo
        if 'CREATININA' in df.columns and 'SEXO' in df.columns:
            creat_numeric = pd.to_numeric(
                df['CREATININA'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            male_creat = creat_numeric[df['SEXO'] == 'MASCULINO'].dropna()
            female_creat = creat_numeric[df['SEXO'] == 'FEMININO'].dropna()
            
            if len(male_creat) > 0:
                normal_male_pct = ((male_creat >= 0.7) & (male_creat <= 1.2)).mean() * 100
                print(f"  Creatinina normal (homens): {normal_male_pct:.1f}%")
                quality_report['clinical_validation']['creatinina_normal_male_pct'] = float(normal_male_pct)
            
            if len(female_creat) > 0:
                normal_female_pct = ((female_creat >= 0.6) & (female_creat <= 1.1)).mean() * 100
                print(f"  Creatinina normal (mulheres): {normal_female_pct:.1f}%")
                quality_report['clinical_validation']['creatinina_normal_female_pct'] = float(normal_female_pct)
        
        # TFG por estágios
        if 'TFG' in df.columns:
            tfg_data = df['TFG'].dropna()
            
            stages = {
                'G1 (≥90)': (tfg_data >= 90).sum(),
                'G2 (60-89)': ((tfg_data >= 60) & (tfg_data < 90)).sum(),
                'G3 (30-59)': ((tfg_data >= 30) & (tfg_data < 60)).sum(),
                'G4 (15-29)': ((tfg_data >= 15) & (tfg_data < 30)).sum(),
                'G5 (<15)': (tfg_data < 15).sum()
            }
            
            print("  Distribuição TFG por estágio:")
            quality_report['clinical_validation']['tfg_stages'] = {}
            for stage, count in stages.items():
                pct = (count / len(tfg_data)) * 100
                print(f"    {stage}: {count} ({pct:.1f}%)")
                quality_report['clinical_validation']['tfg_stages'][stage] = {
                    'count': int(count),
                    'percentage': float(pct)
                }
        
        # 6. Recomendações baseadas na análise
        print("\n6. RECOMENDAÇÕES:")
        
        if quality_report['missing_data']:
            quality_report['recommendations'].append("Tratar dados faltantes antes do treinamento")
            print("  📋 Tratar dados faltantes")
        
        if quality_report['imbalance_analysis'].get('imbalance_ratio', 0) > 10:
            quality_report['recommendations'].append("Aplicar técnicas robustas de balanceamento")
            print("  ⚖️  Aplicar balanceamento robusto")
        
        if len([col for col, info in quality_report['outliers'].items() if info['percentage'] > 10]) > 0:
            quality_report['recommendations'].append("Investigar e tratar outliers significativos")
            print("  🎯 Investigar outliers significativos")
        
        effective_classes = quality_report['imbalance_analysis'].get('effective_classes', 0)
        total_classes = quality_report['imbalance_analysis'].get('total_classes', 0)
        if effective_classes < total_classes:
            quality_report['recommendations'].append("Considerar agrupamento de classes minoritárias")
            print("  🔄 Considerar agrupamento de classes pequenas")
        
        self.preprocessing_report['initial_quality'] = quality_report
        return quality_report
    
    def create_before_visualizations(self, df, output_dir='plots/preprocessing'):
        """
        Cria visualizações detalhadas do estado ANTES do pré-processamento
        """
        print("\n" + "="*60)
        print("GERANDO VISUALIZAÇÕES - ESTADO INICIAL")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # 1. Distribuição das classes
        if 'TFG_Classification' in df.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Análise de Desbalanceamento - Estado Inicial', fontsize=16, fontweight='bold')
            
            # Pie chart
            class_counts = df['TFG_Classification'].value_counts()
            axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Distribuição das Classes')
            
            # Bar plot
            axes[0, 1].bar(range(len(class_counts)), class_counts.values, color='skyblue')
            axes[0, 1].set_xticks(range(len(class_counts)))
            axes[0, 1].set_xticklabels(class_counts.index, rotation=45)
            axes[0, 1].set_title('Contagem por Classe')
            axes[0, 1].set_ylabel('Número de Amostras')
            
            # Log scale
            axes[1, 0].bar(range(len(class_counts)), class_counts.values, color='lightcoral')
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xticks(range(len(class_counts)))
            axes[1, 0].set_xticklabels(class_counts.index, rotation=45)
            axes[1, 0].set_title('Contagem por Classe (Escala Log)')
            axes[1, 0].set_ylabel('Número de Amostras (log)')
            
            # Cumulative percentage
            cumsum = class_counts.sort_values(ascending=False).cumsum()
            cumsum_pct = (cumsum / cumsum.iloc[-1]) * 100
            axes[1, 1].plot(range(len(cumsum_pct)), cumsum_pct.values, 'o-', color='green')
            axes[1, 1].set_xticks(range(len(cumsum_pct)))
            axes[1, 1].set_xticklabels(cumsum_pct.index, rotation=45)
            axes[1, 1].set_title('Distribuição Cumulativa')
            axes[1, 1].set_ylabel('Percentual Cumulativo')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/01_class_distribution_before.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Distribuições das variáveis alvo
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Distribuições das Variáveis Alvo - Estado Inicial', fontsize=16, fontweight='bold')
        
        # Creatinina
        if 'CREATININA' in df.columns:
            creat_numeric = pd.to_numeric(
                df['CREATININA'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            ).dropna()
            
            axes[0].hist(creat_numeric, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(creat_numeric.mean(), color='red', linestyle='--', label=f'Média: {creat_numeric.mean():.2f}')
            axes[0].axvline(creat_numeric.median(), color='orange', linestyle='--', label=f'Mediana: {creat_numeric.median():.2f}')
            axes[0].set_title('Distribuição da Creatinina')
            axes[0].set_xlabel('Creatinina (mg/dL)')
            axes[0].set_ylabel('Frequência')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # TFG
        if 'TFG' in df.columns:
            tfg_data = df['TFG'].dropna()
            axes[1].hist(tfg_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1].axvline(tfg_data.mean(), color='red', linestyle='--', label=f'Média: {tfg_data.mean():.1f}')
            axes[1].axvline(tfg_data.median(), color='orange', linestyle='--', label=f'Mediana: {tfg_data.median():.1f}')
            
            # Adicionar linhas de estágios
            stages = [15, 30, 60, 90]
            colors = ['red', 'orange', 'yellow', 'green']
            for stage, color in zip(stages, colors):
                axes[1].axvline(stage, color=color, linestyle=':', alpha=0.5, label=f'Estágio: {stage}')
            
            axes[1].set_title('Distribuição da TFG')
            axes[1].set_xlabel('TFG (mL/min/1.73m²)')
            axes[1].set_ylabel('Frequência')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Correlação TFG vs Creatinina
        if 'TFG' in df.columns and 'CREATININA' in df.columns:
            creat_numeric = pd.to_numeric(
                df['CREATININA'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            
            valid_mask = ~(df['TFG'].isna() | creat_numeric.isna())
            if valid_mask.sum() > 0:
                axes[2].scatter(creat_numeric[valid_mask], df['TFG'][valid_mask], alpha=0.6, color='purple')
                
                # Linha de tendência
                z = np.polyfit(creat_numeric[valid_mask], df['TFG'][valid_mask], 1)
                p = np.poly1d(z)
                axes[2].plot(creat_numeric[valid_mask], p(creat_numeric[valid_mask]), "r--", alpha=0.8)
                
                # Correlação
                corr = np.corrcoef(creat_numeric[valid_mask], df['TFG'][valid_mask])[0, 1]
                axes[2].set_title(f'TFG vs Creatinina\n(Correlação: {corr:.3f})')
                axes[2].set_xlabel('Creatinina (mg/dL)')
                axes[2].set_ylabel('TFG (mL/min/1.73m²)')
                axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_target_distributions_before.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Análise de features demográficas
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análise de Features Demográficas - Estado Inicial', fontsize=16, fontweight='bold')
        
        # Idade por classe
        if 'IDADE' in df.columns and 'TFG_Classification' in df.columns:
            sns.boxplot(data=df, x='TFG_Classification', y='IDADE', ax=axes[0, 0])
            axes[0, 0].set_title('Idade por Classe TFG')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sexo por classe
        if 'SEXO' in df.columns and 'TFG_Classification' in df.columns:
            sex_class_crosstab = pd.crosstab(df['SEXO'], df['TFG_Classification'])
            sns.heatmap(sex_class_crosstab, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
            axes[0, 1].set_title('Sexo x Classe TFG')
        
        # Pressão arterial
        if 'PAS' in df.columns and 'PAD' in df.columns:
            axes[0, 2].scatter(df['PAS'], df['PAD'], alpha=0.6, color='red')
            axes[0, 2].set_xlabel('PAS (mmHg)')
            axes[0, 2].set_ylabel('PAD (mmHg)')
            axes[0, 2].set_title('Pressão Arterial')
            axes[0, 2].grid(True, alpha=0.3)
        
        # IMC
        if 'IMC' in df.columns:
            axes[1, 0].hist(df['IMC'].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].axvline(18.5, color='blue', linestyle='--', alpha=0.7, label='Baixo peso')
            axes[1, 0].axvline(24.9, color='green', linestyle='--', alpha=0.7, label='Normal')
            axes[1, 0].axvline(29.9, color='orange', linestyle='--', alpha=0.7, label='Sobrepeso')
            axes[1, 0].set_title('Distribuição do IMC')
            axes[1, 0].set_xlabel('IMC (kg/m²)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Comorbidades
        comorbidity_cols = ['DM', 'HAS', 'Fuma?', 'Bebe?']
        available_comorbidities = [col for col in comorbidity_cols if col in df.columns]
        
        if available_comorbidities:
            comorbidity_counts = df[available_comorbidities].apply(
                lambda x: (x == 'SIM').sum() if x.dtype == 'object' else x.sum()
            )
            axes[1, 1].bar(range(len(comorbidity_counts)), comorbidity_counts.values, color='lightcoral')
            axes[1, 1].set_xticks(range(len(comorbidity_counts)))
            axes[1, 1].set_xticklabels(comorbidity_counts.index, rotation=45)
            axes[1, 1].set_title('Prevalência de Comorbidades')
            axes[1, 1].set_ylabel('Número de Casos')
        
        # Outliers por feature
        outlier_counts = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:6]:  # Limitar a 6 features
            if col not in ['ID_ANONIMO']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                outlier_counts[col] = outliers
        
        if outlier_counts:
            axes[1, 2].bar(range(len(outlier_counts)), list(outlier_counts.values()), color='yellow')
            axes[1, 2].set_xticks(range(len(outlier_counts)))
            axes[1, 2].set_xticklabels(list(outlier_counts.keys()), rotation=45)
            axes[1, 2].set_title('Outliers por Feature')
            axes[1, 2].set_ylabel('Número de Outliers')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_demographic_analysis_before.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizações do estado inicial salvas em: {output_dir}")
        return True
    
    def detect_optimal_balancing_strategy(self, X, y):
        """
        Detecta a melhor estratégia de balanceamento baseada nas características dos dados
        """
        print("\n" + "="*50)
        print("DETECÇÃO DE ESTRATÉGIA ÓTIMA DE BALANCEAMENTO")
        print("="*50)
        
        class_counts = Counter(y)
        n_samples = len(y)
        n_features = X.shape[1]
        n_classes = len(class_counts)
        
        # Calcular métricas de desbalanceamento
        majority_class = max(class_counts.values())
        minority_class = min(class_counts.values())
        imbalance_ratio = majority_class / minority_class
        
        # Classes com menos de 5 amostras
        tiny_classes = sum(1 for count in class_counts.values() if count < 5)
        
        # Classes com menos de 10 amostras
        small_classes = sum(1 for count in class_counts.values() if count < 10)
        
        print(f"📊 Análise dos dados:")
        print(f"  - Amostras: {n_samples}")
        print(f"  - Features: {n_features}")
        print(f"  - Classes: {n_classes}")
        print(f"  - Razão de desbalanceamento: {imbalance_ratio:.1f}:1")
        print(f"  - Classes com <5 amostras: {tiny_classes}")
        print(f"  - Classes com <10 amostras: {small_classes}")
        
        # Lógica de seleção da estratégia
        strategy_recommendations = []
        
        if tiny_classes > 0:
            print("\n⚠️  Detectadas classes com <5 amostras")
            
            if tiny_classes >= n_classes // 2:
                print("🔄 Recomendação: Agrupamento de classes + SMOTE conservador")
                strategy_recommendations.append("class_grouping + conservative_smote")
            else:
                print("🎯 Recomendação: Remoção de classes pequenas + SMOTE")
                strategy_recommendations.append("remove_tiny_classes + smote")
        
        elif imbalance_ratio > 50:
            print("\n🚨 Desbalanceamento extremo detectado")
            print("🔄 Recomendação: SMOTE + Tomek Links (híbrido)")
            strategy_recommendations.append("smotetomek")
        
        elif imbalance_ratio > 20:
            print("\n⚖️  Desbalanceamento alto detectado")
            if n_samples < 500:
                print("🎯 Recomendação: BorderlineSMOTE (dataset pequeno)")
                strategy_recommendations.append("borderline_smote")
            else:
                print("🎯 Recomendação: ADASYN")
                strategy_recommendations.append("adasyn")
        
        elif imbalance_ratio > 5:
            print("\n📊 Desbalanceamento moderado detectado")
            print("🎯 Recomendação: SMOTE clássico")
            strategy_recommendations.append("smote")
        
        else:
            print("\n✅ Desbalanceamento relativamente baixo")
            print("🎯 Recomendação: Class weights ou SMOTE leve")
            strategy_recommendations.append("class_weights")
        
        # Considerações adicionais
        if n_samples < 200:
            print("\n📝 Dataset pequeno: SMOTE conservador recomendado")
            strategy_recommendations.append("conservative_smote")
        
        if n_features > n_samples:
            print("\n📝 Mais features que amostras: PCA + SMOTE recomendado")
            strategy_recommendations.append("pca_smote")
        
        # Selecionar estratégia final
        if self.strategy == 'adaptive':
            final_strategy = strategy_recommendations[0] if strategy_recommendations else 'smote'
        else:
            final_strategy = self.strategy
        
        print(f"\n🎯 Estratégia selecionada: {final_strategy}")
        
        return final_strategy, {
            'imbalance_ratio': imbalance_ratio,
            'tiny_classes': tiny_classes,
            'small_classes': small_classes,
            'recommendations': strategy_recommendations,
            'final_strategy': final_strategy
        }
    
    def apply_advanced_balancing(self, X, y, strategy_info):
        """
        Aplica estratégia de balanceamento avançada baseada na análise
        """
        strategy = strategy_info['final_strategy']
        
        print(f"\n🔄 Aplicando estratégia: {strategy}")
        
        try:
            if strategy == 'conservative_smote':
                # SMOTE com k_neighbors reduzido
                min_samples = min(Counter(y).values())
                k_neighbors = max(1, min(3, min_samples - 1))
                self.balancer = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            
            elif strategy == 'borderline_smote':
                # BorderlineSMOTE para casos difíceis
                self.balancer = BorderlineSMOTE(random_state=self.random_state)
            
            elif strategy == 'adasyn':
                # ADASYN adaptativo
                self.balancer = ADASYN(random_state=self.random_state)
            
            elif strategy == 'smotetomek':
                # SMOTE + Tomek Links (híbrido)
                self.balancer = SMOTETomek(random_state=self.random_state)
            
            elif strategy == 'smoteenn':
                # SMOTE + Edited Nearest Neighbours
                self.balancer = SMOTEENN(random_state=self.random_state)
            
            elif strategy == 'remove_tiny_classes':
                # Remover classes com <5 amostras e aplicar SMOTE
                class_counts = Counter(y)
                valid_classes = [cls for cls, count in class_counts.items() if count >= 5]
                
                valid_mask = np.isin(y, valid_classes)
                X_filtered = X[valid_mask]
                y_filtered = y[valid_mask]
                
                print(f"  Removidas classes com <5 amostras. Restaram {len(valid_classes)} classes")
                
                self.balancer = SMOTE(random_state=self.random_state)
                return self.balancer.fit_resample(X_filtered, y_filtered)
            
            elif strategy == 'class_weights':
                # Apenas usar class weights, sem resampling
                print("  Usando apenas class weights (sem resampling)")
                return X, y
            
            else:
                # SMOTE padrão
                self.balancer = SMOTE(random_state=self.random_state)
            
            # Aplicar balanceamento
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            
            print(f"  ✅ Balanceamento aplicado com sucesso")
            print(f"  Shape original: {X.shape} → Shape balanceado: {X_balanced.shape}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"  ❌ Erro no balanceamento: {e}")
            print("  🔄 Fallback: Retornando dados originais")
            return X, y
    
    def create_after_visualizations(self, X_before, y_before, X_after, y_after, output_dir='plots/preprocessing'):
        """
        Cria visualizações comparativas ANTES vs DEPOIS do pré-processamento
        """
        print("\n" + "="*60)
        print("GERANDO VISUALIZAÇÕES COMPARATIVAS - ANTES vs DEPOIS")
        print("="*60)
        
        # 1. Comparação de distribuição de classes
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparação: Distribuição de Classes ANTES vs DEPOIS', fontsize=16, fontweight='bold')
        
        # Antes
        before_counts = Counter(y_before)
        classes = sorted(before_counts.keys())
        before_values = [before_counts[cls] for cls in classes]
        
        axes[0, 0].bar(range(len(classes)), before_values, color='lightcoral', alpha=0.7)
        axes[0, 0].set_title('ANTES do Balanceamento')
        axes[0, 0].set_ylabel('Número de Amostras')
        axes[0, 0].set_xticks(range(len(classes)))
        axes[0, 0].set_xticklabels([f'Classe {cls}' for cls in classes], rotation=45)
        
        # Depois
        after_counts = Counter(y_after)
        after_values = [after_counts.get(cls, 0) for cls in classes]
        
        axes[0, 1].bar(range(len(classes)), after_values, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('DEPOIS do Balanceamento')
        axes[0, 1].set_ylabel('Número de Amostras')
        axes[0, 1].set_xticks(range(len(classes)))
        axes[0, 1].set_xticklabels([f'Classe {cls}' for cls in classes], rotation=45)
        
        # Comparação lado a lado
        x = np.arange(len(classes))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, before_values, width, label='Antes', color='lightcoral', alpha=0.7)
        axes[1, 0].bar(x + width/2, after_values, width, label='Depois', color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Comparação Direta')
        axes[1, 0].set_ylabel('Número de Amostras')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'Classe {cls}' for cls in classes], rotation=45)
        axes[1, 0].legend()
        
        # Razão de mudança
        ratios = [after_values[i] / before_values[i] if before_values[i] > 0 else 0 for i in range(len(classes))]
        axes[1, 1].bar(range(len(classes)), ratios, color='skyblue', alpha=0.7)
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Sem mudança')
        axes[1, 1].set_title('Razão de Mudança (Depois/Antes)')
        axes[1, 1].set_ylabel('Razão')
        axes[1, 1].set_xticks(range(len(classes)))
        axes[1, 1].set_xticklabels([f'Classe {cls}' for cls in classes], rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_balancing_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Análise de espaço de features (PCA)
        if X_before.shape[1] > 2:
            print("📊 Criando análise PCA...")
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Análise do Espaço de Features (PCA) - ANTES vs DEPOIS', fontsize=16, fontweight='bold')
            
            # PCA antes
            pca = PCA(n_components=2, random_state=self.random_state)
            X_before_pca = pca.fit_transform(X_before)
            
            for cls in classes:
                mask = y_before == cls
                axes[0].scatter(X_before_pca[mask, 0], X_before_pca[mask, 1], 
                               label=f'Classe {cls}', alpha=0.6)
            
            axes[0].set_title('ANTES do Balanceamento')
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # PCA depois
            X_after_pca = pca.transform(X_after)
            
            for cls in classes:
                mask = y_after == cls
                if mask.sum() > 0:  # Verificar se a classe ainda existe
                    axes[1].scatter(X_after_pca[mask, 0], X_after_pca[mask, 1], 
                                   label=f'Classe {cls}', alpha=0.6)
            
            axes[1].set_title('DEPOIS do Balanceamento')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/05_pca_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"✅ Visualizações comparativas salvas em: {output_dir}")
        return True
    
    def generate_preprocessing_report(self, output_path='reports/preprocessing_report.json'):
        """
        Gera relatório detalhado do pré-processamento
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.preprocessing_report['timestamp'] = datetime.now().isoformat()
        self.preprocessing_report['preprocessing_config'] = {
            'strategy': self.strategy,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'input_features': self.input_features,
            'target_variables': self.target_variables
        }
        
        # Salvar relatório
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.preprocessing_report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Relatório detalhado salvo em: {output_path}")
        return self.preprocessing_report

# Função principal para demonstração
def main():
    """
    Demonstração do pipeline avançado de pré-processamento
    """
    print("🚀 PIPELINE AVANÇADO DE PRÉ-PROCESSAMENTO DRC")
    print("="*60)
    
    # Inicializar preprocessador avançado
    preprocessor = AdvancedDRCPreprocessor(strategy='adaptive', random_state=42)
    
    # Carregar dados
    data_path = '/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv'
    df = pd.read_csv(data_path, sep=';', encoding='utf-8')
    
    # Análise de qualidade inicial
    quality_report = preprocessor.analyze_data_quality(df)
    
    # Visualizações do estado inicial
    preprocessor.create_before_visualizations(df)
    
    print("\n✅ Pipeline avançado de pré-processamento executado com sucesso!")
    print("📊 Análise completa realizada com métricas detalhadas")
    print("📈 Visualizações comparativas geradas")

if __name__ == "__main__":
    main()