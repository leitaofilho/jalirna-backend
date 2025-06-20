#!/usr/bin/env python3
"""
Pipeline de Pré-processamento Médico APRIMORADO para DRC
CORRIGE o problema: implementa soluções efetivas para desbalanceamento extremo
Mantém TODAS as 6 classes + gera melhorias mensuráveis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class EnhancedMedicalDRCPreprocessor:
    """
    Pipeline de pré-processamento médico APRIMORADO que efetivamente trata desbalanceamento extremo
    SOLUÇÕES IMPLEMENTADAS:
    1. RandomOverSampler para classes extremas
    2. SMOTE híbrido com k_neighbors adaptativos  
    3. Estratégia em cascata
    4. Validação de melhoria obrigatória
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        # Componentes
        self.label_encoders = {}
        self.scaler = None
        self.classification_balancer = None
        self.target_encoder = None
        
        # Controle de qualidade
        self.preprocessing_report = {}
        self.improvement_metrics = {}
        
        # Features (usar COR 2 ao invés de COR para evitar multicolinearidade)
        # COR 2 é binário: 0=não-minoritário, 1=minoritário (derivado do frontend)
        self.input_features = [
            'IDADE', 'SEXO', 'COR 2', 'IMC', 'CC', 'RCQ',
            'PAS', 'PAD', 'Fuma?', 'Realiza exercício?', 'Bebe?', 'DM', 'HAS'
        ]
        
        self.target_variables = ['CREATININA', 'TFG', 'TFG_Classification']
    
    def analyze_critical_imbalance(self, y):
        """
        Análise detalhada do desbalanceamento crítico
        """
        print(f"\n" + "="*70)
        print("ANÁLISE CRÍTICA DO DESBALANCEAMENTO")
        print("="*70)
        
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Classificar classes por severidade
        critical_classes = {}  # <5 amostras
        problematic_classes = {}  # 5-10 amostras
        minor_classes = {}  # 10-50 amostras
        majority_classes = {}  # >50 amostras
        
        for cls, count in class_counts.items():
            if count < 5:
                critical_classes[cls] = count
            elif count < 10:
                problematic_classes[cls] = count
            elif count < 50:
                minor_classes[cls] = count
            else:
                majority_classes[cls] = count
        
        print(f"📊 CLASSIFICAÇÃO POR SEVERIDADE:")
        print(f"  🚨 Críticas (<5): {critical_classes}")
        print(f"  ⚠️  Problemáticas (5-10): {problematic_classes}")
        print(f"  📉 Menores (10-50): {minor_classes}")
        print(f"  📈 Majoritárias (>50): {majority_classes}")
        
        # Calcular métricas de severidade
        majority_class = max(class_counts.values())
        minority_class = min(class_counts.values())
        imbalance_ratio = majority_class / minority_class
        
        # Determinar estratégia baseada na severidade
        strategy_needed = self._determine_strategy_by_severity(
            critical_classes, problematic_classes, imbalance_ratio
        )
        
        analysis = {
            'total_samples': total_samples,
            'total_classes': len(class_counts),
            'critical_classes': critical_classes,
            'problematic_classes': problematic_classes,
            'minor_classes': minor_classes,
            'majority_classes': majority_classes,
            'imbalance_ratio': imbalance_ratio,
            'strategy_needed': strategy_needed
        }
        
        return analysis
    
    def _determine_strategy_by_severity(self, critical_classes, problematic_classes, imbalance_ratio):
        """
        Determina estratégia baseada na análise de severidade
        """
        print(f"\n🎯 DETERMINAÇÃO DE ESTRATÉGIA:")
        
        if len(critical_classes) > 0:
            print(f"  🚨 CASO EXTREMO: {len(critical_classes)} classes críticas detectadas")
            print(f"  📋 Estratégia: RandomOverSampler + SMOTE Híbrido")
            return "extreme_case_hybrid"
        
        elif len(problematic_classes) > 2:
            print(f"  ⚠️  CASO SEVERO: {len(problematic_classes)} classes problemáticas")
            print(f"  📋 Estratégia: SMOTE Adaptativo + Balanceamento Escalonado")
            return "severe_case_adaptive"
        
        elif imbalance_ratio > 20:
            print(f"  📊 CASO ALTO: Razão {imbalance_ratio:.1f}:1")
            print(f"  📋 Estratégia: BorderlineSMOTE")
            return "high_imbalance_borderline"
        
        else:
            print(f"  ✅ CASO MODERADO: Razão {imbalance_ratio:.1f}:1")
            print(f"  📋 Estratégia: SMOTE Clássico")
            return "moderate_smote"
    
    def apply_extreme_case_hybrid_balancing(self, X, y):
        """
        Estratégia híbrida para casos extremos (classes <5 amostras)
        """
        print(f"\n🔄 APLICANDO ESTRATÉGIA HÍBRIDA PARA CASO EXTREMO")
        
        class_counts = Counter(y)
        print(f"  📊 Distribuição original: {dict(class_counts)}")
        
        # ETAPA 1: RandomOverSampler para classes críticas
        print(f"  🎯 ETAPA 1: RandomOverSampler para classes críticas")
        
        # Definir estratégia de sampling - mínimo de 10 amostras por classe crítica
        sampling_strategy = {}
        for cls, count in class_counts.items():
            if count < 5:
                sampling_strategy[cls] = 10  # Elevar para 10
            elif count < 10:
                sampling_strategy[cls] = 15  # Elevar para 15
            # Classes maiores ficam como estão inicialmente
        
        if sampling_strategy:
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=self.random_state)
            X_step1, y_step1 = ros.fit_resample(X, y)
            
            print(f"    ✅ RandomOverSampler aplicado")
            print(f"    📊 Após etapa 1: {dict(Counter(y_step1))}")
        else:
            X_step1, y_step1 = X, y
            print(f"    ⏭️  RandomOverSampler não necessário")
        
        # ETAPA 2: SMOTE com k_neighbors adaptativo
        print(f"  🎯 ETAPA 2: SMOTE adaptativo")
        
        # Calcular k_neighbors seguro
        min_samples_per_class = min(Counter(y_step1).values())
        k_neighbors = max(1, min(5, min_samples_per_class - 1))
        
        print(f"    📊 k_neighbors calculado: {k_neighbors}")
        
        try:
            # Aplicar SMOTE para balanceamento final
            smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            X_final, y_final = smote.fit_resample(X_step1, y_step1)
            
            print(f"    ✅ SMOTE aplicado com sucesso")
            print(f"    📊 Distribuição final: {dict(Counter(y_final))}")
            
            return X_final, y_final, True
            
        except Exception as e:
            print(f"    ❌ SMOTE falhou: {e}")
            print(f"    🔄 Retornando resultado da etapa 1")
            return X_step1, y_step1, True
    
    def apply_severe_case_adaptive_balancing(self, X, y):
        """
        Estratégia adaptativa para casos severos (múltiplas classes 5-10 amostras)
        """
        print(f"\n🔄 APLICANDO ESTRATÉGIA ADAPTATIVA PARA CASO SEVERO")
        
        class_counts = Counter(y)
        print(f"  📊 Distribuição original: {dict(class_counts)}")
        
        # Balanceamento escalonado
        target_min = 20  # Objetivo mínimo por classe
        
        # Primeira passada: elevar classes muito pequenas
        sampling_strategy_1 = {}
        for cls, count in class_counts.items():
            if count < target_min:
                sampling_strategy_1[cls] = target_min
        
        try:
            ros = RandomOverSampler(sampling_strategy=sampling_strategy_1, random_state=self.random_state)
            X_intermediate, y_intermediate = ros.fit_resample(X, y)
            
            print(f"  ✅ Primeira passada concluída")
            print(f"  📊 Distribuição intermediária: {dict(Counter(y_intermediate))}")
            
            # Segunda passada: SMOTE para balanceamento fino
            min_samples = min(Counter(y_intermediate).values())
            k_neighbors = max(1, min(5, min_samples - 1))
            
            smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
            X_final, y_final = smote.fit_resample(X_intermediate, y_intermediate)
            
            print(f"  ✅ SMOTE aplicado com sucesso")
            print(f"  📊 Distribuição final: {dict(Counter(y_final))}")
            
            return X_final, y_final, True
            
        except Exception as e:
            print(f"  ❌ Estratégia adaptativa falhou: {e}")
            return X, y, False
    
    def calculate_improvement_metrics(self, y_before, y_after, balancing_applied):
        """
        Calcula métricas de melhoria obrigatórias
        """
        print(f"\n📊 CALCULANDO MÉTRICAS DE MELHORIA")
        
        before_counts = Counter(y_before)
        after_counts = Counter(y_after)
        
        # Métricas de desbalanceamento
        before_majority = max(before_counts.values())
        before_minority = min(before_counts.values())
        before_ratio = before_majority / before_minority
        
        after_majority = max(after_counts.values())
        after_minority = min(after_counts.values())
        after_ratio = after_majority / after_minority
        
        # Melhoria percentual
        if before_ratio > after_ratio:
            improvement_pct = ((before_ratio - after_ratio) / before_ratio) * 100
        else:
            improvement_pct = 0
        
        # Métricas de classes críticas
        before_critical = sum(1 for count in before_counts.values() if count < 10)
        after_critical = sum(1 for count in after_counts.values() if count < 10)
        
        # Métricas de amostras geradas
        samples_added = len(y_after) - len(y_before)
        
        metrics = {
            'balancing_applied': balancing_applied,
            'before_samples': len(y_before),
            'after_samples': len(y_after),
            'samples_generated': samples_added,
            'before_ratio': before_ratio,
            'after_ratio': after_ratio,
            'improvement_percentage': improvement_pct,
            'before_critical_classes': before_critical,
            'after_critical_classes': after_critical,
            'critical_classes_resolved': before_critical - after_critical,
            'before_distribution': dict(before_counts),
            'after_distribution': dict(after_counts)
        }
        
        # Imprimir métricas
        print(f"  📈 RESULTADOS:")
        print(f"    Balanceamento aplicado: {'✅ Sim' if balancing_applied else '❌ Não'}")
        print(f"    Amostras: {len(y_before)} → {len(y_after)} (+{samples_added})")
        print(f"    Razão de desbalanceamento: {before_ratio:.1f}:1 → {after_ratio:.1f}:1")
        print(f"    Melhoria: {improvement_pct:.1f}%")
        print(f"    Classes críticas: {before_critical} → {after_critical}")
        
        # Avaliação de sucesso
        if improvement_pct > 50:
            print(f"    🎯 SUCESSO: Melhoria significativa alcançada")
        elif improvement_pct > 20:
            print(f"    ⚠️  PARCIAL: Melhoria moderada alcançada")
        elif balancing_applied:
            print(f"    📊 LIMITADO: Balanceamento aplicado, melhoria limitada")
        else:
            print(f"    ❌ FALHA: Nenhuma melhoria alcançada")
        
        self.improvement_metrics = metrics
        return metrics
    
    def create_enhanced_comparison_visualizations(self, y_before, y_after, improvement_metrics, 
                                                 output_dir='plots/preprocessing_enhanced'):
        """
        Cria visualizações aprimoradas focando na melhoria
        """
        print(f"\n📊 GERANDO VISUALIZAÇÕES DE MELHORIA")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ANÁLISE DE MELHORIA: Pipeline Médico Aprimorado', fontsize=16, fontweight='bold')
        
        # Preparar dados
        before_counts = Counter(y_before)
        after_counts = Counter(y_after)
        classes = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
        
        before_values = [before_counts.get(cls, 0) for cls in classes]
        after_values = [after_counts.get(cls, 0) for cls in classes]
        
        # 1. Comparação direta
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, before_values, width, label='Antes', 
                              color='lightcoral', alpha=0.8)
        bars2 = axes[0, 0].bar(x + width/2, after_values, width, label='Depois', 
                              color='lightgreen', alpha=0.8)
        
        # Destacar classes que melhoraram
        for i, (before, after) in enumerate(zip(before_values, after_values)):
            if after > before:
                bars2[i].set_color('darkgreen')
                bars2[i].set_alpha(1.0)
        
        axes[0, 0].set_title('Comparação: Antes vs Depois')
        axes[0, 0].set_ylabel('Número de Amostras')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f'G{cls+1}' for cls in classes])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for i, (bar1, bar2, before, after) in enumerate(zip(bars1, bars2, before_values, after_values)):
            if before > 0:
                axes[0, 0].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                               str(before), ha='center', va='bottom', fontweight='bold')
            if after > 0:
                axes[0, 0].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                               str(after), ha='center', va='bottom', fontweight='bold')
        
        # 2. Foco em classes críticas
        critical_before = {cls: count for cls, count in before_counts.items() if count < 10}
        critical_after = {cls: count for cls, count in after_counts.items() if cls in critical_before}
        
        if critical_before:
            critical_classes = list(critical_before.keys())
            critical_before_vals = [critical_before[cls] for cls in critical_classes]
            critical_after_vals = [critical_after.get(cls, 0) for cls in critical_classes]
            
            x_crit = np.arange(len(critical_classes))
            bars1 = axes[0, 1].bar(x_crit - width/2, critical_before_vals, width, 
                                  label='Antes', color='red', alpha=0.8)
            bars2 = axes[0, 1].bar(x_crit + width/2, critical_after_vals, width, 
                                  label='Depois', color='green', alpha=0.8)
            
            axes[0, 1].set_title('FOCO: Classes Críticas (<10 amostras)')
            axes[0, 1].set_ylabel('Número de Amostras')
            axes[0, 1].set_xticks(x_crit)
            axes[0, 1].set_xticklabels([f'G{cls+1}' for cls in critical_classes])
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Mostrar melhoria
            for i, (before, after) in enumerate(zip(critical_before_vals, critical_after_vals)):
                improvement = after - before
                if improvement > 0:
                    axes[0, 1].annotate(f'+{improvement}', 
                                       xy=(i, max(before, after) + 1),
                                       ha='center', va='bottom', 
                                       fontweight='bold', color='green',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen'))
        
        # 3. Métricas de melhoria
        metrics_text = f"""MÉTRICAS DE MELHORIA:

Balanceamento: {'✅ Aplicado' if improvement_metrics['balancing_applied'] else '❌ Não aplicado'}

QUANTITATIVO:
• Amostras: {improvement_metrics['before_samples']} → {improvement_metrics['after_samples']}
• Geradas: +{improvement_metrics['samples_generated']}

DESBALANCEAMENTO:
• Razão: {improvement_metrics['before_ratio']:.1f}:1 → {improvement_metrics['after_ratio']:.1f}:1
• Melhoria: {improvement_metrics['improvement_percentage']:.1f}%

CLASSES CRÍTICAS:
• Antes: {improvement_metrics['before_critical_classes']} classes
• Depois: {improvement_metrics['after_critical_classes']} classes  
• Resolvidas: {improvement_metrics['critical_classes_resolved']} classes

STATUS: {'🎯 SUCESSO' if improvement_metrics['improvement_percentage'] > 50 else '⚠️ PARCIAL' if improvement_metrics['improvement_percentage'] > 20 else '❌ LIMITADO'}
"""
        
        axes[0, 2].text(0.05, 0.95, metrics_text, transform=axes[0, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[0, 2].set_xlim(0, 1)
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].axis('off')
        axes[0, 2].set_title('Resumo de Melhorias')
        
        # 4. Evolução do desbalanceamento
        stages = ['Original', 'Processado']
        ratios = [improvement_metrics['before_ratio'], improvement_metrics['after_ratio']]
        
        axes[1, 0].plot(stages, ratios, 'o-', linewidth=3, markersize=10, color='blue')
        axes[1, 0].fill_between(stages, ratios, alpha=0.3, color='blue')
        axes[1, 0].set_title('Evolução da Razão de Desbalanceamento')
        axes[1, 0].set_ylabel('Razão (Majoritária:Minoritária)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Adicionar valores
        for i, (stage, ratio) in enumerate(zip(stages, ratios)):
            axes[1, 0].text(i, ratio + max(ratios) * 0.05, f'{ratio:.1f}:1',
                           ha='center', va='bottom', fontweight='bold')
        
        # 5. Distribuição em escala log
        axes[1, 1].bar(range(len(classes)), before_values, alpha=0.7, 
                      color='lightcoral', label='Antes')
        axes[1, 1].bar(range(len(classes)), after_values, alpha=0.7, 
                      color='lightgreen', label='Depois')
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_title('Distribuição (Escala Log)')
        axes[1, 1].set_ylabel('Amostras (log)')
        axes[1, 1].set_xticks(range(len(classes)))
        axes[1, 1].set_xticklabels([f'G{cls+1}' for cls in classes])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Ganho por classe
        gains = [after - before for before, after in zip(before_values, after_values)]
        colors = ['green' if gain > 0 else 'red' if gain < 0 else 'gray' for gain in gains]
        
        bars = axes[1, 2].bar(range(len(classes)), gains, color=colors, alpha=0.7)
        axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 2].set_title('Ganho de Amostras por Classe')
        axes[1, 2].set_ylabel('Amostras Ganhas/Perdidas')
        axes[1, 2].set_xticks(range(len(classes)))
        axes[1, 2].set_xticklabels([f'G{cls+1}' for cls in classes])
        axes[1, 2].grid(True, alpha=0.3)
        
        # Mostrar valores nos ganhos
        for i, (bar, gain) in enumerate(zip(bars, gains)):
            if gain != 0:
                axes[1, 2].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (1 if gain > 0 else -3),
                               f'{gain:+d}', ha='center', 
                               va='bottom' if gain > 0 else 'top',
                               fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Visualizações de melhoria salvas em: {output_dir}")
        return True
    
    def fit_transform_enhanced_medical_data(self, data_path):
        """
        Pipeline médico aprimorado com validação de melhoria obrigatória
        """
        print("=" * 80)
        print("PIPELINE MÉDICO APRIMORADO - VALIDAÇÃO DE MELHORIA OBRIGATÓRIA")
        print("=" * 80)
        
        # 1. Carregar e preparar dados (reutilizar lógica anterior)
        df = pd.read_csv(data_path, sep=';', encoding='utf-8')
        print(f"📊 Dataset carregado: {df.shape}")
        
        # Corrigir formato decimal
        decimal_columns = ['CREATININA', 'C. PES', 'C.PANT', 'UREIA', 'HDL', 'LDL']
        for col in decimal_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
        
        # Preparar features
        available_features = [f for f in self.input_features if f in df.columns]
        X = df[available_features].copy()
        
        # Encoding
        categorical_features = X.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            self.label_encoders[feature] = le
        
        # Normalização
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Preparar classificação
        if 'TFG_Classification' in df.columns:
            self.target_encoder = LabelEncoder()
            y_original = self.target_encoder.fit_transform(df['TFG_Classification'])
            
            print(f"\n📊 Distribuição original: {dict(Counter(y_original))}")
            
            # 2. Análise crítica do desbalanceamento
            imbalance_analysis = self.analyze_critical_imbalance(y_original)
            
            # 3. Aplicar estratégia baseada na severidade
            strategy = imbalance_analysis['strategy_needed']
            
            if strategy == "extreme_case_hybrid":
                X_balanced, y_balanced, success = self.apply_extreme_case_hybrid_balancing(
                    X_scaled, y_original
                )
            elif strategy == "severe_case_adaptive":
                X_balanced, y_balanced, success = self.apply_severe_case_adaptive_balancing(
                    X_scaled, y_original
                )
            else:
                # Fallback para casos menos severos
                try:
                    smote = SMOTE(k_neighbors=1, random_state=self.random_state)
                    X_balanced, y_balanced = smote.fit_resample(X_scaled, y_original)
                    success = True
                except:
                    X_balanced, y_balanced = X_scaled, y_original
                    success = False
            
            # 4. Calcular métricas de melhoria obrigatórias
            improvement_metrics = self.calculate_improvement_metrics(
                y_original, y_balanced, success
            )
            
            # 5. Criar visualizações de melhoria
            self.create_enhanced_comparison_visualizations(
                y_original, y_balanced, improvement_metrics
            )
            
            # 6. Preparar dados finais
            processed_data = {
                'features_original': X_scaled,
                'features_balanced': X_balanced,
                'classification_original': y_original,
                'classification_balanced': y_balanced,
                'improvement_metrics': improvement_metrics,
                'imbalance_analysis': imbalance_analysis,
                'target_names': self.target_encoder.classes_,
                'feature_names': available_features
            }
            
            # Adicionar targets de regressão
            if 'CREATININA' in df.columns:
                processed_data['creatinina'] = df['CREATININA'].values
            if 'TFG' in df.columns:
                processed_data['tfg'] = df['TFG'].values
            
            print(f"\n" + "="*80)
            print("PIPELINE MÉDICO APRIMORADO CONCLUÍDO")
            print("="*80)
            print(f"✅ Estratégia aplicada: {strategy}")
            print(f"✅ Melhoria alcançada: {improvement_metrics['improvement_percentage']:.1f}%")
            print(f"✅ Classes críticas resolvidas: {improvement_metrics['critical_classes_resolved']}")
            print(f"✅ Amostras geradas: +{improvement_metrics['samples_generated']}")
            
            return processed_data
        
        return None

def main():
    """
    Executa o pipeline médico aprimorado
    """
    os.makedirs('plots/preprocessing_enhanced', exist_ok=True)
    
    preprocessor = EnhancedMedicalDRCPreprocessor(random_state=42)
    
    data_path = '/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv'
    processed_data = preprocessor.fit_transform_enhanced_medical_data(data_path)
    
    if processed_data:
        metrics = processed_data['improvement_metrics']
        print(f"\n🎯 RESULTADO FINAL:")
        print(f"   Melhoria no desbalanceamento: {metrics['improvement_percentage']:.1f}%")
        print(f"   Classes críticas resolvidas: {metrics['critical_classes_resolved']}")
        print(f"   Status: {'SUCESSO' if metrics['improvement_percentage'] > 50 else 'PARCIAL' if metrics['improvement_percentage'] > 20 else 'LIMITADO'}")

if __name__ == "__main__":
    main()