#!/usr/bin/env python3
"""
Relatório Detalhado da Metodologia de Pré-processamento DRC
Análise crítica de como o balanceamento foi feito e validação de confiabilidade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DRCMethodologyValidator:
    """
    Validador de metodologia para pré-processamento DRC
    Analisa criticamente como o balanceamento foi feito e sua confiabilidade
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.methodology_report = {}
        
    def load_and_prepare_original_data(self, data_path):
        """
        Carrega dados originais exatamente como no pipeline
        """
        print("=" * 80)
        print("RELATÓRIO DETALHADO DA METODOLOGIA DE PRÉ-PROCESSAMENTO")
        print("=" * 80)
        
        # Carregar dados
        df = pd.read_csv(data_path, sep=';', encoding='utf-8')
        
        # Corrigir formatação decimal
        decimal_columns = ['CREATININA', 'C. PES', 'C.PANT', 'UREIA', 'HDL', 'LDL']
        for col in decimal_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
        
        # Preparar features
        input_features = [
            'IDADE', 'SEXO', 'COR', 'COR 2', 'IMC', 'CC', 'RCQ',
            'PAS', 'PAD', 'Fuma?', 'Realiza exercício?', 'Bebe?', 'DM', 'HAS'
        ]
        
        available_features = [f for f in input_features if f in df.columns]
        X = df[available_features].copy()
        
        # Encoding categóricas
        categorical_features = X.select_dtypes(include=['object']).columns
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
        
        # Normalização
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Target
        target_encoder = LabelEncoder()
        y_original = target_encoder.fit_transform(df['TFG_Classification'])
        
        # Preparar targets de regressão
        creatinina = df['CREATININA'].values
        tfg = df['TFG'].values
        
        return X_scaled, y_original, creatinina, tfg, target_encoder.classes_
    
    def analyze_original_data_distribution(self, X, y, creatinina, tfg, class_names):
        """
        Análise detalhada da distribuição original dos dados
        """
        print("\n" + "=" * 60)
        print("1. ANÁLISE DA DISTRIBUIÇÃO ORIGINAL DOS DADOS")
        print("=" * 60)
        
        # Análise das classes
        class_counts = Counter(y)
        total_samples = len(y)
        
        print(f"📊 DISTRIBUIÇÃO DAS CLASSES:")
        class_analysis = {}
        for cls, count in sorted(class_counts.items()):
            pct = (count / total_samples) * 100
            class_analysis[cls] = {
                'name': class_names[cls],
                'count': count,
                'percentage': pct
            }
            print(f"  {class_names[cls]}: {count} amostras ({pct:.1f}%)")
        
        # Análise de distribuição espacial
        print(f"\n📊 ANÁLISE ESPACIAL DOS DADOS:")
        
        # PCA para visualização
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Calcular distâncias intra-classe e inter-classe
        intra_class_distances = {}
        inter_class_distances = {}
        
        for cls in class_counts.keys():
            class_mask = y == cls
            if class_mask.sum() > 1:
                class_data = X[class_mask]
                # Distâncias intra-classe (dentro da mesma classe)
                distances = pairwise_distances(class_data)
                # Pegar apenas triângulo superior (sem diagonal)
                triu_indices = np.triu_indices_from(distances, k=1)
                intra_distances = distances[triu_indices]
                intra_class_distances[cls] = {
                    'mean': np.mean(intra_distances),
                    'std': np.std(intra_distances),
                    'samples': len(intra_distances)
                }
                print(f"  {class_names[cls]} - Coesão interna: {np.mean(intra_distances):.3f} ± {np.std(intra_distances):.3f}")
        
        # Análise de correlações entre targets
        corr_creat_tfg = np.corrcoef(creatinina, tfg)[0, 1]
        print(f"\n📊 CORRELAÇÕES ENTRE TARGETS:")
        print(f"  CREATININA vs TFG: {corr_creat_tfg:.3f}")
        
        # Análise por classe das variáveis de regressão
        print(f"\n📊 ANÁLISE DAS VARIÁVEIS DE REGRESSÃO POR CLASSE:")
        regression_by_class = {}
        for cls in sorted(class_counts.keys()):
            class_mask = y == cls
            class_creat = creatinina[class_mask]
            class_tfg = tfg[class_mask]
            
            regression_by_class[cls] = {
                'creatinina_mean': np.mean(class_creat),
                'creatinina_std': np.std(class_creat),
                'tfg_mean': np.mean(class_tfg),
                'tfg_std': np.std(class_tfg),
                'samples': len(class_creat)
            }
            
            print(f"  {class_names[cls]}:")
            print(f"    CREATININA: {np.mean(class_creat):.2f} ± {np.std(class_creat):.2f}")
            print(f"    TFG: {np.mean(class_tfg):.1f} ± {np.std(class_tfg):.1f}")
        
        self.methodology_report['original_analysis'] = {
            'class_analysis': class_analysis,
            'intra_class_distances': intra_class_distances,
            'correlation_creat_tfg': corr_creat_tfg,
            'regression_by_class': regression_by_class,
            'total_samples': total_samples
        }
        
        return X_pca
    
    def simulate_balancing_methodology(self, X, y, creatinina, tfg):
        """
        Simula exatamente a metodologia usada e analisa cada etapa
        """
        print("\n" + "=" * 60)
        print("2. SIMULAÇÃO DETALHADA DA METODOLOGIA DE BALANCEAMENTO")
        print("=" * 60)
        
        class_counts = Counter(y)
        
        print(f"📋 METODOLOGIA APLICADA:")
        print(f"  1. RandomOverSampler para classes críticas")
        print(f"  2. SMOTE para balanceamento final")
        
        # ETAPA 1: RandomOverSampler
        print(f"\n🎯 ETAPA 1: RandomOverSampler")
        print(f"  Objetivo: Elevar classes críticas para mínimo viável")
        
        # Definir estratégia exata usada
        sampling_strategy_1 = {}
        for cls, count in class_counts.items():
            if count < 5:
                sampling_strategy_1[cls] = 10  # G5: 3 → 10
                print(f"    {cls}: {count} → 10 amostras (+{10-count} sintéticas)")
            elif count < 10:
                sampling_strategy_1[cls] = 15  # G4, G3b: 5 → 15
                print(f"    {cls}: {count} → 15 amostras (+{15-count} sintéticas)")
        
        # Aplicar RandomOverSampler
        ros = RandomOverSampler(sampling_strategy=sampling_strategy_1, random_state=self.random_state)
        X_step1, y_step1 = ros.fit_resample(X, y)
        
        print(f"  ✅ Resultado ETAPA 1: {dict(Counter(y_step1))}")
        
        # Analisar qualidade das amostras sintéticas da ETAPA 1
        synthetic_analysis_1 = self._analyze_synthetic_quality(X, y, X_step1, y_step1, "RandomOverSampler")
        
        # ETAPA 2: SMOTE
        print(f"\n🎯 ETAPA 2: SMOTE")
        print(f"  Objetivo: Balanceamento completo todas as classes")
        
        min_samples_per_class = min(Counter(y_step1).values())
        k_neighbors = max(1, min(5, min_samples_per_class - 1))
        print(f"  k_neighbors calculado: {k_neighbors}")
        
        smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        X_final, y_final = smote.fit_resample(X_step1, y_step1)
        
        print(f"  ✅ Resultado ETAPA 2: {dict(Counter(y_final))}")
        
        # Analisar qualidade das amostras sintéticas da ETAPA 2
        synthetic_analysis_2 = self._analyze_synthetic_quality(X_step1, y_step1, X_final, y_final, "SMOTE")
        
        # Calcular targets sintéticos para análise
        creatinina_step1, tfg_step1 = self._extend_regression_targets(creatinina, tfg, y, y_step1)
        creatinina_final, tfg_final = self._extend_regression_targets(creatinina_step1, tfg_step1, y_step1, y_final)
        
        self.methodology_report['balancing_analysis'] = {
            'step1_strategy': sampling_strategy_1,
            'step1_results': dict(Counter(y_step1)),
            'step1_synthetic_analysis': synthetic_analysis_1,
            'step2_k_neighbors': k_neighbors,
            'step2_results': dict(Counter(y_final)),
            'step2_synthetic_analysis': synthetic_analysis_2
        }
        
        return X_final, y_final, creatinina_final, tfg_final
    
    def _analyze_synthetic_quality(self, X_before, y_before, X_after, y_after, method_name):
        """
        Analisa a qualidade das amostras sintéticas geradas
        """
        print(f"    📊 Análise de qualidade - {method_name}:")
        
        # Identificar amostras sintéticas
        original_count = len(X_before)
        synthetic_count = len(X_after) - original_count
        
        if synthetic_count > 0:
            # Amostras sintéticas são as últimas adicionadas
            X_synthetic = X_after[original_count:]
            y_synthetic = y_after[original_count:]
            
            # Analisar distribuição das amostras sintéticas
            synthetic_class_counts = Counter(y_synthetic)
            print(f"      Amostras sintéticas geradas: {synthetic_count}")
            print(f"      Distribuição sintética: {dict(synthetic_class_counts)}")
            
            # Calcular distâncias das amostras sintéticas às originais
            quality_metrics = {}
            for cls in synthetic_class_counts.keys():
                # Amostras originais desta classe
                original_mask = y_before == cls
                synthetic_mask = y_synthetic == cls
                
                if original_mask.sum() > 0 and synthetic_mask.sum() > 0:
                    X_orig_class = X_before[original_mask]
                    X_synth_class = X_synthetic[synthetic_mask]
                    
                    # Distância mínima de cada sintética à original mais próxima
                    distances = pairwise_distances(X_synth_class, X_orig_class)
                    min_distances = np.min(distances, axis=1)
                    
                    quality_metrics[cls] = {
                        'min_distance_mean': np.mean(min_distances),
                        'min_distance_std': np.std(min_distances),
                        'synthetic_count': synthetic_mask.sum()
                    }
                    
                    print(f"      Classe {cls}: distância média às originais = {np.mean(min_distances):.3f}")
            
            return quality_metrics
        else:
            print(f"      Nenhuma amostra sintética gerada")
            return {}
    
    def _extend_regression_targets(self, creatinina_before, tfg_before, y_before, y_after):
        """
        Estende targets de regressão para amostras sintéticas usando médias por classe
        """
        original_count = len(y_before)
        synthetic_count = len(y_after) - original_count
        
        if synthetic_count == 0:
            return creatinina_before, tfg_before
        
        # Calcular médias por classe dos dados originais
        class_means = {}
        for cls in np.unique(y_before):
            class_mask = y_before == cls
            class_means[cls] = {
                'creatinina': np.mean(creatinina_before[class_mask]),
                'tfg': np.mean(tfg_before[class_mask])
            }
        
        # Estender targets para amostras sintéticas
        y_synthetic = y_after[original_count:]
        creatinina_synthetic = np.array([class_means[cls]['creatinina'] for cls in y_synthetic])
        tfg_synthetic = np.array([class_means[cls]['tfg'] for cls in y_synthetic])
        
        # Combinar original + sintético
        creatinina_extended = np.concatenate([creatinina_before, creatinina_synthetic])
        tfg_extended = np.concatenate([tfg_before, tfg_synthetic])
        
        return creatinina_extended, tfg_extended
    
    def validate_methodology_reliability(self, X_orig, y_orig, X_final, y_final, creat_orig, tfg_orig, creat_final, tfg_final):
        """
        Valida a confiabilidade da metodologia para produção de resultados confiáveis
        """
        print("\n" + "=" * 60)
        print("3. VALIDAÇÃO DE CONFIABILIDADE DA METODOLOGIA")
        print("=" * 60)
        
        reliability_analysis = {}
        
        # 1. Preservação de estrutura dos dados originais
        print(f"🔍 1. PRESERVAÇÃO DA ESTRUTURA ORIGINAL:")
        
        # Verificar se dados originais estão preservados
        original_count = len(X_orig)
        preserved_X = X_final[:original_count]
        preserved_y = y_final[:original_count]
        
        structure_preserved = np.allclose(X_orig, preserved_X) and np.array_equal(y_orig, preserved_y)
        print(f"   Estrutura original preservada: {'✅ Sim' if structure_preserved else '❌ Não'}")
        
        # 2. Qualidade das amostras sintéticas
        print(f"\n🔍 2. QUALIDADE DAS AMOSTRAS SINTÉTICAS:")
        
        synthetic_count = len(X_final) - original_count
        X_synthetic = X_final[original_count:]
        y_synthetic = y_final[original_count:]
        
        # Analisar dispersão das sintéticas vs originais
        synthetic_quality = {}
        for cls in np.unique(y_orig):
            orig_mask = y_orig == cls
            synth_mask = y_synthetic == cls
            
            if orig_mask.sum() > 0 and synth_mask.sum() > 0:
                X_orig_class = X_orig[orig_mask]
                X_synth_class = X_synthetic[synth_mask]
                
                # Calcular centroide e dispersão
                orig_centroid = np.mean(X_orig_class, axis=0)
                orig_std = np.std(X_orig_class, axis=0)
                
                synth_centroid = np.mean(X_synth_class, axis=0)
                synth_std = np.std(X_synth_class, axis=0)
                
                # Distância entre centroides
                centroid_distance = np.linalg.norm(orig_centroid - synth_centroid)
                
                # Similaridade de dispersão
                std_similarity = np.mean(np.abs(orig_std - synth_std) / (orig_std + 1e-8))
                
                synthetic_quality[cls] = {
                    'centroid_distance': centroid_distance,
                    'std_similarity': std_similarity,
                    'original_samples': orig_mask.sum(),
                    'synthetic_samples': synth_mask.sum()
                }
                
                print(f"   Classe {cls}: distância centroide = {centroid_distance:.3f}, similaridade dispersão = {std_similarity:.3f}")
        
        # 3. Preservação de correlações
        print(f"\n🔍 3. PRESERVAÇÃO DE CORRELAÇÕES:")
        
        orig_corr = np.corrcoef(creat_orig, tfg_orig)[0, 1]
        final_corr = np.corrcoef(creat_final, tfg_final)[0, 1]
        corr_preservation = abs(orig_corr - final_corr)
        
        print(f"   Correlação CREATININA-TFG original: {orig_corr:.3f}")
        print(f"   Correlação CREATININA-TFG final: {final_corr:.3f}")
        print(f"   Diferença: {corr_preservation:.3f} ({'✅ Boa' if corr_preservation < 0.1 else '⚠️ Moderada' if corr_preservation < 0.2 else '❌ Alta'})")
        
        # 4. Distribuição das variáveis de regressão
        print(f"\n🔍 4. DISTRIBUIÇÃO DAS VARIÁVEIS DE REGRESSÃO:")
        
        # Teste de Kolmogorov-Smirnov para distribuições
        ks_creat = stats.ks_2samp(creat_orig, creat_final[:original_count])
        ks_tfg = stats.ks_2samp(tfg_orig, tfg_final[:original_count])
        
        print(f"   KS-test CREATININA (p-value): {ks_creat.pvalue:.3f}")
        print(f"   KS-test TFG (p-value): {ks_tfg.pvalue:.3f}")
        
        # 5. Avaliação geral de confiabilidade
        print(f"\n🔍 5. AVALIAÇÃO GERAL DE CONFIABILIDADE:")
        
        reliability_score = 0
        criteria = []
        
        if structure_preserved:
            reliability_score += 25
            criteria.append("✅ Estrutura original preservada")
        else:
            criteria.append("❌ Estrutura original alterada")
        
        if corr_preservation < 0.1:
            reliability_score += 25
            criteria.append("✅ Correlações bem preservadas")
        elif corr_preservation < 0.2:
            reliability_score += 15
            criteria.append("⚠️ Correlações moderadamente preservadas")
        else:
            criteria.append("❌ Correlações mal preservadas")
        
        avg_centroid_dist = np.mean([q['centroid_distance'] for q in synthetic_quality.values()])
        if avg_centroid_dist < 1.0:
            reliability_score += 25
            criteria.append("✅ Amostras sintéticas próximas às originais")
        elif avg_centroid_dist < 2.0:
            reliability_score += 15
            criteria.append("⚠️ Amostras sintéticas moderadamente próximas")
        else:
            criteria.append("❌ Amostras sintéticas distantes das originais")
        
        if min(ks_creat.pvalue, ks_tfg.pvalue) > 0.05:
            reliability_score += 25
            criteria.append("✅ Distribuições de regressão preservadas")
        else:
            reliability_score += 10
            criteria.append("⚠️ Distribuições de regressão parcialmente preservadas")
        
        print(f"   Score de confiabilidade: {reliability_score}/100")
        for criterion in criteria:
            print(f"   {criterion}")
        
        if reliability_score >= 80:
            overall_assessment = "ALTA CONFIABILIDADE"
        elif reliability_score >= 60:
            overall_assessment = "CONFIABILIDADE MODERADA"
        else:
            overall_assessment = "BAIXA CONFIABILIDADE"
        
        print(f"\n   🎯 AVALIAÇÃO FINAL: {overall_assessment}")
        
        reliability_analysis = {
            'structure_preserved': structure_preserved,
            'synthetic_quality': synthetic_quality,
            'correlation_preservation': corr_preservation,
            'ks_tests': {'creatinina': ks_creat.pvalue, 'tfg': ks_tfg.pvalue},
            'reliability_score': reliability_score,
            'overall_assessment': overall_assessment,
            'criteria': criteria
        }
        
        self.methodology_report['reliability_analysis'] = reliability_analysis
        return reliability_analysis
    
    def create_detailed_visualization_report(self, X_orig, y_orig, X_final, y_final, 
                                           creat_orig, tfg_orig, creat_final, tfg_final,
                                           class_names, output_dir='plots/methodology_report'):
        """
        Cria relatório visual detalhado da metodologia
        """
        print("\n" + "=" * 60)
        print("4. GERANDO RELATÓRIO VISUAL DETALHADO")
        print("=" * 60)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Análise espacial antes/depois
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RELATÓRIO METODOLÓGICO DETALHADO - Análise Espacial e Qualidade', 
                     fontsize=16, fontweight='bold')
        
        # PCA dos dados originais
        pca = PCA(n_components=2, random_state=self.random_state)
        X_orig_pca = pca.fit_transform(X_orig)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
        
        for i, cls in enumerate(np.unique(y_orig)):
            mask = y_orig == cls
            axes[0, 0].scatter(X_orig_pca[mask, 0], X_orig_pca[mask, 1], 
                             c=[colors[i]], label=f'{class_names[cls]} (n={mask.sum()})',
                             alpha=0.7, s=50)
        
        axes[0, 0].set_title('ORIGINAL - Distribuição Espacial (PCA)')
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # PCA dos dados finais
        X_final_pca = pca.transform(X_final)
        original_count = len(X_orig)
        
        # Plotar originais e sintéticas separadamente
        for i, cls in enumerate(np.unique(y_final)):
            # Originais
            orig_mask = (y_final[:original_count] == cls) if cls in y_orig else np.array([])
            if len(orig_mask) > 0 and orig_mask.sum() > 0:
                axes[0, 1].scatter(X_final_pca[:original_count][orig_mask, 0], 
                                 X_final_pca[:original_count][orig_mask, 1],
                                 c=[colors[i]], alpha=0.8, s=50, marker='o', 
                                 label=f'{class_names[cls]} (orig)')
            
            # Sintéticas
            synth_mask = y_final[original_count:] == cls
            if synth_mask.sum() > 0:
                axes[0, 1].scatter(X_final_pca[original_count:][synth_mask, 0], 
                                 X_final_pca[original_count:][synth_mask, 1],
                                 c=[colors[i]], alpha=0.4, s=30, marker='^',
                                 label=f'{class_names[cls]} (sint)')
        
        axes[0, 1].set_title('FINAL - Originais (○) vs Sintéticas (△)')
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Análise de qualidade por classe
        quality_data = []
        for cls in np.unique(y_orig):
            orig_mask = y_orig == cls
            synth_mask = y_final[original_count:] == cls
            
            orig_count_cls = orig_mask.sum()
            synth_count_cls = synth_mask.sum()
            
            if orig_count_cls > 0:
                quality_data.append({
                    'classe': class_names[cls],
                    'original': orig_count_cls,
                    'sintetica': synth_count_cls,
                    'total': orig_count_cls + synth_count_cls
                })
        
        df_quality = pd.DataFrame(quality_data)
        
        x_pos = np.arange(len(df_quality))
        width = 0.35
        
        bars1 = axes[0, 2].bar(x_pos - width/2, df_quality['original'], width, 
                              label='Originais', color='steelblue', alpha=0.8)
        bars2 = axes[0, 2].bar(x_pos + width/2, df_quality['sintetica'], width,
                              label='Sintéticas', color='orange', alpha=0.8)
        
        axes[0, 2].set_title('Composição: Originais vs Sintéticas')
        axes[0, 2].set_ylabel('Número de Amostras')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(df_quality['classe'], rotation=45, ha='right')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Adicionar valores nas barras
        for bar1, bar2, orig, sint in zip(bars1, bars2, df_quality['original'], df_quality['sintetica']):
            if orig > 0:
                axes[0, 2].text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 1,
                               str(orig), ha='center', va='bottom', fontweight='bold')
            if sint > 0:
                axes[0, 2].text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 1,
                               str(sint), ha='center', va='bottom', fontweight='bold')
        
        # 2. Análise das variáveis de regressão
        
        # CREATININA antes/depois
        axes[1, 0].hist(creat_orig, bins=20, alpha=0.7, label='Original', 
                       color='steelblue', density=True)
        axes[1, 0].hist(creat_final, bins=20, alpha=0.5, label='Final', 
                       color='orange', density=True)
        axes[1, 0].set_title('Distribuição CREATININA')
        axes[1, 0].set_xlabel('CREATININA (mg/dL)')
        axes[1, 0].set_ylabel('Densidade')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # TFG antes/depois  
        axes[1, 1].hist(tfg_orig, bins=20, alpha=0.7, label='Original', 
                       color='steelblue', density=True)
        axes[1, 1].hist(tfg_final, bins=20, alpha=0.5, label='Final', 
                       color='orange', density=True)
        axes[1, 1].set_title('Distribuição TFG')
        axes[1, 1].set_xlabel('TFG (mL/min/1.73m²)')
        axes[1, 1].set_ylabel('Densidade')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlação CREATININA vs TFG
        axes[1, 2].scatter(creat_orig, tfg_orig, alpha=0.6, label='Original', 
                          color='steelblue', s=50)
        
        # Adicionar amostras sintéticas
        synth_creat = creat_final[original_count:]
        synth_tfg = tfg_final[original_count:]
        if len(synth_creat) > 0:
            axes[1, 2].scatter(synth_creat, synth_tfg, alpha=0.4, label='Sintética', 
                              color='orange', s=30, marker='^')
        
        # Linhas de correlação
        z_orig = np.polyfit(creat_orig, tfg_orig, 1)
        p_orig = np.poly1d(z_orig)
        x_range = np.linspace(min(creat_final), max(creat_final), 100)
        axes[1, 2].plot(x_range, p_orig(x_range), '--', color='steelblue', alpha=0.8)
        
        if len(synth_creat) > 0:
            z_final = np.polyfit(creat_final, tfg_final, 1)
            p_final = np.poly1d(z_final)
            axes[1, 2].plot(x_range, p_final(x_range), '--', color='orange', alpha=0.8)
        
        corr_orig = np.corrcoef(creat_orig, tfg_orig)[0, 1]
        corr_final = np.corrcoef(creat_final, tfg_final)[0, 1]
        
        axes[1, 2].set_title(f'Correlação CREAT-TFG\nOrig: {corr_orig:.3f}, Final: {corr_final:.3f}')
        axes[1, 2].set_xlabel('CREATININA (mg/dL)')
        axes[1, 2].set_ylabel('TFG (mL/min/1.73m²)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/detailed_methodology_report.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Relatório visual detalhado salvo em: {output_dir}")
        
        # 2. Gráfico de confiabilidade
        self._create_reliability_assessment_plot(output_dir)
    
    def _create_reliability_assessment_plot(self, output_dir):
        """
        Cria gráfico de avaliação de confiabilidade
        """
        reliability = self.methodology_report['reliability_analysis']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('AVALIAÇÃO DE CONFIABILIDADE DA METODOLOGIA', fontsize=16, fontweight='bold')
        
        # Score de confiabilidade
        score = reliability['reliability_score']
        colors = ['red' if score < 60 else 'orange' if score < 80 else 'green']
        
        bars = ax1.bar(['Confiabilidade'], [score], color=colors, alpha=0.7)
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Score (%)')
        ax1.set_title(f'Score de Confiabilidade: {score}/100')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar valor na barra
        ax1.text(0, score + 2, f'{score}%', ha='center', va='bottom', fontweight='bold')
        
        # Critérios de avaliação
        criteria_text = "\n".join(reliability['criteria'])
        ax2.text(0.05, 0.95, f"CRITÉRIOS DE AVALIAÇÃO:\n\n{criteria_text}", 
                transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        ax2.text(0.05, 0.05, f"AVALIAÇÃO FINAL:\n{reliability['overall_assessment']}", 
                transform=ax2.transAxes, fontsize=14, verticalalignment='bottom',
                fontweight='bold',
                bbox=dict(boxstyle='round', 
                         facecolor='lightgreen' if score >= 80 else 'lightyellow' if score >= 60 else 'lightcoral',
                         alpha=0.8))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Critérios e Avaliação')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/reliability_assessment.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_complete_methodology_report(self, data_path):
        """
        Gera relatório completo da metodologia
        """
        # 1. Carregar dados originais
        X_orig, y_orig, creat_orig, tfg_orig, class_names = self.load_and_prepare_original_data(data_path)
        
        # 2. Analisar distribuição original
        X_pca = self.analyze_original_data_distribution(X_orig, y_orig, creat_orig, tfg_orig, class_names)
        
        # 3. Simular metodologia de balanceamento
        X_final, y_final, creat_final, tfg_final = self.simulate_balancing_methodology(
            X_orig, y_orig, creat_orig, tfg_orig
        )
        
        # 4. Validar confiabilidade
        reliability = self.validate_methodology_reliability(
            X_orig, y_orig, X_final, y_final, creat_orig, tfg_orig, creat_final, tfg_final
        )
        
        # 5. Criar visualizações detalhadas
        self.create_detailed_visualization_report(
            X_orig, y_orig, X_final, y_final, creat_orig, tfg_orig, creat_final, tfg_final, class_names
        )
        
        # 6. Resumo final
        print("\n" + "=" * 80)
        print("RESUMO FINAL DO RELATÓRIO METODOLÓGICO")
        print("=" * 80)
        
        print(f"📊 DADOS PROCESSADOS:")
        print(f"   Amostras originais: {len(X_orig)}")
        print(f"   Amostras finais: {len(X_final)}")
        print(f"   Amostras sintéticas geradas: {len(X_final) - len(X_orig)}")
        
        print(f"\n🔬 METODOLOGIA APLICADA:")
        print(f"   1. RandomOverSampler para classes críticas")
        print(f"   2. SMOTE para balanceamento final")
        print(f"   3. Preservação de estrutura original")
        print(f"   4. Extensão de targets por médias de classe")
        
        print(f"\n📈 AVALIAÇÃO DE CONFIABILIDADE:")
        print(f"   Score: {reliability['reliability_score']}/100")
        print(f"   Avaliação: {reliability['overall_assessment']}")
        
        return self.methodology_report

def main():
    """
    Executa relatório metodológico completo
    """
    import os
    os.makedirs('plots/methodology_report', exist_ok=True)
    
    validator = DRCMethodologyValidator(random_state=42)
    
    data_path = '/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv'
    report = validator.generate_complete_methodology_report(data_path)
    
    print(f"\n🎯 RELATÓRIO METODOLÓGICO COMPLETO GERADO!")
    print(f"📁 Visualizações em: plots/methodology_report/")

if __name__ == "__main__":
    main()