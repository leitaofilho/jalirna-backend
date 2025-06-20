#!/usr/bin/env python3
"""
Pipeline de Pré-processamento Robusto para Dataset DRC
Implementação das melhores práticas para dados médicos desbalanceados
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

class DRCPreprocessor:
    """
    Pipeline de pré-processamento especializado para dados de DRC
    """
    
    def __init__(self, strategy='smote', random_state=42):
        self.strategy = strategy
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = None
        self.balancer = None
        self.feature_names = None
        self.target_encoder = None
        self.is_fitted = False
        
        # Definir features de entrada baseadas no DRCForm
        self.input_features = [
            'IDADE', 'SEXO', 'COR', 'COR 2', 'IMC', 'CC', 'RCQ',
            'PAS', 'PAD', 'Fuma?', 'Realiza exercício?', 'Bebe?', 'DM', 'HAS'
        ]
        
        # Variáveis alvo
        self.target_variables = ['CREATININA', 'TFG', 'TFG_Classification']
        
    def load_and_clean_data(self, file_path):
        """
        Carrega e limpa o dataset bruto
        """
        print("=== CARREGANDO E LIMPANDO DADOS ===")
        
        # Carregar dados
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
        print(f"Dataset original: {df.shape}")
        
        # Corrigir formatação de números decimais
        decimal_columns = ['C. PES', 'C.PANT', 'CREATININA', 'UREIA', 'HDL', 'LDL']
        for col in decimal_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(',', '.'), 
                    errors='coerce'
                )
                print(f"Corrigida formatação decimal: {col}")
        
        # Verificar e remover linhas com valores faltantes nas variáveis críticas
        critical_vars = [var for var in self.target_variables if var in df.columns]
        initial_rows = len(df)
        df = df.dropna(subset=critical_vars)
        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            print(f"Removidas {removed_rows} linhas com valores faltantes nas variáveis alvo")
        
        # Verificar disponibilidade das features de entrada
        available_features = [f for f in self.input_features if f in df.columns]
        missing_features = [f for f in self.input_features if f not in df.columns]
        
        print(f"Features disponíveis: {len(available_features)}/{len(self.input_features)}")
        if missing_features:
            print(f"Features faltantes: {missing_features}")
        
        self.input_features = available_features
        return df
    
    def prepare_features_and_targets(self, df):
        """
        Prepara features de entrada e variáveis alvo
        """
        print("\n=== PREPARANDO FEATURES E TARGETS ===")
        
        # Extrair features de entrada
        X = df[self.input_features].copy()
        
        # Preparar targets
        targets = {}
        
        # CREATININA (regressão)
        if 'CREATININA' in df.columns:
            targets['creatinina'] = pd.to_numeric(
                df['CREATININA'].astype(str).str.replace(',', '.'), 
                errors='coerce'
            )
            print(f"CREATININA: min={targets['creatinina'].min():.2f}, max={targets['creatinina'].max():.2f}")
        
        # TFG (regressão)
        if 'TFG' in df.columns:
            targets['tfg'] = df['TFG'].astype(float)
            print(f"TFG: min={targets['tfg'].min():.1f}, max={targets['tfg'].max():.1f}")
        
        # TFG_Classification (classificação)
        if 'TFG_Classification' in df.columns:
            targets['classification'] = df['TFG_Classification']
            print(f"TFG_Classification distribuição:")
            print(targets['classification'].value_counts().sort_index())
        
        return X, targets
    
    def encode_categorical_features(self, X, fit=True):
        """
        Codifica features categóricas
        """
        print("\n=== CODIFICANDO FEATURES CATEGÓRICAS ===")
        
        X_encoded = X.copy()
        categorical_features = X.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            if fit:
                le = LabelEncoder()
                X_encoded[feature] = le.fit_transform(X[feature].astype(str))
                self.label_encoders[feature] = le
                print(f"Codificado {feature}: {len(le.classes_)} classes")
            else:
                if feature in self.label_encoders:
                    # Para dados novos, usar transform com tratamento de categorias não vistas
                    le = self.label_encoders[feature]
                    # Mapear categorias não vistas para uma categoria 'unknown'
                    mask = X[feature].isin(le.classes_)
                    X_encoded[feature] = 0  # valor padrão
                    X_encoded.loc[mask, feature] = le.transform(X.loc[mask, feature])
                    
        return X_encoded
    
    def scale_features(self, X, fit=True):
        """
        Escala features numéricas usando RobustScaler (mais resistente a outliers)
        """
        print("\n=== ESCALANDO FEATURES ===")
        
        if fit:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
            print(f"Features escaladas usando RobustScaler")
        else:
            X_scaled = self.scaler.transform(X)
        
        # Manter nomes das colunas
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled
    
    def balance_data(self, X, y, fit=True):
        """
        Aplica técnicas de balanceamento de dados
        """
        print(f"\n=== BALANCEANDO DADOS ({self.strategy.upper()}) ===")
        
        if not fit:
            return X, y  # Não balancear dados de teste
        
        print("Distribuição original:")
        print(pd.Series(y).value_counts().sort_index())
        
        # Escolher estratégia de balanceamento
        if self.strategy == 'smote':
            self.balancer = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.strategy == 'adasyn':
            self.balancer = ADASYN(random_state=self.random_state)
        elif self.strategy == 'smoteenn':
            self.balancer = SMOTEENN(random_state=self.random_state)
        elif self.strategy == 'smotetomek':
            self.balancer = SMOTETomek(random_state=self.random_state)
        elif self.strategy == 'undersample':
            self.balancer = RandomUnderSampler(random_state=self.random_state)
        else:
            print("Estratégia não reconhecida, usando SMOTE")
            self.balancer = SMOTE(random_state=self.random_state, k_neighbors=3)
        
        try:
            X_balanced, y_balanced = self.balancer.fit_resample(X, y)
            print("Distribuição após balanceamento:")
            print(pd.Series(y_balanced).value_counts().sort_index())
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"Erro no balanceamento: {e}")
            print("Retornando dados originais")
            return X, y
    
    def fit_transform(self, df):
        """
        Ajusta o preprocessador e transforma os dados
        """
        print("=== INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO ===")
        
        # 1. Carregar e limpar dados
        df_clean = self.load_and_clean_data(df) if isinstance(df, str) else df.copy()
        
        # 2. Preparar features e targets
        X, targets = self.prepare_features_and_targets(df_clean)
        
        # 3. Codificar features categóricas
        X_encoded = self.encode_categorical_features(X, fit=True)
        
        # 4. Escalar features
        X_scaled = self.scale_features(X_encoded, fit=True)
        
        # 5. Preparar dados de classificação balanceados
        processed_data = {'features': X_scaled}
        
        # Para classificação (TFG_Classification)
        if 'classification' in targets:
            # Codificar target de classificação
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
                y_class_encoded = self.target_encoder.fit_transform(targets['classification'])
            else:
                y_class_encoded = self.target_encoder.transform(targets['classification'])
            
            # Balancear dados para classificação
            X_class_balanced, y_class_balanced = self.balance_data(
                X_scaled, y_class_encoded, fit=True
            )
            
            processed_data['classification'] = {
                'X': X_class_balanced,
                'y': y_class_balanced,
                'target_names': self.target_encoder.classes_
            }
        
        # Para regressão (CREATININA e TFG) - não balancear
        if 'creatinina' in targets:
            processed_data['creatinina'] = {
                'X': X_scaled,
                'y': targets['creatinina'].values
            }
        
        if 'tfg' in targets:
            processed_data['tfg'] = {
                'X': X_scaled,
                'y': targets['tfg'].values
            }
        
        self.feature_names = X_scaled.columns.tolist()
        self.is_fitted = True
        
        print("\n=== PRÉ-PROCESSAMENTO CONCLUÍDO ===")
        print(f"Features finais: {len(self.feature_names)}")
        print(f"Tarefas preparadas: {list(processed_data.keys())}")
        
        return processed_data
    
    def transform(self, df):
        """
        Transforma novos dados usando o preprocessador ajustado
        """
        if not self.is_fitted:
            raise ValueError("Preprocessador não foi ajustado. Use fit_transform primeiro.")
        
        print("=== TRANSFORMANDO NOVOS DADOS ===")
        
        # Preparar features
        X = df[self.input_features].copy()
        
        # Codificar e escalar
        X_encoded = self.encode_categorical_features(X, fit=False)
        X_scaled = self.scale_features(X_encoded, fit=False)
        
        return X_scaled
    
    def save_preprocessor(self, filepath):
        """
        Salva o preprocessador ajustado
        """
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names,
            'input_features': self.input_features,
            'strategy': self.strategy,
            'is_fitted': self.is_fitted
        }, filepath)
        print(f"Preprocessador salvo em: {filepath}")
    
    def load_preprocessor(self, filepath):
        """
        Carrega preprocessador previamente ajustado
        """
        data = joblib.load(filepath)
        self.label_encoders = data['label_encoders']
        self.scaler = data['scaler']
        self.target_encoder = data['target_encoder']
        self.feature_names = data['feature_names']
        self.input_features = data['input_features']
        self.strategy = data['strategy']
        self.is_fitted = data['is_fitted']
        print(f"Preprocessador carregado de: {filepath}")

def main():
    """
    Exemplo de uso do preprocessador
    """
    # Inicializar preprocessador
    preprocessor = DRCPreprocessor(strategy='smote', random_state=42)
    
    # Carregar e processar dados
    data_path = '/Users/aiacontext/PycharmProjects/jaliRNA/dataset/dataset_drc_processed.csv'
    processed_data = preprocessor.fit_transform(data_path)
    
    # Salvar preprocessador
    preprocessor.save_preprocessor('/Users/aiacontext/PycharmProjects/jaliRNA/model/drc_preprocessor.joblib')
    
    # Exibir informações dos dados processados
    print("\n=== RESUMO DOS DADOS PROCESSADOS ===")
    for task, data in processed_data.items():
        if task == 'features':
            print(f"Features shape: {data.shape}")
        else:
            X, y = data['X'], data['y']
            print(f"{task.upper()}:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            if hasattr(y, 'dtype'):
                print(f"  y type: {y.dtype}")

if __name__ == "__main__":
    main()