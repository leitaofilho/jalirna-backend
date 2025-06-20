#!/usr/bin/env python3
"""
Modelo de Produção para Predição DRC
Multi-task learning: CREATININA (regressão), TFG (regressão), TFG_Classification (classificação)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, mean_squared_error, r2_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

class DRCDataset(Dataset):
    """Dataset customizado para dados DRC multi-task"""
    
    def __init__(self, X, y_creatinina, y_tfg, y_classification):
        self.X = torch.FloatTensor(X)
        self.y_creatinina = torch.FloatTensor(y_creatinina)
        self.y_tfg = torch.FloatTensor(y_tfg)
        self.y_classification = torch.LongTensor(y_classification)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            self.X[idx], 
            self.y_creatinina[idx], 
            self.y_tfg[idx], 
            self.y_classification[idx]
        )

class FocalLoss(nn.Module):
    """Focal Loss para classes desbalanceadas"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DRCMultiTaskModel(nn.Module):
    """
    Modelo Multi-Task para predição DRC
    - Shared layers para extração de features
    - Cabeças específicas para cada tarefa
    """
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], n_classes=6, dropout=0.3):
        super(DRCMultiTaskModel, self).__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        self.creatinina_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 1)
        )
        
        self.tfg_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(32, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Task-specific predictions
        creatinina_raw = self.creatinina_head(shared_features).squeeze()
        tfg_raw = self.tfg_head(shared_features).squeeze()
        class_pred = self.classification_head(shared_features)
        
        # Aplicar constraints clínicos muito flexíveis para permitir aprendizado
        creatinina_pred = torch.clamp(creatinina_raw, min=0.1, max=25.0)  # Range muito amplo
        tfg_pred = tfg_raw  # SEM CLAMP para permitir aprendizado natural do TFG
        
        return creatinina_pred, tfg_pred, class_pred

class DRCTrainer:
    """Treinador para modelo DRC multi-task"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions com melhorias médicas
        self.mse_loss = nn.MSELoss()
        
        # Focal Loss com pesos específicos para classes médicas
        # G2 (classe 1) precisa mais atenção baseado na avaliação
        class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.5, 1.5]).to(device)  # Peso maior para G2, G4, G5
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.5)  # Gamma maior para focar em casos difíceis
        
        # Optimizer com learning rate menor para estabilidade
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.0005,  # LR menor para convergência mais estável
            weight_decay=0.02  # Regularização maior
        )
        
        # Scheduler mais conservador
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Redução mais agressiva
            patience=8,  # Paciência menor
            min_lr=1e-6  # LR mínimo
        )
        
        # Training history
        self.train_history = {
            'total_loss': [],
            'creatinina_loss': [],
            'tfg_loss': [],
            'classification_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, train_loader):
        """Treina uma época"""
        self.model.train()
        total_loss = 0
        creatinina_loss_sum = 0
        tfg_loss_sum = 0
        class_loss_sum = 0
        
        for batch_idx, (X, y_creat, y_tfg, y_class) in enumerate(train_loader):
            X = X.to(self.device)
            y_creat = y_creat.to(self.device)
            y_tfg = y_tfg.to(self.device)
            y_class = y_class.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            creat_pred, tfg_pred, class_pred = self.model(X)
            
            # Losses com pesos otimizados para medicina
            creat_loss = self.mse_loss(creat_pred, y_creat)
            tfg_loss = self.mse_loss(tfg_pred, y_tfg)
            class_loss = self.focal_loss(class_pred, y_class)
            
            # Consistency loss para melhorar relação CREAT-TFG
            consistency_loss = self._compute_consistency_loss(creat_pred, tfg_pred, y_class)
            
            # Normalizar losses antes de combinar (importante para escalas diferentes)
            creat_loss_norm = creat_loss / 2.0   # Normalizar CREATININA menos agressivamente
            tfg_loss_norm = tfg_loss / 100.0     # Normalizar TFG muito menos agressivamente
            
            # Loss total com mais peso para TFG para forçar aprendizado
            loss = 0.2 * creat_loss_norm + 0.6 * tfg_loss_norm + 0.2 * class_loss + 0.0 * consistency_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Acumular losses
            total_loss += loss.item()
            creatinina_loss_sum += creat_loss.item()
            tfg_loss_sum += tfg_loss.item()
            class_loss_sum += class_loss.item()
        
        # Médias
        avg_total_loss = total_loss / len(train_loader)
        avg_creat_loss = creatinina_loss_sum / len(train_loader)
        avg_tfg_loss = tfg_loss_sum / len(train_loader)
        avg_class_loss = class_loss_sum / len(train_loader)
        
        return avg_total_loss, avg_creat_loss, avg_tfg_loss, avg_class_loss
    
    def _compute_consistency_loss(self, creat_pred, tfg_pred, y_class):
        """
        Computa loss de consistência clínica entre CREATININA, TFG e classificação
        """
        # Ranges esperados por classe (mais flexíveis)
        expected_ranges = {
            0: {'creat': 0.8, 'tfg': 105.0},   # G1
            1: {'creat': 0.95, 'tfg': 70.0},   # G2
            2: {'creat': 1.25, 'tfg': 50.0},   # G3a
            3: {'creat': 1.9, 'tfg': 35.0},    # G3b
            4: {'creat': 2.7, 'tfg': 20.0},    # G4
            5: {'creat': 7.2, 'tfg': 10.0}     # G5 - valor menos restritivo
        }
        
        consistency_loss = 0.0
        
        for cls in range(6):
            # Máscara para amostras desta classe
            mask = (y_class == cls)
            
            if mask.sum() > 0:
                # Valores esperados para esta classe
                expected_creat = expected_ranges[cls]['creat']
                expected_tfg = expected_ranges[cls]['tfg']
                
                # Predições para esta classe
                class_creat_pred = creat_pred[mask]
                class_tfg_pred = tfg_pred[mask]
                
                # Loss de consistência (distância aos valores esperados)
                creat_consistency = F.mse_loss(
                    class_creat_pred, 
                    torch.full_like(class_creat_pred, expected_creat)
                )
                tfg_consistency = F.mse_loss(
                    class_tfg_pred, 
                    torch.full_like(class_tfg_pred, expected_tfg)
                )
                
                consistency_loss += (creat_consistency + tfg_consistency) / mask.sum()
        
        return consistency_loss / 6  # Média sobre todas as classes
    
    def validate(self, val_loader):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X, y_creat, y_tfg, y_class in val_loader:
                X = X.to(self.device)
                y_creat = y_creat.to(self.device)
                y_tfg = y_tfg.to(self.device)
                y_class = y_class.to(self.device)
                
                creat_pred, tfg_pred, class_pred = self.model(X)
                
                creat_loss = self.mse_loss(creat_pred, y_creat)
                tfg_loss = self.mse_loss(tfg_pred, y_tfg)
                class_loss = self.focal_loss(class_pred, y_class)
                consistency_loss = self._compute_consistency_loss(creat_pred, tfg_pred, y_class)
                
                # Mesmos pesos do treinamento (normalizados)
                creat_loss_norm = creat_loss / 10.0
                tfg_loss_norm = tfg_loss / 1000.0
                loss = 0.3 * creat_loss_norm + 0.3 * tfg_loss_norm + 0.35 * class_loss + 0.05 * consistency_loss
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, X):
        """Faz predições"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            creat_pred, tfg_pred, class_pred = self.model(X_tensor)
            class_probs = F.softmax(class_pred, dim=1)
            class_pred_labels = torch.argmax(class_probs, dim=1)
        
        return (
            creat_pred.cpu().numpy(),
            tfg_pred.cpu().numpy(),
            class_pred_labels.cpu().numpy(),
            class_probs.cpu().numpy()
        )
    
    def train_model(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """Treina o modelo completo"""
        print(f"🚀 Iniciando treinamento por {epochs} épocas...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Treinar
            train_loss, creat_loss, tfg_loss, class_loss = self.train_epoch(train_loader)
            
            # Validar
            val_loss = self.validate(val_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Salvar histórico
            self.train_history['total_loss'].append(train_loss)
            self.train_history['creatinina_loss'].append(creat_loss)
            self.train_history['tfg_loss'].append(tfg_loss)
            self.train_history['classification_loss'].append(class_loss)
            self.train_history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salvar melhor modelo no diretório temporário
                torch.save(self.model.state_dict(), 'temp_best_model.pth')
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch+1}/{epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Creatinina: {creat_loss:.4f}")
                print(f"  TFG: {tfg_loss:.4f}")
                print(f"  Classificação: {class_loss:.4f}")
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping na época {epoch+1}")
                break
        
        # Carregar melhor modelo
        self.model.load_state_dict(torch.load('temp_best_model.pth'))
        # Limpar arquivo temporário
        import os
        if os.path.exists('temp_best_model.pth'):
            os.remove('temp_best_model.pth')
        print(f"✅ Treinamento concluído! Melhor val_loss: {best_val_loss:.4f}")
        
        return self.train_history

def evaluate_model(trainer, test_loader, class_names, output_dir='results'):
    """Avalia o modelo e gera relatórios"""
    print("📊 Avaliando modelo...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Coletar predições
    all_X, all_y_creat, all_y_tfg, all_y_class = [], [], [], []
    
    for X, y_creat, y_tfg, y_class in test_loader:
        all_X.append(X.numpy())
        all_y_creat.append(y_creat.numpy())
        all_y_tfg.append(y_tfg.numpy())
        all_y_class.append(y_class.numpy())
    
    X_test = np.vstack(all_X)
    y_creat_true = np.concatenate(all_y_creat)
    y_tfg_true = np.concatenate(all_y_tfg)
    y_class_true = np.concatenate(all_y_class)
    
    # Predições
    creat_pred, tfg_pred, class_pred, class_probs = trainer.predict(X_test)
    
    # Métricas de regressão
    creat_mse = mean_squared_error(y_creat_true, creat_pred)
    creat_r2 = r2_score(y_creat_true, creat_pred)
    
    tfg_mse = mean_squared_error(y_tfg_true, tfg_pred)
    tfg_r2 = r2_score(y_tfg_true, tfg_pred)
    
    # Métricas de classificação
    class_accuracy = balanced_accuracy_score(y_class_true, class_pred)
    class_report = classification_report(
        y_class_true, class_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    # Relatório final
    results = {
        'timestamp': datetime.now().isoformat(),
        'regression_metrics': {
            'creatinina': {'mse': creat_mse, 'r2': creat_r2},
            'tfg': {'mse': tfg_mse, 'r2': tfg_r2}
        },
        'classification_metrics': {
            'balanced_accuracy': class_accuracy,
            'detailed_report': class_report
        },
        'model_architecture': str(trainer.model),
        'training_history': trainer.train_history
    }
    
    # Salvar resultados
    import json
    with open(f'{output_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Imprimir resumo
    print(f"\n📈 RESULTADOS FINAIS:")
    print(f"  CREATININA - MSE: {creat_mse:.4f}, R²: {creat_r2:.4f}")
    print(f"  TFG - MSE: {tfg_mse:.4f}, R²: {tfg_r2:.4f}")
    print(f"  Classificação - Balanced Accuracy: {class_accuracy:.4f}")
    
    return results

def save_model_for_production(trainer, preprocessor, output_dir='model_production'):
    """Salva modelo e preprocessador para produção"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Detectar configuração atual do modelo automaticamente
    model = trainer.model
    
    # Extrair hidden dims da arquitetura atual
    hidden_dims = []
    for i in range(0, len(model.shared_layers), 4):  # Cada bloco tem 4 camadas (Linear, BatchNorm, ReLU, Dropout)
        if hasattr(model.shared_layers[i], 'out_features'):
            hidden_dims.append(model.shared_layers[i].out_features)
    
    # Salvar modelo PyTorch com configuração correta
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.shared_layers[0].in_features,
            'hidden_dims': hidden_dims,
            'n_classes': model.classification_head[-1].out_features,
            'dropout': 0.4  # Valor usado atualmente
        },
        'timestamp': datetime.now().isoformat()
    }, f'{output_dir}/drc_model.pth')
    
    # Salvar preprocessador
    joblib.dump(preprocessor, f'{output_dir}/drc_preprocessor.joblib')
    
    print(f"✅ Modelo salvo para produção em: {output_dir}")
    
    return True