import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pickle

print("="*70)
print("NEURAL FUSION MODEL FOR PRICE PREDICTION")
print("="*70)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ========== 1. Load Data ==========
print("\n1. Loading embeddings...")
X_img_train = np.load("embeddings/train/image_embeddings.npy")
X_txt_train = np.load("embeddings/train/text_embeddings.npy")
X_img_test = np.load("embeddings/test/image_embeddings.npy")
X_txt_test = np.load("embeddings/test/text_embeddings.npy")

# Concatenate
X_train = np.hstack([X_img_train, X_txt_train]).astype('float32')
X_test = np.hstack([X_img_test, X_txt_test]).astype('float32')

print(f"   Train embeddings: {X_train.shape}")
print(f"   Test embeddings: {X_test.shape}")

# Load prices
train_df = pd.read_csv("student_resource/dataset/train.csv")
test_df = pd.read_csv("student_resource/dataset/test.csv")
y_train = train_df['price'].values

# Log transform target
y_train_log = np.log1p(y_train).astype('float32')

print(f"   Train samples: {len(y_train)}")
print(f"   Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")

# ========== 2. Define Neural Network ==========
class PriceFusionNet(nn.Module):
    """
    Small MLP for price prediction from concatenated embeddings.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        super(PriceFusionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


# ========== 3. Dataset and DataLoader ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# ========== 4. SMAPE-like loss (differentiable approximation) ==========
def smape_loss(pred, target):
    """
    Differentiable approximation of SMAPE.
    """
    epsilon = 1e-8
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2 + epsilon
    return torch.mean(numerator / denominator) * 100


# ========== 5. Training Function ==========
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


# ========== 6. Cross-Validation Training ==========
print("\n2. Starting 5-fold cross-validation...")

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros((len(X_test), n_folds))

# Hyperparameters
batch_size = 256
learning_rate = 0.001
n_epochs = 100
early_stop_patience = 15

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}/{n_folds}")
    print(f"{'='*70}")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train_log[train_idx], y_train_log[val_idx]
    
    # Standardize features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = EmbeddingDataset(X_tr, y_tr)
    val_dataset = EmbeddingDataset(X_val, y_val)
    test_dataset = EmbeddingDataset(X_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = PriceFusionNet(
        input_dim=X_train.shape[1],
        hidden_dims=[512, 256, 128],
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Use SMAPE loss for training
    criterion = smape_loss
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"models/neural_fusion_fold{fold+1}.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(f"models/neural_fusion_fold{fold+1}.pt"))
    
    # OOF predictions
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        oof_pred_log = model(X_val_tensor).cpu().numpy()
        oof_predictions[val_idx] = np.expm1(oof_pred_log)
    
    # Test predictions
    model.eval()
    test_preds_fold = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            pred_log = model(X_batch).cpu().numpy()
            test_preds_fold.append(pred_log)
    
    test_predictions[:, fold] = np.expm1(np.concatenate(test_preds_fold))
    
    print(f"\nFold {fold+1} best validation loss: {best_val_loss:.4f}")

# ========== 7. Final Results ==========
print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)

# Compute OOF SMAPE
def compute_smape(y_true, y_pred):
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

oof_smape = compute_smape(y_train, oof_predictions)
print(f"\nOut-of-Fold SMAPE: {oof_smape:.2f}%")

# Average test predictions across folds
test_predictions_avg = test_predictions.mean(axis=1)

print(f"\nTest predictions statistics:")
print(f"  Mean: ${test_predictions_avg.mean():.2f}")
print(f"  Std: ${test_predictions_avg.std():.2f}")
print(f"  Min: ${test_predictions_avg.min():.2f}")
print(f"  Max: ${test_predictions_avg.max():.2f}")

# ========== 8. Save Predictions ==========
print("\n" + "="*70)
print("SAVING PREDICTIONS")
print("="*70)

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_predictions_avg
})

submission.to_csv("submissions/neural_fusion_predictions.csv", index=False)
print("\nSaved to: submissions/neural_fusion_predictions.csv")

# Also save OOF predictions for stacking
oof_df = pd.DataFrame({
    'sample_id': train_df['sample_id'],
    'price_actual': y_train,
    'price_pred': oof_predictions
})
oof_df.to_csv("submissions/neural_fusion_oof.csv", index=False)
print("Saved OOF to: submissions/neural_fusion_oof.csv")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print(f"\nNeural Fusion CV SMAPE: {oof_smape:.2f}%")
print(f"LightGBM test SMAPE: 54.00%")
print(f"k-NN CV SMAPE: 54.73%")
print("\nBlend all three models for best results!")
print("="*70)
