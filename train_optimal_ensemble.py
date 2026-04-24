"""
Train Optimal Ensemble: 65% Neural Fusion + 35% LGB with Aggregates
Complete 5-Fold CV with real-time SMAPE calculation

This script:
1. Trains LightGBM with aggregates (5-fold CV)
2. Trains Neural Fusion model (5-fold CV)
3. Combines predictions with optimal weights
4. Shows real-time SMAPE for each fold
5. Generates final submission and OOF predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

def smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    return np.mean(200 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def create_aggregate_features(df, train_df=None, is_train=True):
    """Create aggregate features for LGB model"""
    agg_features = []
    
    if is_train:
        # Category aggregates
        for cat in ['product_category_code', 'product_subcategory_code', 'brand_id', 'material_group']:
            if cat in df.columns:
                agg_df = train_df.groupby(cat)['price'].agg(['mean', 'std', 'median']).reset_index()
                agg_df.columns = [cat, f'{cat}_price_mean', f'{cat}_price_std', f'{cat}_price_median']
                df = df.merge(agg_df, on=cat, how='left')
                agg_features.extend([f'{cat}_price_mean', f'{cat}_price_std', f'{cat}_price_median'])
    else:
        # For test, use precomputed aggregates
        for cat in ['product_category_code', 'product_subcategory_code', 'brand_id', 'material_group']:
            if cat in df.columns and train_df is not None:
                agg_df = train_df.groupby(cat)['price'].agg(['mean', 'std', 'median']).reset_index()
                agg_df.columns = [cat, f'{cat}_price_mean', f'{cat}_price_std', f'{cat}_price_median']
                df = df.merge(agg_df, on=cat, how='left')
                agg_features.extend([f'{cat}_price_mean', f'{cat}_price_std', f'{cat}_price_median'])
    
    return df, agg_features

class NeuralFusionModel(nn.Module):
    """Neural network for price prediction"""
    def __init__(self, input_dim):
        super(NeuralFusionModel, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.bn_input(x)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

print("="*80)
print("TRAINING OPTIMAL ENSEMBLE: 65% Neural + 35% LGB")
print("="*80)

# Load data
print("\n📁 Loading data...")
train_df = pd.read_csv("student_resource/dataset/train.csv")
test_df = pd.read_csv("student_resource/dataset/test.csv")

print(f"   Train: {len(train_df):,} samples")
print(f"   Test:  {len(test_df):,} samples")

# Drop text columns that can't be used directly
text_cols = ['catalog_content', 'image_link']
for col in text_cols:
    if col in train_df.columns:
        train_df = train_df.drop(columns=[col])
    if col in test_df.columns:
        test_df = test_df.drop(columns=[col])

print(f"\n🔧 Preparing features...")
print(f"   Dropped text columns: {text_cols}")

# Prepare categorical features
categorical_cols = ['product_category_code', 'product_subcategory_code', 'brand_id', 'material_group']
label_encoders = {}

for col in categorical_cols:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le

# Feature columns
feature_cols = [col for col in train_df.columns if col not in ['sample_id', 'price']]
print(f"   Feature count: {len(feature_cols)}")
print(f"   Categorical features: {len([c for c in categorical_cols if c in feature_cols])}")

# Initialize storage
lgb_oof_predictions = np.zeros(len(train_df))
lgb_test_predictions = np.zeros(len(test_df))
neural_oof_predictions = np.zeros(len(train_df))
neural_test_predictions = np.zeros(len(test_df))

# Setup 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*80)
print("STARTING 5-FOLD CROSS-VALIDATION")
print("="*80)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/5")
    print(f"{'='*80}")
    
    # Split data
    X_train_fold = train_df.iloc[train_idx]
    X_val_fold = train_df.iloc[val_idx]
    
    y_train = X_train_fold['price'].values
    y_val = X_val_fold['price'].values
    
    print(f"\n   Train: {len(X_train_fold):,} | Validation: {len(X_val_fold):,}")
    
    # ==========================================
    # 1. TRAIN LIGHTGBM WITH AGGREGATES
    # ==========================================
    print(f"\n   🌲 Training LightGBM...")
    
    # Add aggregate features
    X_train_agg, agg_cols = create_aggregate_features(X_train_fold.copy(), X_train_fold, is_train=True)
    X_val_agg, _ = create_aggregate_features(X_val_fold.copy(), X_train_fold, is_train=True)
    
    lgb_features = feature_cols + agg_cols
    
    # Train LGB
    train_data = lgb.Dataset(X_train_agg[lgb_features], y_train)
    val_data = lgb.Dataset(X_val_agg[lgb_features], y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    lgb_model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # Predict LGB
    lgb_val_pred = lgb_model.predict(X_val_agg[lgb_features])
    lgb_oof_predictions[val_idx] = lgb_val_pred
    
    lgb_fold_smape = smape(y_val, lgb_val_pred)
    print(f"      ✓ LGB SMAPE: {lgb_fold_smape:.2f}%")
    
    # Predict on test
    test_agg, _ = create_aggregate_features(test_df.copy(), X_train_fold, is_train=False)
    lgb_test_pred = lgb_model.predict(test_agg[lgb_features])
    lgb_test_predictions += lgb_test_pred / 5
    
    # ==========================================
    # 2. TRAIN NEURAL FUSION
    # ==========================================
    print(f"\n   🧠 Training Neural Fusion...")
    
    # Prepare neural data
    X_train_neural = X_train_fold[feature_cols].values.astype(np.float32)
    X_val_neural = X_val_fold[feature_cols].values.astype(np.float32)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_neural = scaler.fit_transform(X_train_neural)
    X_val_neural = scaler.transform(X_val_neural)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_neural)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_neural)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralFusionModel(X_train_neural.shape[1]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    batch_size = 256
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size].to(device)
            batch_y = y_train_tensor[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor.to(device)).cpu().numpy()
            val_loss = np.mean((val_pred - y_val.reshape(-1, 1))**2)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()
    
    # Predict Neural
    with torch.no_grad():
        neural_val_pred = model(X_val_tensor.to(device)).cpu().numpy().flatten()
    
    neural_oof_predictions[val_idx] = neural_val_pred
    neural_fold_smape = smape(y_val, neural_val_pred)
    print(f"      ✓ Neural SMAPE: {neural_fold_smape:.2f}%")
    
    # Predict on test
    X_test_neural = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    X_test_tensor = torch.FloatTensor(X_test_neural)
    with torch.no_grad():
        neural_test_pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    neural_test_predictions += neural_test_pred / 5
    
    # ==========================================
    # 3. ENSEMBLE PREDICTIONS
    # ==========================================
    print(f"\n   🎯 Ensemble (65% Neural + 35% LGB)...")
    
    ensemble_val_pred = 0.65 * neural_val_pred + 0.35 * lgb_val_pred
    ensemble_fold_smape = smape(y_val, ensemble_val_pred)
    
    print(f"      ✓ Ensemble SMAPE: {ensemble_fold_smape:.2f}%")
    
    # Store results
    fold_results.append({
        'fold': fold,
        'lgb_smape': lgb_fold_smape,
        'neural_smape': neural_fold_smape,
        'ensemble_smape': ensemble_fold_smape,
        'samples': len(y_val)
    })
    
    print(f"\n   📊 Fold {fold} Summary:")
    print(f"      LGB:      {lgb_fold_smape:.2f}%")
    print(f"      Neural:   {neural_fold_smape:.2f}%")
    print(f"      Ensemble: {ensemble_fold_smape:.2f}% ⭐")
    print(f"      Improvement: {min(lgb_fold_smape, neural_fold_smape) - ensemble_fold_smape:.2f} pts")

# ==========================================
# FINAL RESULTS
# ==========================================
print("\n" + "="*80)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*80)

# Calculate ensemble OOF
ensemble_oof_predictions = 0.65 * neural_oof_predictions + 0.35 * lgb_oof_predictions
overall_oof_smape = smape(train_df['price'].values, ensemble_oof_predictions)

print(f"\n{'Fold':<8} {'LGB':<12} {'Neural':<12} {'Ensemble':<12} {'Improvement':<12}")
print("-"*80)

for result in fold_results:
    improvement = min(result['lgb_smape'], result['neural_smape']) - result['ensemble_smape']
    print(f"{result['fold']:<8} {result['lgb_smape']:>6.2f}%     {result['neural_smape']:>6.2f}%     {result['ensemble_smape']:>6.2f}%     {improvement:>+5.2f} pts")

print("-"*80)

# Summary statistics
lgb_scores = [r['lgb_smape'] for r in fold_results]
neural_scores = [r['neural_smape'] for r in fold_results]
ensemble_scores = [r['ensemble_smape'] for r in fold_results]

print(f"\n📊 SUMMARY STATISTICS:")
print(f"\n   LGB Model:")
print(f"      Mean:  {np.mean(lgb_scores):.2f}%")
print(f"      Std:   {np.std(lgb_scores):.2f}%")

print(f"\n   Neural Model:")
print(f"      Mean:  {np.mean(neural_scores):.2f}%")
print(f"      Std:   {np.std(neural_scores):.2f}%")

print(f"\n   🏆 ENSEMBLE (65% Neural + 35% LGB):")
print(f"      Mean:  {np.mean(ensemble_scores):.2f}%")
print(f"      Std:   {np.std(ensemble_scores):.2f}%")
print(f"      Overall OOF: {overall_oof_smape:.2f}%")

if np.std(ensemble_scores) < 1.0:
    print(f"      ✅ Very consistent! (std < 1.0%)")
elif np.std(ensemble_scores) < 2.0:
    print(f"      ✅ Good consistency (std < 2.0%)")

# ==========================================
# SAVE SUBMISSIONS
# ==========================================
print("\n" + "="*80)
print("SAVING PREDICTIONS")
print("="*80)

# Calculate final ensemble test predictions
ensemble_test_predictions = 0.65 * neural_test_predictions + 0.35 * lgb_test_predictions

# Save test submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': ensemble_test_predictions
})
submission.to_csv('submissions/optimal_ensemble_submission.csv', index=False)
print(f"\n✅ Test submission saved: submissions/optimal_ensemble_submission.csv")
print(f"   Samples: {len(submission):,}")
print(f"   Mean price: ${submission['price'].mean():.2f}")

# Save OOF predictions
oof_df = pd.DataFrame({
    'sample_id': train_df['sample_id'],
    'price_actual': train_df['price'],
    'price_pred': ensemble_oof_predictions
})
oof_df.to_csv('submissions/optimal_ensemble_oof.csv', index=False)
print(f"\n✅ OOF predictions saved: submissions/optimal_ensemble_oof.csv")
print(f"   Samples: {len(oof_df):,}")
print(f"   OOF SMAPE: {overall_oof_smape:.2f}%")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

print(f"\n📈 PERFORMANCE:")
print(f"   Cross-Validation: {overall_oof_smape:.2f}%")
print(f"   Expected LB:      ~{overall_oof_smape + 0.5:.2f}%")
print(f"   Improvement:      {53.5 - (overall_oof_smape + 0.5):.2f} points from 53.5% baseline")

print(f"\n🔧 ENSEMBLE CONFIG:")
print(f"   Neural Fusion:     65%")
print(f"   LGB + Aggregates:  35%")

print(f"\n📁 FILES CREATED:")
print(f"   1. submissions/optimal_ensemble_submission.csv  (for leaderboard)")
print(f"   2. submissions/optimal_ensemble_oof.csv         (validation)")

print(f"\n🚀 Ready to submit!")

print("\n" + "="*80)
