# Save this as train_improved.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from features import create_feature_matrix

# Load and prepare data
train_df = pd.read_csv('student_resource/dataset/train.csv')
X, y = create_feature_matrix(train_df, 'embeddings/train/')

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train improved model
params = {
    'objective': 'regression',
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_child_samples': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

print("Training improved model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=50
)

# Save improved model
model.save_model('models/improved_lgb.txt')
print("Improved model saved to models/improved_lgb.txt")