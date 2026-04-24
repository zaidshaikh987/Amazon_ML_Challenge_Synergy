import pandas as pd
import numpy as np
import sys
import os
sys.path.append('src')
from features import create_feature_matrix

# Define paths
train_embeddings = os.path.join('embeddings', 'train')
test_embeddings = os.path.join('embeddings', 'test')

print(f"Looking for train embeddings at: {os.path.abspath(train_embeddings)}")
print(f"Looking for test embeddings at: {os.path.abspath(test_embeddings)}")

# Load data
train_df = pd.read_csv(os.path.join('student_resource', 'dataset', 'train.csv'))
test_df = pd.read_csv(os.path.join('student_resource', 'dataset', 'test.csv'))

try:
    # Call create_feature_matrix and handle its return value
    train_result = create_feature_matrix(train_df, train_embeddings)
    test_result = create_feature_matrix(test_df, test_embeddings)
    
    # Handle different return types
    X_train = train_result[0] if isinstance(train_result, tuple) else train_result
    X_test = test_result[0] if isinstance(test_result, tuple) else test_result
    
    # Print feature information
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    if hasattr(X_train, 'columns'):  # If it's a DataFrame
        # Compare feature names
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        print(f"\nFeatures in train but not in test: {train_features - test_features}")
        print(f"Features in test but not in train: {test_features - train_features}")
    
    # Check for NaN values if it's a DataFrame
    if hasattr(X_train, 'isna'):
        print("\nNaN values in train:", X_train.isna().sum().sum())
        print("NaN values in test:", X_test.isna().sum().sum())
    
except Exception as e:
    print(f"\nError: {str(e)}")
    print("\nPlease share the output of this command from your Python console:")
    print("""
from features import create_feature_matrix
import inspect
print(inspect.signature(create_feature_matrix))
    """)