# Save this as check_importance_fixed.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model = joblib.load('models/baseline_lgb.pkl')

# Check model type
print("Model type:", type(model))

# If it's a dictionary, get the actual model
if isinstance(model, dict):
    model = model.get('model', model)

# Get feature importance
try:
    if hasattr(model, 'feature_importances_'):
        features = model.feature_name_ if hasattr(model, 'feature_name_') else [f'f{i}' for i in range(len(model.feature_importances_))]
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 20 important features:")
        print(importance.head(20))
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance['feature'][:20], importance['importance'][:20])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    else:
        print("Model doesn't have feature_importances_ attribute")
        print("Available attributes:", [a for a in dir(model) if not a.startswith('_')])
except Exception as e:
    print("Error getting feature importance:", str(e))