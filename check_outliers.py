import pandas as pd

train = pd.read_csv('student_resource/dataset/train.csv')

print("Outlier analysis:")
print(f"Prices > $500: {(train['price'] > 500).sum()}")
print(f"Prices > $1000: {(train['price'] > 1000).sum()}")
print(f"99th percentile: ${train['price'].quantile(0.99):.2f}")
print(f"95th percentile: ${train['price'].quantile(0.95):.2f}")
