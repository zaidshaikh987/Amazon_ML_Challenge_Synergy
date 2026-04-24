import pandas as pd
import numpy as np

print("="*70)
print("ANALYZING WHY SMAPE IS STUCK AT 54%")
print("="*70)

# Load data
train = pd.read_csv('student_resource/dataset/train.csv')
test = pd.read_csv('student_resource/dataset/test.csv')

print("\n1. PRICE DISTRIBUTION ANALYSIS")
print("-"*70)
print(f"Train samples: {len(train)}")
print(f"Test samples: {len(test)}")
print(f"\nTrain price stats:")
print(f"  Mean: ${train['price'].mean():.2f}")
print(f"  Median: ${train['price'].median():.2f}")
print(f"  Std: ${train['price'].std():.2f}")
print(f"  Min: ${train['price'].min():.2f}")
print(f"  Max: ${train['price'].max():.2f}")
print(f"  Coefficient of Variation: {train['price'].std() / train['price'].mean():.2f}")

print(f"\nPrice percentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: ${train['price'].quantile(p/100):.2f}")

# Check for price skew
print(f"\n2. PRICE DISTRIBUTION SKEW")
print("-"*70)
skew = train['price'].skew()
print(f"Skewness: {skew:.2f}")
if skew > 2:
    print("⚠️ HIGHLY SKEWED - log transform is essential!")
elif skew > 1:
    print("⚠️ MODERATELY SKEWED - log transform helps")
else:
    print("✓ Normal-ish distribution")

# Check for outliers
print(f"\n3. OUTLIER ANALYSIS")
print("-"*70)
Q1 = train['price'].quantile(0.25)
Q3 = train['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = train[(train['price'] < Q1 - 1.5*IQR) | (train['price'] > Q3 + 1.5*IQR)]
print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(train)*100:.1f}%)")
print(f"Max non-outlier price: ${(Q3 + 1.5*IQR):.2f}")

# Extreme prices
extreme_high = train[train['price'] > 1000]
extreme_low = train[train['price'] < 1]
print(f"Prices > $1000: {len(extreme_high)} ({len(extreme_high)/len(train)*100:.1f}%)")
print(f"Prices < $1: {len(extreme_low)} ({len(extreme_low)/len(train)*100:.1f}%)")

# Check text data quality
print(f"\n4. TEXT DATA QUALITY")
print("-"*70)
train['text_len'] = train['catalog_content'].str.len()
print(f"Mean text length: {train['text_len'].mean():.0f} chars")
print(f"Median text length: {train['text_len'].median():.0f} chars")
print(f"Empty/Very short text (<50 chars): {len(train[train['text_len'] < 50])} ({len(train[train['text_len'] < 50])/len(train)*100:.1f}%)")

# Calculate baseline SMAPE if you predicted median
print(f"\n5. BASELINE SMAPE (if you just predict median)")
print("-"*70)
median_price = train['price'].median()
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100

baseline_smape = smape(train['price'].values, np.full(len(train), median_price))
print(f"SMAPE if always predict median (${median_price:.2f}): {baseline_smape:.2f}%")

# Calculate baseline SMAPE if you predicted mean
mean_price = train['price'].mean()
baseline_smape_mean = smape(train['price'].values, np.full(len(train), mean_price))
print(f"SMAPE if always predict mean (${mean_price:.2f}): {baseline_smape_mean:.2f}%")

print(f"\n6. THEORETICAL ANALYSIS")
print("-"*70)
print("If your model SMAPE is 54%:")
if baseline_smape > 54:
    print(f"✅ You're BETTER than baseline ({baseline_smape:.1f}%)")
    print(f"   Improvement: {baseline_smape - 54:.1f} percentage points")
else:
    print(f"❌ You're WORSE than baseline ({baseline_smape:.1f}%)")
    print(f"   This suggests: MODEL IS LEARNING WRONG PATTERNS")

# Price variance analysis
print(f"\n7. PRICE VARIANCE (why SMAPE might be stuck)")
print("-"*70)
cv = train['price'].std() / train['price'].mean()
print(f"Coefficient of Variation: {cv:.2f}")
if cv > 2:
    print("⚠️ VERY HIGH VARIANCE - prices are all over the place!")
    print("   This makes prediction inherently difficult")
    print(f"   Even perfect models might struggle to get below {50 + cv*10:.0f}% SMAPE")

# Check for duplicate or similar products with different prices
print(f"\n8. CHECKING FOR PRICE INCONSISTENCY")
print("-"*70)
# Group by text content similarity (first 100 chars)
train['text_prefix'] = train['catalog_content'].str[:100]
duplicates = train.groupby('text_prefix')['price'].agg(['count', 'std', 'mean']).reset_index()
problematic = duplicates[(duplicates['count'] > 5) & (duplicates['std'] > duplicates['mean'] * 0.5)]
if len(problematic) > 0:
    print(f"⚠️ Found {len(problematic)} product groups with high price variance")
    print("   Same/similar products have wildly different prices")
    print("   This makes learning difficult!")
else:
    print("✓ Price consistency looks reasonable")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

if baseline_smape < 60:
    print("✅ Your 54% SMAPE is reasonable for this dataset")
    print("   The data has inherent noise/variance")
    print("\n💡 TO IMPROVE FURTHER:")
    print("   1. Remove outliers (prices > $1000 or < $1)")
    print("   2. Use stratified ensembles (models for different price ranges)")
    print("   3. Add external features (if allowed)")
    print("   4. Post-process predictions based on text patterns")
else:
    print("❌ Your model should be able to beat 54% SMAPE")
    print("   Something is wrong with training/prediction pipeline")
    print("\n🔍 CHECK:")
    print("   1. Are you using the SAME preprocessing for train/test?")
    print("   2. Are sample_ids aligned correctly?")
    print("   3. Are you inverse-transforming predictions correctly?")

print("="*70)
