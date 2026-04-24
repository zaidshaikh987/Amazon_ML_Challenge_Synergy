import pandas as pd

f1 = pd.read_csv('submissions/lgb_with_aggregates.csv')
f2 = pd.read_csv('submissions/final_lgb_agg_only.csv')

print("="*70)
print("CHECKING IF FILES ARE IDENTICAL")
print("="*70)

print(f"\nlgb_with_aggregates.csv: {len(f1)} rows")
print(f"final_lgb_agg_only.csv: {len(f2)} rows")

print(f"\nAre they identical? {f1.equals(f2)}")

if f1.equals(f2):
    print("\n✅ YES - THEY ARE EXACTLY THE SAME FILE!")
    print("\nYou can submit EITHER:")
    print("  • submissions/lgb_with_aggregates.csv")
    print("  • submissions/final_lgb_agg_only.csv")
    print("\nBoth have 52.82% CV SMAPE")
else:
    print("\nChecking differences...")
    print(f"Max price difference: ${abs(f1['price'] - f2['price']).max():.10f}")
    print(f"Mean price difference: ${abs(f1['price'] - f2['price']).mean():.10f}")

print("\n" + "="*70)
print("RECOMMENDATION: Submit submissions/final_lgb_agg_only.csv")
print("(Clear name indicating it's the final choice)")
print("="*70)
