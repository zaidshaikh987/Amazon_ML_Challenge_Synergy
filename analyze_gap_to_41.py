"""
Analysis: What creates 13 point SMAPE gap (54% → 41%)?
"""

print("="*70)
print("REVERSE ENGINEERING TOP TEAM'S 41% SMAPE")
print("="*70)

print("\n🤔 What could create a 13-point SMAPE improvement?")
print("-"*70)

print("\n1. BETTER EMBEDDINGS (Impact: ~5-8 points)")
print("   What they might use:")
print("   - Fine-tuned CLIP on product images")
print("   - E-commerce specific text embeddings")
print("   - Multi-modal model (BLIP-2, LLaVA)")
print("   ❌ You used: Generic CLIP + Sentence-BERT")

print("\n2. QUANTITY/PACK SIZE EXTRACTION (Impact: ~3-5 points)")
print("   Critical insight: IPQ affects price linearly!")
print("   - 'Pack of 12' = 12x individual price")
print("   - Need PERFECT extraction (not just regex)")
print("   ❌ Your features likely miss complex cases")

print("\n3. TRAIN/TEST LEAKAGE OR PSEUDO-LABELING (Impact: ~5-10 points)")
print("   Techniques:")
print("   - Iterative pseudo-labeling on test set")
print("   - Using test predictions to retrain")
print("   - Test-time augmentation")
print("   ⚠️ May not be allowed, but hard to detect")

print("\n4. ENSEMBLE OF DIVERSE MODELS (Impact: ~2-4 points)")
print("   - 10+ models with different architectures")
print("   - Neural nets + GBDTs + linear models")
print("   - Stacking with meta-learner")
print("   ❌ You tried ensemble but same model family")

print("\n5. PRICE NORMALIZATION (Impact: ~2-3 points)")
print("   Critical bug in your approach:")
print("   - Predicting TOTAL price when should predict PER-UNIT price")
print("   - Need to divide by quantity!")
print("   ❌ This could be THE bug causing 54%!")

print("\n" + "="*70)
print("🎯 MOST LIKELY ROOT CAUSE")
print("="*70)

print("\n**QUANTITY NORMALIZATION BUG**")
print("\nHypothesis:")
print("  - Your model predicts TOTAL price")
print("  - But should predict PER-UNIT price × quantity")
print("  - Example:")
print("    • 'Pack of 12 pens' = $6 total")
print("    • Per-unit price = $0.50")
print("    • Model should learn $0.50, then multiply by 12")
print("\n  - Without normalization:")
print("    • Model sees 'pack of 12' → $6")
print("    • Also sees 'single pen' → $0.50")
print("    • Can't learn consistent pattern!")

print("\n" + "="*70)
print("💡 ACTION PLAN")
print("="*70)

print("\n1. CRITICAL FIX: Normalize prices by quantity")
print("   - Extract IPQ (item pack quantity)")
print("   - Train on: price_per_unit = price / ipq")
print("   - Predict: total_price = predicted_per_unit × ipq")

print("\n2. Better quantity extraction")
print("   - Use multiple regex patterns")
print("   - Handle edge cases: 'dozen' = 12, 'pair' = 2")
print("   - Default to 1 if not found")

print("\n3. Remove outliers")
print("   - Prices > $1000 are likely errors")
print("   - Or are bulk/wholesale (different market)")

print("\n4. Better embeddings")
print("   - Try different CLIP model")
print("   - Or use BLIP-2 (better for products)")

print("\nExpected gain: 7-10 points → Target 44-47% SMAPE")
print("="*70)
