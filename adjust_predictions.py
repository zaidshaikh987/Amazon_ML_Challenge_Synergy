# Quick check for target distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('student_resource/dataset/train.csv')
prices = df['price']
print(f"Price stats:\n{prices.describe(percentiles=[.1, .25, .5, .75, .9, .99])}")

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.hist(prices, bins=50)
plt.title('Price Distribution')
plt.subplot(122)
plt.hist(np.log1p(prices), bins=50)
plt.title('Log(1+Price) Distribution')
plt.tight_layout()
plt.show()