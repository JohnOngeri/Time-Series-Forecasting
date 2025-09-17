import numpy as np
import pandas as pd
from scripts.data_utils import scale_features

# Test data cleaning
X_test = np.array([[1, 2, 3], [4, np.inf, 6], [7, 8, 9], [10, 1e20, 12]])
print("Original data:")
print(X_test)

try:
    X_scaled, scaler = scale_features(X_test)
    print("Scaled data:")
    print(X_scaled)
    print("Success!")
except Exception as e:
    print(f"Error: {e}")