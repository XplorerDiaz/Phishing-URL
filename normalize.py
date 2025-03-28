import numpy as np
# Load normalization parameters
feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

def normalize_features(features):
    """Normalize input features using saved mean and std values."""
    return (features - feature_mean) / feature_std