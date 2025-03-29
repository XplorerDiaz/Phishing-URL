import numpy as np

feature_min = np.load("feature_min.npy")
feature_scale = np.load("feature_scale.npy")

def normalize_features(features):
    return (features - feature_min) / feature_scale
