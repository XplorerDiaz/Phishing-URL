from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import joblib  # Optional if you want to use scaler.pkl
from feature_extraction import extract_features

# Load normalization parameters
feature_mean = np.load("feature_mean.npy")
feature_std = np.load("feature_std.npy")

def normalize_features(features):
    """Normalize input features using saved mean and std values."""
    return (features - feature_mean) / feature_std

# Load saved autoencoder model for feature compression
class Autoencoder(nn.Module):
    def __init__(self, input_dim=55, latent_dim=26):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z

autoencoder = Autoencoder()
try:
    autoencoder.load_state_dict(torch.load("autoencoder.pth"))
    autoencoder.eval()
    print("Autoencoder loaded successfully.")
except Exception as e:
    print("Error loading Autoencoder:", str(e))

# Load classification model
class CNNBinaryClassifier1D(nn.Module):
    def __init__(self):
        super(CNNBinaryClassifier1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(576, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.batchnorm1(self.conv1(x))))
        f2 = self.relu(self.batchnorm2(self.conv2(x)))
        f2_pooled = self.pool(f2)
        f3 = self.relu(self.batchnorm3(self.conv3(f2)))
        f3_pooled = self.pool(f3)
        f2_flat = torch.flatten(f2_pooled, start_dim=1)
        f3_flat = torch.flatten(f3_pooled, start_dim=1)
        fused = torch.cat((f2_flat, f3_flat), dim=1)
        x = self.relu(self.fc1(fused))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

classifier = CNNBinaryClassifier1D()
try:
    classifier.load_state_dict(torch.load("classifier.pth"))
    classifier.eval()
    print("Classifier loaded successfully.")
except Exception as e:
    print("Error loading classifier:", str(e))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("Received request:", request.json)
        input_data = request.json["input_data"]
        print("Input data:", input_data)
        
        # Extract features
        features_55 = extract_features(input_data)
        print("Extracted features (55):", features_55)
        print("Extracted features shape:", len(features_55))

        # Normalize features using saved mean and std
        normalized_features = normalize_features(np.array(features_55))
        print("Normalized features:", normalized_features)

        # Convert to tensor for PyTorch model
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
        print("Features tensor shape before autoencoder:", features_tensor.shape)

        # Pass through Autoencoder to get compressed representation
        _, compressed_features = autoencoder(features_tensor)
        print("Compressed features shape after autoencoder:", compressed_features.shape)

        # Reshape for CNN input
        compressed_features = compressed_features.view(1, 1, 26)
        print("Reshaped compressed features for CNN:", compressed_features.shape)

        # Get prediction from classifier
        with torch.no_grad():
            output = classifier(compressed_features)
        print("Raw model output:", output.item())

        # Convert output to binary prediction
        prediction = 1 if output.item() > 0.5 else 0
        print("Final prediction:", prediction)

        return jsonify({"prediction": prediction})
    
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
