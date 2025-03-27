# Phishing URL Classification using Deep Learning

## Overview
This project focuses on classifying URLs as either phishing websites or legitimate ones using deep learning techniques. The approach leverages feature extraction, Variational Autoencoders (VAEs), Autoencoders (AEs), and Convolutional Neural Networks (CNNs) to achieve high classification accuracy.

## Dataset
- **Total Samples:** 50,000
  - 25,000 phishing URLs
  - 25,000 legitimate URLs
- **Features:** 55 extracted features based on a research paper
- **Train-Test Split:** 80-20 (manually divided)

## Methodology
1. **Feature Extraction & Preprocessing:**
   - Extracted 55 features per URL based on research findings.
   - Split dataset into 80% training and 20% testing.

2. **Hyperparameter Tuning using Optuna:**
   - Used Optuna to find the optimal latent space size for both VAEs and AEs.
   - **Best Latent Size for VAEs:** 19 (after 20 trials)
   - **Best Latent Size for AEs:** 16 (after 20 trials)

3. **Training the Autoencoders:**
   - Trained VAE on the training dataset and extracted latent features for the test dataset.
   - Saved extracted features as `train_latent.csv` and `test_latent.csv`.
   - Repeated the process for AEs, saving extracted features as `train_latent2.csv` and `test_latent2.csv`.

4. **CNN-based Classification:**
   - Used a 3-layered CNN for binary classification, taking 1D feature vectors as input.
   - Evaluated performance using **multi-layer feature fusion:**
     - Features from layers 1, 2, and 3 combined at classification head.
     - Features from layers 2 and 3 only.
     - Features from layer 3 alone (baseline).
   - The best model was for **fusion of layers 2 and 3**.
   - Applied the same fusion approach to autoencoder-extracted features.

## Post-Training Steps
Once the best model was identified, the following steps were performed:

1. Train the model only with the training dataset. During training, if batch normalization is applied, store the normalized values graph for each batch separately in an output file (`studentname-regnumber-normalizedgraph.jpeg/pdf/doc`).
2. During training, display parameters layer-by-layer.
3. Print training accuracy and loss on the terminal and store them in an Excel file for plotting.
4. Plot **accuracy vs epochs** and **loss vs epochs** graphs on the terminal, saving them as `studentname-regnumber-accuracygraph.jpeg` and `studentname-regnumber-lossgraph.jpeg`.
5. Once training is complete, save the trained classification model.
6. Load the trained model in a separate program and test it on an unseen dataset.
7. Print the actual and predicted class labels for all test inputs on the terminal and display the number of misclassified samples. Store these results in `studentname-regnumber-prediction.excel`.
8. Measure testing accuracy, precision, recall, F1-score, MCC, and confusion matrix, printing them on the terminal and saving them to an appropriate file.
9. Plot the ROC curve on the terminal with the AUC value and save it as `studentname-regnumber-rocgraph.jpeg`.
10. Measure training time, display it on the terminal, and store it in `studentname-regnumber-trainingtime.excel`.
11. Measure the testing time for each input separately and calculate the average testing time. Display this information on the terminal and save it in `studentname-regnumber-trainingtime.excel`.

## Results
- **VAE-based feature extraction:** around 95% accuracy
- **Autoencoder-based feature extraction:** above 99% accuracy

## Dependencies
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas tensorflow keras optuna
```



## Conclusion
This project demonstrates an effective method for phishing URL classification using deep learning. By leveraging VAEs, AEs, and CNN-based classification with feature fusion, the model achieves high accuracy, showcasing the potential of deep learning in cybersecurity applications.
