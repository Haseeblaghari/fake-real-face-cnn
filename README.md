# ================================================
# Fake vs Real Face Detection Using CNN
# ================================================
# This repository contains a Convolutional Neural Network (CNN)
# built with TensorFlow/Keras to classify images as Fake or Real.
# The model is trained on a dataset containing two main folders:
# 'training_fake' and 'training_real'.

# ----------------------------------------
# Table of Contents
# ----------------------------------------
# - Dataset
# - Installation
# - Training the Model
# - Testing the Model
# - Model Details
# - Results
# - Future Improvements

# ----------------------------------------
# Dataset
# ----------------------------------------
# The dataset folder is named "real_and_fake_face" and should have the following structure:
#
# real_and_fake_face/
# │
# ├── training_fake/      # Folder containing fake face images
# └── training_real/      # Folder containing real face images
#
# Images are automatically split into training (80%) and validation (20%) by the code.

# ----------------------------------------
# Installation
# ----------------------------------------
# 1. Clone this repository:
# git clone https://github.com/YourUsername/fake-real-face-detection.git
# cd fake-real-face-detection
#
# 2. Install required packages:
# pip install -r requirements.txt
#
# Make sure you are using Python 3.8+ and TensorFlow >= 2.10

# ----------------------------------------
# Training the Model
# ----------------------------------------
# The training script train_model.py includes:
# - Dataset loading from "real_and_fake_face"
# - Image normalization
# - Data augmentation (flip, rotation, zoom, translation)
# - CNN architecture with dropout and batch normalization
# - Early stopping to prevent overfitting
#
# To train the model:
# python train_model.py
#
# The trained model will be saved as:
# fake_real_cnn_augmented.keras

# ----------------------------------------
# Testing the Model
# ----------------------------------------
# Use the test_model.py script to test the model on a single image:
# python test_model.py
#
# - Replace IMAGE_PATH in test_model.py with your test image path.
# - Output example:
# Prediction: REAL
# Confidence: 0.87
#
# Confidence is the probability of the image being "Real".

# ----------------------------------------
# Model Details
# ----------------------------------------
# - Input: (224, 224, 3) images
# - Architecture: 3 Conv2D layers with MaxPooling + BatchNormalization, Flatten + Dense layers
# - Activation: ReLU for hidden layers, Sigmoid for output layer
# - Loss: Binary Crossentropy
# - Optimizer: Adam
# - Regularization: Dropout (0.5) + Data Augmentation + EarlyStopping

# ----------------------------------------
# Results
# ----------------------------------------
# - Training Accuracy: ~82%
# - Validation Accuracy: ~55-60% (can improve with larger dataset or transfer learning)
# - The model can classify unseen images as Fake or Real with confidence scores.

# ----------------------------------------
# Future Improvements
# ----------------------------------------
# - Use Transfer Learning (EfficientNet, MobileNet) for higher accuracy
# - Collect a larger dataset to reduce overfitting
# - Add multi-class classification for deepfake detection types
# - Deploy the model via Web App / Flask / Streamlit

# ----------------------------------------
# License
# ----------------------------------------
# This project is open-source and free to use for educational purposes.
