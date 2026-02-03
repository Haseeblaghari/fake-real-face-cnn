# Fake vs Real Face Detection Using CNN
# A CNN built with TensorFlow/Keras to classify face images as Fake or Real.
# The model is trained on the dataset folder "real_and_fake_face" containing:
# - training_fake/
# - training_real/

# Dataset
# Folder structure:
# real_and_fake_face/
# ├── training_fake/
# └── training_real/
# Images are split into training (80%) and validation (20%) automatically.

# Installation
# 1. Clone the repo:
# git clone https://github.com/YourUsername/fake-real-face-detection.git
# cd fake-real-face-detection
# 2. Install dependencies:
# pip install -r requirements.txt
# Python 3.8+ and TensorFlow >= 2.10 recommended

# Training
# Run the training script:
# python train_model.py
# The trained model will be saved as "fake_real_cnn_augmented.keras"

# Testing
# Run the testing script:
# python test_model.py
# Replace IMAGE_PATH with your test image
# Output example:
# Prediction: REAL
# Confidence: 0.87

# Model
# - Input: 224x224x3 images
# - 3 Conv2D layers + MaxPooling + BatchNormalization
# - Flatten + Dense layers, Dropout(0.5)
# - Activation: ReLU (hidden), Sigmoid (output)
# - Loss: Binary Crossentropy, Optimizer: Adam

# Results
# - Training Accuracy: ~82%
# - Validation Accuracy: ~55-60%
# - Can classify unseen images as Fake or Real with confidence scores

# Future Improvements
# - Use Transfer Learning for higher accuracy
# - Collect more data to reduce overfitting
# - Multi-class classification for deepfakes
# - Deploy via Web App / Flask / Streamlit

# License
# Open-source, free for educational purposes
