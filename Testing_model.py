import tensorflow as tf
import numpy as np

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = (224, 224)  # Resize images to 224x224 (same as training)

# ----------------------------
# 1️⃣ Function to load and preprocess a single image
# ----------------------------
def load_and_preprocess_image(image_path):
    """
    Loads an image from the given path, resizes it to IMG_SIZE,
    normalizes pixel values to [0,1], and adds a batch dimension.
    """
    # Load the image and resize
    img = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    # Convert to numpy array
    img = tf.keras.utils.img_to_array(img)
    
    # Normalize pixel values to [0,1]
    img = img / 255.0
    
    # Add batch dimension (model expects input shape: [1, 224, 224, 3])
    img = np.expand_dims(img, 0)
    return img

# ----------------------------
# 2️⃣ Load the trained model
# ----------------------------
# Make sure the path matches where you saved your trained CNN
model = tf.keras.models.load_model(
    r"C:\Users\Haseeb\OneDrive\Documents\fake_image\fake_real_cnn_augmented.keras"
)

# ----------------------------
# 3️⃣ Path to the image you want to test
# ----------------------------
image_path = "IMAGE_PATH"  # Replace this with your actual image path

# ----------------------------
# 4️⃣ Preprocess the image
# ----------------------------
img = load_and_preprocess_image(image_path)

# ----------------------------
# 5️⃣ Predict using the trained CNN
# ----------------------------
# Model output is a sigmoid value: 0 -> FAKE, 1 -> REAL
prediction = model.predict(img)

# ----------------------------
# 6️⃣ Map prediction to class names
# ----------------------------
# Since dataset folders were ['fake', 'real'], assign accordingly
class_names = ['fake', 'real']

# Convert sigmoid output to binary class
confidence = prediction[0][0]               # Probability of "real"
predicted_class = 1 if confidence > 0.5 else 0
label = class_names[predicted_class]

# ----------------------------
# 7️⃣ Print results
# ----------------------------
print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}")
