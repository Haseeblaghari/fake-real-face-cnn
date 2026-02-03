import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ----------------------------
# Constants
# ----------------------------
IMG_SIZE = (224, 224)   # Resize all images to 224x224
BATCH_SIZE = 32         # Number of images per batch
DATASET_PATH = r"C:\Users\Haseeb\OneDrive\Documents\fake_image\real_and_fake_face"  # Path to dataset

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
# Using image_dataset_from_directory to automatically load images from folders
# Dataset should have two subfolders: 'fake' and 'real'
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,    # 20% data for validation
    subset="training",        # This is training set
    seed=42,                  # Seed for reproducibility
    image_size=IMG_SIZE,      # Resize images to 224x224
    batch_size=BATCH_SIZE,
    label_mode="binary"       # Binary labels (0 or 1)
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",      # This is validation set
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# ----------------------------
# 2️⃣ Normalize Images
# ----------------------------
# Scale pixel values from [0, 255] to [0, 1] for better training
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ----------------------------
# 3️⃣ Performance Optimization
# ----------------------------
# Cache and prefetch for faster data loading during training
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# 4️⃣ Data Augmentation
# ----------------------------
# Apply transformations to training images to reduce overfitting
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),      # Flip image left-right
    layers.RandomRotation(0.1),           # Rotate +-10%
    layers.RandomZoom(0.1),               # Zoom in/out 10%
    layers.RandomTranslation(0.1, 0.1)    # Translate horizontally & vertically
])

# ----------------------------
# 5️⃣ Build CNN Model with Regularization
# ----------------------------
model = models.Sequential([
    data_augmentation,  # Augmentation layer applied only during training

    # First Conv block
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224,224,3)),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    # Second Conv block
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    # Third Conv block
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),
    layers.BatchNormalization(),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(64, activation="relu"),  # Smaller dense layer to reduce overfitting
    layers.Dropout(0.5),                  # Dropout for regularization
    layers.Dense(1, activation="sigmoid") # Output layer (0 = FAKE, 1 = REAL)
])

# Compile the model
model.compile(
    optimizer="adam",                   # Adam optimizer
    loss="binary_crossentropy",         # Binary classification loss
    metrics=["accuracy"]                # Track accuracy
)

# Print model summary
model.summary()

# ----------------------------
# 6️⃣ Early Stopping Callback
# ----------------------------
# Stop training early if validation loss does not improve for 9 epochs
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=9,
    restore_best_weights=True           # Restore best model weights after stopping
)

# ----------------------------
# 7️⃣ Train Model
# ----------------------------
# Train the CNN with early stopping to avoid overfitting
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,                          # Train up to 50 epochs
    callbacks=[early_stop]              # Use early stopping
)

# ----------------------------
# 8️⃣ Save Model
# ----------------------------
# Save the trained model in Keras format (.keras)
model.save("fake_real_cnn_augmented.keras")
