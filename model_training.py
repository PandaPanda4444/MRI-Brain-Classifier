import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow GPU support: ", tf.test.is_built_with_cuda())
print("Is GPU available: ", tf.config.list_physical_devices('GPU'))

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU memory growth enabled")

#-----------------------------------------------------#
# Step 1: Load and Prepare Images
def load_data(data_dir, img_size=(128, 128)):
    """
    Input: Takes folder with 'healthy' and 'diseased' images
    Process: 
    - Loads images
    - Converts to grayscale
    - Resizes all images to same size
    - Normalizes pixel values to 0-1 range
    Output: Returns processed images and their labels
    """
    classes = ['healthy', 'diseased']
    images = []
    labels = []
    
    for label_idx, label_name in enumerate(classes):
        folder_path = os.path.join(data_dir, label_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Read image in grayscale (typical for MRI)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label_idx)
    
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int')
    
    # Normalize images to [0, 1]
    images = images / 255.0
    # Expand dimensions: (N, width, height, channels)
    images = np.expand_dims(images, axis=-1)
    
    return images, labels

#-----------------------------------------------------#
# Step 2: Build Neural Network Structure
def build_model(input_shape):
    """
    Creates a CNN with:
    - 3 Convolutional blocks (each with 2 Conv2D layers)
    - MaxPooling and Dropout after each block
    - Final Dense layers for classification
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 classes: healthy, diseased
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Paths and parameters
    data_directory = os.path.join("data")
    IMG_WIDTH, IMG_HEIGHT = 128, 128
    test_size_ratio = 0.2
    
    # Load data from the 'data' folder
    X, y = load_data(data_directory, (IMG_WIDTH, IMG_HEIGHT))
    
    # Add this to model_training.py after loading data
    print(f"Total images loaded: {len(X)}")
    print(f"Number of healthy images: {np.sum(y == 0)}")
    print(f"Number of diseased images: {np.sum(y == 1)}")

    #-----------------------------------------------------#
    # Step 3: Prepare Training and Validation Data
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
        test_size=test_size_ratio, 
        random_state=42, 
        stratify=y)
    
    # Data augmentation for more training samples
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Build the CNN model
    model = build_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, 1))
    model.summary()
    
    #-----------------------------------------------------#
    # Step 4: Set Training Parameters
    # Training settings
    batch_size = 32
    epochs = 50

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    #-----------------------------------------------------#
    # Step 5: Train the Model
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        callbacks=[early_stopping]
    )
    
    #-----------------------------------------------------#
    # Step 6: Evaluate and Visualize Results
    # Evaluate model performance
    val_loss, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title("Accuracy")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title("Loss")
    plt.legend()
    
    plt.show()

    # Add this to model_training.py after loading data
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X[y == 0][0].squeeze(), cmap='gray')
    plt.title("Sample Healthy Image")
    plt.subplot(1, 2, 2)
    plt.imshow(X[y == 1][0].squeeze(), cmap='gray')
    plt.title("Sample Diseased Image")
    plt.show()

    #-----------------------------------------------------#
    # Step 7: Save the Trained Model
    os.makedirs("models", exist_ok=True)
    model.save(os.path.join("models", "model.h5"))
