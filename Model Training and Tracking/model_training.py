import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# Set the MLflow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5002"

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")

# Enable MPS (Metal Performance Shaders) if available
if tf.config.list_physical_devices('MPS'):
    print("MPS is available and will be used for training")
else:
    print("MPS is not available. Training will use CPU")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../Data Fetching and Preprocessing/files/preprocessed_files/final_output.csv')

# Drop the 'Unnamed: 0' column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Merge capital and small letters into the same class
df['label'] = df['label'].str.lower()

# Separate features and labels
print("Processing features and labels...")
X = df.drop(columns=['Label', 'label']).values
y = df['label'].values

# Encode the labels using OneHotEncoder
print("Encoding labels...")
onehot_encoder = OneHotEncoder(sparse_output=False)
y = onehot_encoder.fit_transform(y.reshape(-1, 1))
label_names = onehot_encoder.categories_[0]

# Save the one-hot encoder
with open('models/onehot_encoder.pkl', 'wb') as f:
    pickle.dump(onehot_encoder, f)

# Reshape the features to be in the format (number of samples, height, width, channels)
X = X.reshape(-1, 28, 28, 1)  # Assuming the images are 28x28 pixels

# Normalize the pixel values
print("Normalizing pixel values...")
X = X / 255.0

# Visualize some datapoints
def visualize_samples(X, y, label_names, n_samples=20):
    plt.figure(figsize=(20, 2))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {label_names[np.argmax(y[i])]}")
        plt.axis('off')
    plt.show()

visualize_samples(X, y, label_names)

# Perform basic EDA
print("Performing basic EDA...")
# Distribution of labels
plt.figure(figsize=(12, 6))
sns.countplot(x=pd.Series(np.argmax(y, axis=1)).map({i: label for i, label in enumerate(label_names)}))
plt.title('Distribution of Labels in the Dataset')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.show()

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Define the CNN model with batch normalization and increased dropout
print("Defining the CNN model...")
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')  # Adjust the number of output classes dynamically
])

# Compile the model
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning rate scheduler callback
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return float(lr * tf.math.exp(-0.1))

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Setup MLflow
mlflow.set_experiment('handwritten_character_recognition')

# Train the model with MLflow logging
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)

    # Enable automatic logging
    mlflow.tensorflow.autolog()

    # Train the model with the scheduler
    print("Training the model with learning rate scheduler...")
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                        epochs=50,
                        validation_data=(X_test, y_test),
                        verbose=1,
                        callbacks=[callback])

    # Evaluate the model
    print("Evaluating the model on test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Log metrics explicitly to see if this fixes the issue
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric('train_loss', history.history['loss'][epoch], step=epoch)
        mlflow.log_metric('val_loss', history.history['val_loss'][epoch], step=epoch)
        mlflow.log_metric('train_accuracy', history.history['accuracy'][epoch], step=epoch)
        mlflow.log_metric('val_accuracy', history.history['val_accuracy'][epoch], step=epoch)

    # Save the model
    model_save_path = 'models/handwritten_character_model.h5'
    model.save(model_save_path)
    mlflow.log_artifact(model_save_path)

    print("Model saved at:", model_save_path)
    print("Training complete.")
