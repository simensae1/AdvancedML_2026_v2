import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd


def create_model(input_shape=(640, 640, 3)):
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Layer 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Layer 3
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        # Layer 4
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        # Dense layers with Dropout to prevent overfitting
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.1),
        layers.Dropout(0.5),

        layers.Dense(64),
        layers.LeakyReLU(alpha=0.1),

        # Output layer for Binary Classification
        layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == "__main__":
    # Load the datasets
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Model initialization
    model = create_model()
    lr = 0.0001  # Start low to avoid skipping the minority class patterns
    opt = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall()]
    )

    model.summary()

    # Re-initializing and compiling with standard Adam for the training phase
    model = create_model()
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.summary()

    # Training process
    history = model.fit(
        X_train, y_train,
        epochs=15,             # Number of complete passes over the dataset
        batch_size=32,         # Number of images processed before updating the model
        validation_split=0.2   # Uses 20% of data for validation during training
    )

    # Save the trained model and weights
    model.save('first_CNN.keras')
    model.save_weights('first_CNN_weights.weights.h5')

    # Save training history to a CSV file
    pd.DataFrame(history.history).to_csv("history.csv", index=False)
