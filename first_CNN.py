import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def create_model(input_shape=(640, 640, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(), 
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == "__main__":

    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    # Initialisation du modèle
    model = create_model()

    # Compilation
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.summary()

    model = create_model()
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.summary()
    history = model.fit(
        X_train, y_train,
        epochs=10,             # Nombre de passages complets sur le dataset
        batch_size=8,         # Nombre d'images traitées avant de mettre à jour le modèle
        validation_split=0.2   # Utilise 20% des données pour valider pendant l'entraînement
    )
    model.save('first_CNN.keras')
    model.save_weights('first_CNN_weights.weights.h5')

    # Save history to a CSV
    pd.DataFrame(history.history).to_csv("history.csv", index=False)


    # Affichage de la précision (accuracy)
    plt.plot(history.history['accuracy'], label='Précision Entraînement')
    plt.plot(history.history['val_accuracy'], label='Précision Validation')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.show()
