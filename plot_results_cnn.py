import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import first_CNN

# ==========================================
# 3. FONCTIONS D'ÉVALUATION ET PLOTTING
# ==========================================

def plot_learning_curves(history):
    """Affiche l'évolution de la perte et de la précision."""
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'r*-', label='Validation acc')
    plt.title('Précision (Accuracy)')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Perte (Loss)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('évolution de la perte et de la précision.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred_classes, classes):
    """Affiche la matrice de confusion sous forme de Heatmap."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.savefig(' matrice de confusion.png')
    plt.show()

def plot_classification_report(y_true, y_pred_classes):
    """Convertit le rapport textuel de sklearn en heatmap pandas."""
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Rapport de Classification (Precision, Recall, F1)")
    plt.savefig('Rapport de Classification (Precision, Recall, F1).png')
    plt.show()


def plot_error_analysis(x_test, y_test, y_pred_classes, num_images=10):
    """Affiche des exemples d'images où le modèle s'est trompé."""

    y_test_flat = np.array(y_test).flatten()
    y_pred_flat = np.array(y_pred_classes).flatten()

    # On s'assure que x_test correspond à la taille des prédictions
    # pour éviter l'IndexError si un batch a été coupé
    x_test_subset = x_test[:len(y_pred_flat)]
    y_test_subset = y_test_flat[:len(y_pred_flat)]

    # Calcul du masque d'erreurs
    errors = (y_pred_flat != y_test_subset)

    x_errors = x_test_subset[errors]
    y_true_errors = y_test_subset[errors]
    y_pred_errors = y_pred_flat[errors]

    if len(x_errors) == 0:
        print("Aucune erreur trouvée sur ce jeu de test")
        return

    plt.figure(figsize=(15, 6))
    plt.suptitle("Exemples d'erreurs de classification (Vrai -> Prédit)", fontsize=16)

    for i in range(min(num_images, len(x_errors))):
        plt.subplot(2, 5, i + 1)
        # On affiche l'image (si normalisée 0-1 ou en 0-255)
        plt.imshow(x_errors[i])
        plt.title(f"Vrai: {y_true_errors[i]} | Pred: {y_pred_errors[i]}", color='red', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('erreurs_classification.png')
    plt.show()

# ==========================================
# 4. EXÉCUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    # 1. Load Data

    model = first_CNN.create_model() 
    model.load_weights('first_CNN_weights.weights.h5')
    model.summary()
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    print(y_test)
    classes = [str(i) for i in range(2)]

    y_pred_probs = model.predict(X_test)
    y_pred_probs_array = np.array(y_pred_probs)
    y_pred_probs_binary_list = np.round(y_pred_probs_array).astype(int).flatten().tolist()
    print(y_pred_probs_binary_list)

    # 4. Evaluate & Visualize
    print("\n--- Visualisation des Performances ---")
    
    # Graphique 1: Courbes d'apprentissage
    history = pd.read_csv("history.csv")
    plot_learning_curves(history)
    
    # Graphique 2: Matrice de confusion
    plot_confusion_matrix(y_test, y_pred_probs_binary_list, classes)
    
    # Graphique 3: Rapport métrique (Precision/Recall/F1)
    plot_classification_report(y_test, y_pred_probs_binary_list)
    
    # Graphique 4: Inspection des erreurs
    plot_error_analysis(X_test, y_test, y_pred_probs_binary_list)
