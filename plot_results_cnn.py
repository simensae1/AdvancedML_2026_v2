import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import first_CNN

# ==========================================
#  EVALUATION AND PLOTTING FUNCTIONS
# ==========================================


def plot_learning_curves(history):
    """Displays the evolution of loss and accuracy."""
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
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'r*-', label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred_classes, classes):
    """Displays the confusion matrix as a Heatmap."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_classification_report(y_true, y_pred_classes):
    """Converts the sklearn textual report into a pandas heatmap."""
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    plt.figure(figsize=(10, 6))
    # Excluding the last rows (accuracy, macro avg, etc.) for a cleaner heatmap of metrics
    sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report (Precision, Recall, F1)")
    plt.savefig('classification_report.png')
    plt.show()


def plot_error_analysis(x_test, y_test, y_pred_classes, num_images=10):
    """Displays examples of images where the model made a mistake."""

    y_test_flat = np.array(y_test).flatten()
    y_pred_flat = np.array(y_pred_classes).flatten()

    # Ensure x_test matches the prediction size to avoid IndexError
    x_test_subset = x_test[:len(y_pred_flat)]
    y_test_subset = y_test_flat[:len(y_pred_flat)]

    # Calculate error mask
    errors = (y_pred_flat != y_test_subset)

    x_errors = x_test_subset[errors]
    y_true_errors = y_test_subset[errors]
    y_pred_errors = y_pred_flat[errors]

    if len(x_errors) == 0:
        print("No errors found on this test set.")
        return

    plt.figure(figsize=(15, 6))
    plt.suptitle("Classification Error Examples (True -> Predicted)", fontsize=16)

    for i in range(min(num_images, len(x_errors))):
        plt.subplot(2, 5, i + 1)
        # Display image (assumes normalization 0-1 or 0-255)
        plt.imshow(x_errors[i])
        plt.title(f"True: {y_true_errors[i]} | Pred: {y_pred_errors[i]}", color='red', fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('classification_errors.png')
    plt.show()

# ==========================================
#  MAIN EXECUTION
# ==========================================


if __name__ == "__main__":
    # 1. Load Model and Data
    model = first_CNN.create_model()
    # Ensure the path below matches your folder structure
    model.load_weights('models/first_CNN_weights.weights.h5')
    model.summary()

    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    classes = [str(i) for i in range(2)]

    # 2. Make Predictions
    print("Running predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred_probs_array = np.array(y_pred_probs)
    y_pred_probs_binary_list = np.round(y_pred_probs_array).astype(int).flatten().tolist()

    # 3. Evaluate & Visualize
    print("\n--- Performance Visualization ---")

    # Plot 1: Learning Curves
    try:
        history = pd.read_csv("history.csv")
        plot_learning_curves(history)
    except FileNotFoundError:
        print("History file not found. Skipping learning curves.")

    # Plot 2: Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_probs_binary_list, classes)

    # Plot 3: Metrics Report (Precision/Recall/F1)
    plot_classification_report(y_test, y_pred_probs_binary_list)

    # Plot 4: Error Inspection
    plot_error_analysis(X_test, y_test, y_pred_probs_binary_list)
