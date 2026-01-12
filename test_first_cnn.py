import create_data_sets_for_1st_CNN_v2
import os
import pandas as pd
import first_CNN
import numpy as np 
import plot_results_cnn


directory = "test_set_for_1st_CNN"

if not os.path.exists(directory):
    os.makedirs(directory)

# 1. Lire le fichier de référence (celui qui contient les lignes à exclure)
with open('data_set_for_1st_CNN/motor_vehicle_images_subset.txt', 'r', encoding='utf-8') as f:
    # On utilise un set pour une recherche ultra-rapide
    lignes_a_exclure = set(ligne.strip() for ligne in f)

# 2. Filtrer le fichier source
lignes_filtrees = []
with open('data_set_for_1st_CNN/motor_vehicle_images.txt', 'r', encoding='utf-8') as f:
    for ligne in f:
        if ligne.strip() not in lignes_a_exclure:
            lignes_filtrees.append(ligne)

# 3. Sauvegarder le résultat
with open('test_set_for_1st_CNN/motor_vehicul_test_set.txt', 'w', encoding='utf-8') as f:
    f.writelines(lignes_filtrees)



folder_path_pedestrian = 'data/pedestrian_Traffic_Light_v1i_yolov11/test/images/'
output_file_pedestrian = 'test_set_for_1st_CNN/pedestrian_traffic_light.txt'


fichiers = ['test_set_for_1st_CNN/motor_vehicul_test_set.txt', 'test_set_for_1st_CNN/pedestrian_traffic_light.txt']

create_data_sets_for_1st_CNN_v2.extract_filenames_to_txt(folder_path_pedestrian, output_file_pedestrian)
create_data_sets_for_1st_CNN_v2.fusionne_txt_et_ajoute_label(fichiers,"test_set_for_1st_CNN/fusion_avec_labels.txt")

df = pd.read_csv('test_set_for_1st_CNN/fusion_avec_labels.txt', names=['filename', 'label'], skipinitialspace=True)

X_test, y_test = create_data_sets_for_1st_CNN_v2.load_and_preprocess_dataset(df,"test")

print(X_test)


model = first_CNN.create_model() 
model.load_weights('first_CNN_weights.weights.h5')
model.summary()
classes = [str(i) for i in range(2)]

y_pred_probs = model.predict(X_test)
y_pred_probs = model.predict(X_test)
y_pred_probs_array = np.array(y_pred_probs)
y_pred_probs_binary_list = np.round(y_pred_probs_array).astype(int).flatten().tolist()
print(y_pred_probs_binary_list)

print(len(y_pred_probs))

# 4. Evaluate & Visualize
print("\n--- Visualisation des Performances ---")


# Graphique 2: Matrice de confusion
plot_results_cnn.plot_confusion_matrix(y_test, y_pred_probs_binary_list, classes)

# Graphique 3: Rapport métrique (Precision/Recall/F1)
plot_results_cnn.plot_classification_report(y_test, y_pred_probs_binary_list)

# Graphique 4: Inspection des erreurs
plot_results_cnn.plot_error_analysis(X_test, y_test, y_pred_probs_binary_list)
