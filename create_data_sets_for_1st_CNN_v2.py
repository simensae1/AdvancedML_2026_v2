import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
from ultralytics import YOLO
import cv2


def extract_traffic_lights(results, output_folder="extracted_lights"):
    """
    Extracts detected traffic lights from YOLO results and saves them as individual images.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    counter = 0
    for result in results:
        # Get the original image (numpy array)
        img = result.orig_img

        # Iterate through detected boxes
        for box in result.boxes:
            # Check if the detected class is 'traffic light'
            class_id = int(box.cls[0])
            label = result.names[class_id]

            if label == "traffic light":
                # Get coordinates: [x1, y1, x2, y2]
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)

                # Crop the image using numpy slicing: img[y1:y2, x1:x2]
                crop = img[y1:y2, x1:x2]

                # Save the cropped image
                save_path = os.path.join(output_folder, f"light_{counter}.jpg")
                cv2.imwrite(save_path, crop)
                print(f"Saved: {save_path}")
                counter += 1
    return counter


def filter_motor_vehicle_lights_to_txt(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use a set to keep filenames unique
    selected_images = set()

    for ann in data.get("annotations", []):
        # ignore: 0 represents valid motor vehicle lights in the dataset
        if ann.get("ignore") == 0:
            # Check if there is actual light information in the 'inbox'
            if len(ann.get("inbox", [])) > 0:
                for item in ann["inbox"]:
                    # Ensure it has a recognized color 
                    if item.get("color") in ["red", "green", "yellow"]:
                        selected_images.add(ann.get("filename"))
                        break  # move to next annotation

    # Save the filtered filenames
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for filename in sorted(selected_images):
            filename = filename.replace("\\", "/")
            f_out.write(f"{filename}\n")
    print(f"Done! Saved {len(selected_images)} image paths to {output_path}")


def extract_filenames_to_txt(directory, output_txt):
    if not os.path.exists(directory):
        print(f"Error: The folder '{directory}' does not exist.")
        return
    # 2. Get a list of all files in the directory
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]

    # 3. Write the names to the text file
    with open(output_txt, 'w', encoding='utf-8') as f:
        for filename in sorted(files):
            f.write(filename + '\n')
    print(f"Success! {len(files)} filenames have been saved to {output_txt}")


def fusionne_txt_et_ajoute_label(fichiers, fichier_final_name):
    fichiers_et_labels = {
        fichiers[0]: '0',  # vehicule
        fichiers[1]: '1'  # pedestrian
    }
    fichier_final = fichier_final_name
    with open(fichier_final, 'w', encoding='utf-8') as f_out:
        # On boucle sur le dictionnaire (nom du fichier et son label associé)
        for nom_fich, label in fichiers_et_labels.items():
            try:
                with open(nom_fich, 'r', encoding='utf-8') as f_in:
                    for ligne in f_in:
                        # On nettoie la ligne (enlève les espaces/sauts de ligne invisibles)
                        nom_image = ligne.strip()

                        # On n'écrit la ligne que si elle n'est pas vide
                        if nom_image:
                            # On écrit : nom_image, label
                            f_out.write(f"{nom_image}, {label}\n")
            except FileNotFoundError:
                print(f"Attention : Le fichier {nom_fich} est introuvable.")

    print(f"Fusion terminée avec labels dans : {fichier_final}")


def resize_with_padding(image, target_size=(640, 640)):
    """
    Redimensionne une image en gardant le ratio et ajoute du padding noir.
    """
    old_size = image.shape[:2]  # (hauteur, largeur)
    ratio = min(float(target_size[i]) / old_size[i] for i in range(len(target_size)))
    new_size = tuple([int(x * ratio) for x in old_size])

    # Redimensionnement de l'image d'origine
    image = cv2.resize(image, (new_size[1], new_size[0]))

    # Création d'une nouvelle image noire à la taille cible
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]  # Noir
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_img


def load_and_preprocess_dataset(df, type_sets, target_size=(640, 640)):
    X = []
    y = []
    model_yolo = YOLO("yolo11x.pt")
    model_specialized = YOLO("models/best.pt")
    for index, row in df.iterrows():
        img_name = row['filename'].strip()
        if row['label'] == 0:
            bout_de_chemin = "data/kaggle_traffic_light_data_set/" + "train" + "_dataset/"
            img_path = os.path.join(bout_de_chemin, img_name)
            results = model_yolo(img_path)
            first_result = results[0]
            extract_traffic_lights(first_result)
            path_after_extraction = "extracted_lights/light_0.jpg"
            img = cv2.imread(path_after_extraction)

        if row['label'] == 1:
            bout_de_chemin = "data/pedestrian_Traffic_Light_v1i_yolov11/" + type_sets + "/images"
            img_path = os.path.join(bout_de_chemin, img_name)
            results = model_specialized(img_path)
            first_result = results[0]
            extract_traffic_lights(first_result)
            path_after_extraction = "extracted_lights/light_0.jpg"
            img = cv2.imread(path_after_extraction)

        if img is not None:
            # 1. Padding et Resize
            img = resize_with_padding(img, target_size)
            # 2. Normalisation (0 à 1)
            img = img.astype('float32') / 255.0

            X.append(img)
            y.append(row['label'])

    return np.array(X), np.array(y)


if __name__ == "__main__":

    directory = "data_set_for_1st_CNN"

    if not os.path.exists(directory):
        os.makedirs(directory)

    folder_path_pedestrian = 'data/pedestrian_Traffic_Light_v1i_yolov11/train/images'
    output_file_pedestrian = 'data_set_for_1st_CNN/pedestrian_traffic_light.txt'

    input_json_motor_traffic_light = 'data/kaggle_traffic_light_data_set/train_dataset/train.json'
    output_file_motor_traffic_light = 'data_set_for_1st_CNN/motor_vehicle_images.txt'

    fichiers = ['data_set_for_1st_CNN/motor_vehicle_images_subset.txt', 'data_set_for_1st_CNN/pedestrian_traffic_light.txt']

    extract_filenames_to_txt(folder_path_pedestrian, output_file_pedestrian)
    filter_motor_vehicle_lights_to_txt(input_json_motor_traffic_light, output_file_motor_traffic_light)
    import random

    # Paramètres
    nom_fichier = output_file_motor_traffic_light
    nom_fichier_sortie = 'data_set_for_1st_CNN/motor_vehicle_images_subset.txt'  # Le nom du nouveau fichier
    nombre_lignes_a_extraire = 833

    # 1. Lecture du fichier source
    with open(nom_fichier, 'r', encoding='utf-8') as f:
        lignes = f.readlines()

    # 2. Sélection aléatoire
    if len(lignes) >= nombre_lignes_a_extraire:
        lignes_choisies = random.sample(lignes, nombre_lignes_a_extraire)
    else:
        print(f"Attention : Le fichier ne contient que {len(lignes)} lignes.")
        lignes_choisies = lignes

    # 3. Sauvegarde dans le nouveau fichier
    with open(nom_fichier_sortie, 'w', encoding='utf-8') as f_out:
        for ligne in lignes_choisies:
            f_out.write(ligne.strip() + '\n')

    fusionne_txt_et_ajoute_label(fichiers,"data_set_for_1st_CNN/fusion_avec_labels.txt")

    df = pd.read_csv('data_set_for_1st_CNN/fusion_avec_labels.txt', names=['filename', 'label'], skipinitialspace=True)
    X, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])  # stratify is a argument to unsure a good proportion of each label in the train and test set

    X_train, y_train = load_and_preprocess_dataset(X, "train")
    X_test, y_test = load_and_preprocess_dataset(X_test, "train")

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
