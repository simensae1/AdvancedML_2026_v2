from ultralytics import YOLO
import cv2
import os
import first_CNN
import create_data_sets_for_1st_CNN_v2
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

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
            # In COCO dataset, 'traffic light' is class index 9
            class_id = int(box.cls[0])
            label = result.names[class_id]

            if label == "traffic light" or label == "pedestrian Traffic Light":
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


def display_image_grid(image_list, value_list, cols=3):
    """
    Displays a grid of images with their associated values as titles.
    
    image_list: List of image paths or numpy arrays
    value_list: List of values (labels, scores, etc.)
    cols: Number of columns in the grid
    """
    num_images = len(image_list)
    rows = math.ceil(num_images / cols)
    
    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten() # Flatten in case of multiple rows
    
    for i in range(num_images):
        img = image_list[i]
        val = value_list[i]
        
        # Load image if it's a path string
        if isinstance(img, str):
            img = mpimg.imread(img)
            
        axes[i].imshow(img)
        axes[i].set_title(f"Value: {val}", fontsize=12, pad=10)
        axes[i].axis('off') # Hide the x/y axis pixels
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig("result_full_model", bbox_inches='tight', dpi=300)
    plt.show()


def classify_traffic_light(path_image):
    #model_yolo = YOLO("runs/detect/train/weights/best.pt")
    model_yolo = YOLO("yolo11x.pt")
    model_first_cnn = first_CNN.create_model()
    model_first_cnn.load_weights('first_CNN_weights.weights.h5')
    model_first_cnn.summary()

    results = model_yolo(path_image)
    fichiers = glob.glob("extracted_lights/*.jpg")

    for f in fichiers:
        os.remove(f)

    # 3. Visualize and save the results
    for result in results:
        result.show()  # Opens a window with bounding boxes
        result.save(filename="output.jpg")  # Saves the image to disk

    counter = extract_traffic_lights(results)
    print(counter)

    if counter == 0:
        return "no traffic light detected"

    else:
        liste_image = []
        repertoire = Path("extracted_lights")
        liste_image_path = []
        for fichier in os.listdir(repertoire):
            image_path = os.path.join(repertoire, fichier)
            img = cv2.imread(image_path)
            print(img.shape)

            image = create_data_sets_for_1st_CNN_v2.resize_with_padding(img)
            cv2.imwrite('my_saved_image0.jpg', image)
            print(image.shape)
            liste_image_path.append(image_path)

            image_norm = image.astype('float32') / 255.0
            liste_image.append(image_norm)

        batch = np.array(liste_image)
        print(batch.shape)

        prediction = model_first_cnn.predict(batch)
        print(prediction)
        binary_prediction = (prediction > 0.5).astype("int32")
        print(binary_prediction)
        labels_list = ["vehiculte traffic light" if x == 0 else "pedestrian traffic light" for x in binary_prediction]
        print(labels_list)
        display_image_grid(liste_image_path, labels_list)


# 1 = pedestrian traffic light
# 0 = vehiculte traffic light
classify_traffic_light("heon_IMG_0766.JPG")

# heon_IMG_0602 one pedestrian traffic light no vehicule traffic light
# heon_IMG_0552 no pedestrian traffic light two vehicule traffic light
