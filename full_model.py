from ultralytics import YOLO
import cv2
import os
import first_CNN
import create_data_sets_for_1st_CNN_v2
import numpy as np
from pathlib import Path
import glob


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
        repertoire = Path("extracted_lights")
        liste_image = []
        for fichier in os.listdir(repertoire):
            image_path = os.path.join(repertoire, fichier)
            img = cv2.imread(image_path)
            print(img.shape)

            image = create_data_sets_for_1st_CNN_v2.resize_with_padding(img)
            cv2.imwrite('my_saved_image0.jpg', image)
            print(image.shape)

            image_norm = image.astype('float32') / 255.0
            liste_image.append(image_norm)

        batch = np.array(liste_image)
        print(batch.shape)

        prediction = model_first_cnn.predict(batch)
        print(prediction)


#classify_traffic_light("PTL_Dataset_768x576/23_jpg.rf.85ea24e72f8d75fd606a7efded7bcdf8.JPG")
classify_traffic_light("heon_IMG_0766.JPG")

# heon_IMG_0602 one pedestrian traffic light no vehicule traffic light
# heon_IMG_0552 no pedestrian traffic light two vehicule traffic light