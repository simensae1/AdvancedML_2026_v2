import s3fs
import os
import boto3
import zipfile

"""
AWS_S3_ENDPOINT = "minio.lab.sspcloud.fr"
print(os.environ.get('AWS_S3_BUCKET'))

# Onyxia automatically provides these environment variables
#endpoint = os.environ['AWS_S3_ENDPOINT']

# Initialize the filesystem
S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': S3_ENDPOINT_URL})
#fs = s3fs.S3FileSystem()

# List files in your bucket
# Replace 'your-bucket-name' with yours (usually 'projet-yourname')
my_bucket = "sim2023"

print(fs.ls(my_bucket))
"""
fs = s3fs.S3FileSystem()
# load data set
s3_zip_path = "https://minio.lab.sspcloud.fr/sim2023/advancedeml/kaggle_traffic_light_data_set.zip"
local_zip_path = "temp_data.zip"
extraction_dir = "data"

# 3. Téléchargement du fichier zip depuis S3
print(f"Téléchargement de {s3_zip_path}...")
fs.get(s3_zip_path, local_zip_path)

# 4. Extraction du fichier
print(f"Extraction dans le dossier '{extraction_dir}'...")
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# 5. Nettoyage (supprimer le .zip pour gagner de la place)
os.remove(local_zip_path)

print("Extraction terminée et fichier zip supprimé.")

s3_zip_path = "https://minio.lab.sspcloud.fr/sim2023/advancedeml/pedestrian_Traffic_Light_v1i_yolov11.zip"
local_zip_path = "temp_data.zip"
extraction_dir = "data"

# 3. Téléchargement du fichier zip depuis S3
print(f"Téléchargement de {s3_zip_path}...")
fs.get(s3_zip_path, local_zip_path)

# 4. Extraction du fichier
print(f"Extraction dans le dossier '{extraction_dir}'...")
if not os.path.exists(extraction_dir):
    os.makedirs(extraction_dir)

with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# 5. Nettoyage (supprimer le .zip pour gagner de la place)
os.remove(local_zip_path)

print("Extraction terminée et fichier zip supprimé.")




# load pre_trained_models: /advancedmlmodels/best.pt

s3_model_path = "s3://" + my_bucket + "/advancedmlmodels/best.pt"
local_model_path = "models/best.pt"

# 3. Download the model if it doesn't exist locally
if not os.path.exists(local_model_path):
    print(f"Downloading model from {s3_model_path}...")
    fs.get(s3_model_path, local_model_path)
    print("Download complete.")
else:
    print("Model already exists locally.")
