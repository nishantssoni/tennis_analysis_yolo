# imports
import os
import shutil
import requests

os.system('pip install ultralytics')

# downloadding yolo8x and fetching it to model
from ultralytics import YOLO
model_8 = YOLO('yolov8x')

# Downloading input data files
def download_file(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)

# URLs of the files to download
url_image = 'https://raw.githubusercontent.com/abdullahtarek/tennis_analysis/main/input_videos/image.png'
url_video = 'https://raw.githubusercontent.com/abdullahtarek/tennis_analysis/main/input_videos/input_video.mp4'

# Download and save files to current working directory
download_file(url_image, 'image.png')
download_file(url_video, 'input_video.mp4')

# Making folders and moving the downloaded files
destination_dir = 'input_videos'  # Adjusted path for Windows

# Ensure the destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Move the downloaded files to the destination directory
shutil.move('input_video.mp4', os.path.join(destination_dir, 'input_video.mp4'))
shutil.move('image.png', os.path.join(destination_dir, 'image.png'))

print("Files downloaded and moved successfully.")
