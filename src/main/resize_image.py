import os
from PIL import Image

# Target directory and size
folder_path = "/home/airl010/1_Thesis/visionNav/fusion/dataset/IISc_drive/image_02/data"
target_size = (1242, 375)

# List all .png files
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    try:
        img = Image.open(file_path)
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        img_resized.save(file_path)
        print(f"Resized and replaced: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {e}")
