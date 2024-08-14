import cv2
import numpy as np
import os

# Hardcoded path to the folder containing images
SCENE_FOLDER = 'data/4318f8bb3c'
OUT_FOLDER = 'data/4318f8bb3c_downscaled'
SCALE = 1.5

FOLDERS = [
    # 'color',
    # 'depth',
    # 'color_train',
    'train/depth_MG',
    'train/uncertainty_MG',
]

def load_image(filepath):
    if not filepath.endswith('.npy'):
        return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    else:
        return np.load(filepath)

def write_image(filepath, image):
    if not filepath.endswith('.npy'):
        cv2.imwrite(filepath, image)
    else:
        np.save(filepath, image)

def downscale_images():

    # Iterate over all files in the folder
    for folder in FOLDERS:
        cur_folder = os.path.join(SCENE_FOLDER, folder)
        output_folder = os.path.join(OUT_FOLDER, folder)
         # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(cur_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg', 'JPG', '.npy')):  # Add other image extensions if needed
                # Construct full file path
                file_path = os.path.join(cur_folder, filename)
                # Read the image

                image = load_image(file_path)

                new_width = int(image.shape[1] / SCALE)
                new_height = int(image.shape[0] / SCALE)
                new_dimensions = (new_width, new_height)

                # Downscale the image
                downscaled_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

                # Save the downscaled image to the output directory
                output_path = os.path.join(output_folder, filename)
                write_image(output_path, downscaled_image)


if __name__ == "__main__":
    downscale_images()
