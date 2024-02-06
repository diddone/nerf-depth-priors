import os, shutil
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
#from tqdm import tqdm_notebook as tqdm
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
#%matplotlib inline
import json
#import cv2 as cv
from scipy.spatial.transform import Rotation
import subprocess, sys

DATASET_DIRECTORY = '.'
NUM_IMAGES_PER_SCENE = 15
OUT_IMAGES_WIDTH = 800
#scenes = [f for f in os.listdir(DATASET_DIRECTORY) if os.path.isdir(os.path.join(DATASET_DIRECTORY, f))]
scenes = ['lego']

def matrix_to_quaternion_translation(matrix):
    # Extract the rotation part (3x3 submatrix)
    rotation_matrix = matrix[0:3, 0:3]

    # Convert the rotation matrix to a quaternion (qx, qy, qz, qw)
    rotation = Rotation.from_matrix(rotation_matrix.transpose())
    quaternion = rotation.as_quat()

    # Extract the translation vector
    translation_vector = np.matmul(rotation_matrix.transpose(), (matrix[0:3, 3] / matrix[3, 3]))
    return quaternion, translation_vector

def rescale_image(input_path, output_path, new_width):
    # Open the image file
    original_image = Image.open(input_path)

    # Get the original size
    original_size = original_image.size

    # Calculate the new size after rescaling
    scale_factor = original_size[0] / float(new_width)
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

    # Resize the image
    rescaled_image = original_image.resize(new_size)

    # Save the rescaled image
    rescaled_image.save(output_path)
    return new_size

def run_colmap_on_scene(scene_path):
    train_images_path = os.path.join(scene_path, 'train')
    print('Reading camera parameters...')

    # Create new empty folder
    resized_dir_name = os.path.join(scene_path, 'train_' + str(OUT_IMAGES_WIDTH))
    if os.path.isdir(resized_dir_name):
        shutil.rmtree(resized_dir_name)
    os.mkdir(resized_dir_name)

    # Load every camera position
    with open(os.path.join(scene_path, 'transforms_train.json')) as transforms_train_json:
        transforms_train = json.load(transforms_train_json)
    
    num_cameras = len(transforms_train['frames'])
    camera_positions = np.empty(shape=[num_cameras, 3], dtype=np.float32)
    camera_extrinsics = np.empty(shape=[num_cameras, 4, 4], dtype=np.float32)
    
    for i in range(num_cameras):
        transform_matrix = np.array(transforms_train['frames'][i]['transform_matrix'])
        camera_positions[i, :] = transform_matrix[:3, 3]
        camera_extrinsics[i, :, :] = transform_matrix
    print('Reading camera parameters done')

    # Rescale every image
    print('Rescaling images...')
    for image_idx in range(num_cameras):
        image_name = os.path.basename(transforms_train['frames'][image_idx]['file_path']) + '.png'
        input_image_path = os.path.join(train_images_path, image_name)
        output_image_path = os.path.join(resized_dir_name, image_name)
        new_size = rescale_image(input_image_path, output_image_path, OUT_IMAGES_WIDTH)
    print('Rescaling done')

    OUT_IMAGES_HEIGHT = new_size[1]
    # Prepare colmap files
    print('Preparing colmap files...')
    colmap_dir_path = os.path.join(scene_path, 'colmap')
    if os.path.isdir(colmap_dir_path):
        shutil.rmtree(colmap_dir_path)
    os.mkdir(colmap_dir_path)

    # Points file (empty)
    f = open(os.path.join(colmap_dir_path, 'points3D.txt'), 'w')
    f.close()
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # Number of cameras: 3
    # Note: for some reason colmap doesn't seem to care about this file at all
    f = open(os.path.join(colmap_dir_path, 'cameras.txt'), 'w')
    camera_id = 1
    focal_length = (0.5 * OUT_IMAGES_WIDTH) / np.tan(0.5 * transforms_train['camera_angle_x'])
    # 1111.111 400 400
    f.write(str(camera_id) + ' SIMPLE_PINHOLE ' + str(OUT_IMAGES_WIDTH) + ' ' + str(OUT_IMAGES_HEIGHT) + ' ' + str(focal_length) + ' ' + \
            str(OUT_IMAGES_WIDTH / 2.0) + ' ' + str(OUT_IMAGES_HEIGHT / 2.0) + '\n')
    f.close()
    # Images file (every other line is empty)
    # Image list with two lines of data per image:
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # POINTS2D[] as (X, Y, POINT3D_ID)
    f = open(os.path.join(colmap_dir_path, 'images.txt'), 'w')
    for i in range(num_cameras):
        image_id = i
        image_name = os.path.basename(transforms_train['frames'][i]['file_path']) + '.png'
        q, t = matrix_to_quaternion_translation(camera_extrinsics[i, :, :])
        f.write(str(image_id) + ' ' + str(q[3]) + ' ' + str(q[0]) + ' ' + str(q[1]) + ' ' + str(q[2]) + ' ' + \
                str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]) + ' ' + str(camera_id) + ' ' + str(image_name) + '\n')
        f.write('\n')
    f.close()
    print('Colmap files ready')
    
    # Run colmap commands for extracting the points in the scene
    print('Running colmap commands...')
    # Feature extractor command
    feature_extrator_command = 'colmap feature_extractor --database_path ' + os.path.join(colmap_dir_path, 'database.db')\
                                + ' --image_path ' + resized_dir_name\
                                + ' --ImageReader.camera_model SIMPLE_PINHOLE'\
                                + ' --ImageReader.single_camera 1'\
                                + ' --ImageReader.single_camera_per_folder 1'\
                                + ' --ImageReader.default_focal_length_factor 1'\
                                + ' --ImageReader.camera_params ' + str(focal_length) + ',' + str(OUT_IMAGES_WIDTH / 2.0) + ',' + str(OUT_IMAGES_HEIGHT / 2.0)
    subprocess.run(feature_extrator_command, shell = True, executable="/bin/bash")
    # Exhaustive matcher command
    matcher_command = 'colmap exhaustive_matcher --database_path ' + os.path.join(colmap_dir_path, 'database.db')
    subprocess.run(matcher_command, shell = True, executable="/bin/bash")
    # Triangulation command
    triangulation_dir_path = os.path.join(colmap_dir_path, 'colmap')
    if os.path.isdir(triangulation_dir_path):
        shutil.rmtree(triangulation_dir_path)
    os.mkdir(triangulation_dir_path)
    triangulation_command = 'colmap point_triangulator --database_path ' + os.path.join(colmap_dir_path, 'database.db')\
                                + ' --image_path ' + resized_dir_name + ' --input_path ' + colmap_dir_path\
                                + ' --output_path ' + triangulation_dir_path
    subprocess.run(triangulation_command, shell = True, executable="/bin/bash")
    print('Colmap commands done')

# Main program
if __name__ == "__main__":
    for current_scene in scenes:
        run_colmap_on_scene(os.path.join(DATASET_DIRECTORY, current_scene))