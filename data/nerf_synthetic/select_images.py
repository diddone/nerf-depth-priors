import os, shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable
#%matplotlib inline
import json
import cv2 as cv
#import pycolmap

DATASET_DIRECTORY = '.'
NUM_IMAGES_PER_SCENE = 15
OUT_IMAGES_WIDTH = 800
#scenes = [f for f in os.listdir(DATASET_DIRECTORY) if os.path.isdir(os.path.join(DATASET_DIRECTORY, f))]
scenes = ['lego']

# Source: https://github.com/opencv/opencv/blob/4.x/samples/python/camera_calibration_show_extrinsics.py
def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv

def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1,1] = 0
    M[1,2] = 1
    M[2,1] = -1
    M[2,2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))
    
def create_camera_model(focal_length, width, height, scale_focal, draw_frame_axis=False):
    fx = focal_length #camera_matrix[0,0]
    fy = focal_length #camera_matrix[1,1]
    focal = 2 / (fx + fy)
    f_scale = -scale_focal * focal   # Our cameras face towards -z 

    # draw image plane
    X_img_plane = np.ones((4,5))
    X_img_plane[0:3,0] = [-width, height, f_scale]
    X_img_plane[0:3,1] = [width, height, f_scale]
    X_img_plane[0:3,2] = [width, -height, f_scale]
    X_img_plane[0:3,3] = [-width, -height, f_scale]
    X_img_plane[0:3,4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4,3))
    X_triangle[0:3,0] = [width, height, f_scale]
    X_triangle[0:3,1] = [0, 2*height, f_scale]
    X_triangle[0:3,2] = [-width, height, f_scale]

    # draw camera
    X_center1 = np.ones((4,2))
    X_center1[0:3,0] = [0, 0, 0]
    X_center1[0:3,1] = [-width, height, f_scale]

    X_center2 = np.ones((4,2))
    X_center2[0:3,0] = [0, 0, 0]
    X_center2[0:3,1] = [width, height, f_scale]

    X_center3 = np.ones((4,2))
    X_center3[0:3,0] = [0, 0, 0]
    X_center3[0:3,1] = [width, -height, f_scale]

    X_center4 = np.ones((4,2))
    X_center4[0:3,0] = [0, 0, 0]
    X_center4[0:3,1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4,2))
    X_frame1[0:3,0] = [0, 0, 0]
    X_frame1[0:3,1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4,2))
    X_frame2[0:3,0] = [0, 0, 0]
    X_frame2[0:3,1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4,2))
    X_frame3[0:3,0] = [0, 0, 0]
    X_frame3[0:3,1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]
    
def create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
    width = board_width*square_size
    height = board_height*square_size

    # draw calibration board
    X_board = np.ones((4,5))
    #X_board_cam = np.ones((extrinsics.shape[0],4,5))
    X_board[0:3,0] = [0,0,0]
    X_board[0:3,1] = [width,0,0]
    X_board[0:3,2] = [width,height,0]
    X_board[0:3,3] = [0,height,0]
    X_board[0:3,4] = [0,0,0]

    # draw board frame axis
    # X_frame1 = np.ones((4,2))
    # X_frame1[0:3,0] = [0, 0, 0]
    # X_frame1[0:3,1] = [height/2, 0, 0]

    # X_frame2 = np.ones((4,2))
    # X_frame2[0:3,0] = [0, 0, 0]
    # X_frame2[0:3,1] = [0, height/2, 0]

    # X_frame3 = np.ones((4,2))
    # X_frame3[0:3,0] = [0, 0, 0]
    # X_frame3[0:3,1] = [0, 0, height/2]

    if draw_frame_axis:
        return [X_board, X_frame1, X_frame2, X_frame3]
    else:
        return [X_board]
    
def draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                       extrinsics, board_width, board_height=1.0, square_size=1.0,
                       patternCentric=True):
    from matplotlib import cm

    min_values = np.zeros((3,1))
    min_values = np.inf
    max_values = np.zeros((3,1))
    max_values = -np.inf

    if patternCentric:
        X_moving = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
        X_static = create_board_model(extrinsics, board_width, board_height, square_size)
    else:
        X_static = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
        X_moving = create_board_model(extrinsics, board_width, board_height, square_size)

    cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [ cm.jet(x) for x in cm_subsection ]

    for i in range(len(X_static)):
        X = np.zeros(X_static[i].shape)
        for j in range(X_static[i].shape[1]):
            X[:,j] = transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
        ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
        min_values = np.minimum(min_values, X[0:3,:].min(1))
        max_values = np.maximum(max_values, X[0:3,:].max(1))

    for idx in range(extrinsics.shape[0]):
        # R, _ = cv.Rodrigues(extrinsics[idx,0:3, 0:3])
        cMo = np.eye(4,4)
        cMo = inverse_homogeneoux_matrix(extrinsics[idx, :, :])
        # cMo[0:3,0:3] = R
        # cMo[0:3,3] = extrinsics[idx,3:6]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
            ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

    return min_values, max_values

def plot_camera_positions(images_shape, focal, extrinsics, camera_angle_x):

    #fs = cv.FileStorage(cv.samples.findFile(args.calibration), cv.FILE_STORAGE_READ)
    #board_width = int(fs.getNode('board_width').real())
    #board_height = int(fs.getNode('board_height').real())
    #square_size = fs.getNode('square_size').real()
    #camera_matrix = poses #fs.getNode('camera_matrix').mat()
    #extrinsics = poses #fs.getNode('extrinsic_parameters').mat()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect("auto")

    camera_scale = 1 / 150.0  # We do not have the pixel to world units relation
    cam_width = camera_scale * images_shape[0] / 2.0 #args.cam_width
    cam_height = camera_scale * images_shape[1] / 2.0  #args.cam_height
    focal_length = .5 * images_shape[0] / np.tan(.5 * camera_angle_x)
    scale_focal = 1.0 #focal
    min_values, max_values = draw_camera_boards(ax, focal_length, cam_width, cam_height,
                                                scale_focal, extrinsics, False)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    plt.show()
    print('Done')


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

# def run_colmap():
#     output_path: pathlib.Path
#     image_dir: pathlib.Path

#     output_path.mkdir()
#     mvs_path = output_path / "mvs"
#     database_path = output_path / "database.db"

#     pycolmap.extract_features(database_path, image_dir)
#     pycolmap.match_exhaustive(database_path)
#     maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
#     maps[0].write(output_path)
#     # dense reconstruction
#     pycolmap.undistort_images(mvs_path, output_path, image_dir)
#     pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
#     pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)


def prepare_scene(scene_path):
    train_images_path = os.path.join(scene_path, 'train')

    # Create new empty folder
    new_dir_name = os.path.join(scene_path, 'selected_images_train_' + str(OUT_IMAGES_WIDTH))
    if os.path.isdir(new_dir_name):
        shutil.rmtree(new_dir_name)
    os.mkdir(new_dir_name)

    # Load every camera position
    print(os.path.join(scene_path, 'transforms_train.json'))
    with open(os.path.join(scene_path, 'transforms_train.json')) as transforms_train_json:
        transforms_train = json.load(transforms_train_json)
    print(transforms_train['frames'][0]['transform_matrix'])
    num_cameras = len(transforms_train['frames'])
    camera_positions = np.empty(shape=[num_cameras, 3], dtype=np.float32)
    camera_extrinsics = np.empty(shape=[num_cameras, 4, 4], dtype=np.float32)
    print(camera_positions.shape)
    for i in range(num_cameras):
        transform_matrix = np.array(transforms_train['frames'][i]['transform_matrix'])
        camera_positions[i, :] = transform_matrix[:3, 3]
        camera_extrinsics[i, :, :] = transform_matrix

    # Choose N cameras that are sparsely distributed
    chosen_cameras_idx = [0]

    for i in range(NUM_IMAGES_PER_SCENE - 1):
        min_distances = np.full(shape=[num_cameras], fill_value=np.finfo('d').max, dtype=np.float32)
        for j in range(len(min_distances)):
            for k in chosen_cameras_idx:
                distance = np.linalg.norm(camera_positions[k, :] - camera_positions[j, :])
                if distance < min_distances[j]:
                    min_distances[j] = distance
        # Get the maximum minimum distance
        new_chosen_camera_idx = np.argmax(min_distances)
        chosen_cameras_idx.append(new_chosen_camera_idx)
    print(chosen_cameras_idx)
    print(camera_positions[chosen_cameras_idx, :])
    # Plot camera positions for the scene
    image_name = os.path.basename(transforms_train['frames'][0]['file_path']) + '.png'
    input_image_path = os.path.join(train_images_path, image_name)
    original_image = Image.open(input_image_path)
    original_size = original_image.size
    print(camera_extrinsics[chosen_cameras_idx, :, :].shape)
    #transforms_train['camera_angle_x']
    plot_camera_positions(original_size, 12.0, camera_extrinsics[chosen_cameras_idx, :, :], transforms_train['camera_angle_x'])

    # Rescale every image
    for image_idx in chosen_cameras_idx:
        image_name = os.path.basename(transforms_train['frames'][image_idx]['file_path']) + '.png'
        input_image_path = os.path.join(train_images_path, image_name)
        output_image_path = os.path.join(new_dir_name, image_name)
        rescale_image(input_image_path, output_image_path, OUT_IMAGES_WIDTH)


    # data = np.load('tiny_nerf_data.npz')
    # images = data['images']
    # poses = data['poses']
    # focal = data['focal']
    # H, W = images.shape[1:3]
    # print(f'Images shapes: {images.shape}. Poses: {poses.shape}. Focal length: {focal}')

    # testimg, testpose = images[101], poses[101]
    # images = images[:100,...,:3]
    # poses = poses[:100]

    # num_images = images.shape[0]
    # grid_side = 5
    # grid_size = grid_side**2
    # rand_imgs_idxs = np.random.randint(low=0, high=num_images, size=grid_size)
    # #simple axesgrid

    # import matplotlib.pyplot as plt
    # from mpl_toolkits.axes_grid1 import ImageGrid
    # import numpy as np

    # # Source: https://matplotlib.org/stable/gallery/axes_grid1/simple_axesgrid.html
    # fig = plt.figure(figsize=(12, 12))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                 nrows_ncols=(grid_side, grid_side),  # creates nxn grid of axes
    #                 axes_pad=0.1,  # pad between axes in inch.
    #                 )

    # for ax, im in zip(grid, [img for img in images[rand_imgs_idxs]]):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)
    # plt.show()

prepare_scene(os.path.join(DATASET_DIRECTORY, scenes[0]))
# for current_scene in scenes:
#     prepare_scene('./' + current_scene)




