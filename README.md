# Dense Depth Priors for Efficient NeRF from Sparse Input Views using Depth Estimation
This repository contains the an expansion build over the CVPR 2022 paper: Dense Depth Priors for Neural Radiance Fields from Sparse Input Views.

[Arxiv](https://arxiv.org/abs/2112.03288) | [Video](https://t.co/zjH9JvkuQq) | [Project Page](https://barbararoessle.github.io/dense_depth_priors_nerf/)

![](docs/static/images/modified_pipeline.png)

## Step 1: Obtain Dense Depth Priors 

### Prepare ScanNet++
Download a scene from the ScanNet++ dataset, select the desired images and undistort them.

### Compute camera parameters
Run the [SuperPoint](https://github.com/rpautrat/SuperPoint) keypoint detector and the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) feature matching. Then run [COLMAP](https://github.com/colmap/colmap) bundle adjustment step on all RGB images of ScanNet++. 
For this, the RGB files need to be isolated from the other scene data, f.ex. create a temporary directory `tmp` and copy each `<scene>/color/<rgb_filename>` to `tmp/<scene>/color/<rgb_filename>`. 
Then run: 
```
colmap feature_extractor  --database_path scannet_sift_database.db --image_path tmp
```
When working with different relative paths or filenames, the database reading in `scannet_dataset.py` needs to be adapted accordingly. 

### Create configuration

Run the notebook `depth_alignment\generate_config.ipynb` to generate the `config.json` file needed to relate COLMAP scale to metric scale.

### Depth Estimation using Marigold

Run the Google Colab notebook `depth_estimation\estimate_depth_marigold.ipynb` to obtain monocular affine-invariant depth predictions for all the images in the scene.

## Step 2: Optimizing NeRF with Dense Depth Priors
### Prepare scenes
You can skip the scene preparation and directly download the [scene](https://drive.google.com/drive/folders/1jiR3_yF9KpfL0wa5I5URfykS1_EBg81d?usp=sharing). 
To prepare a scene and render sparse depth maps from COLMAP sparse reconstructions, run: 
```
cd preprocessing
mkdir build
cd build
cmake ..
make -j
./extract_scannet_scene <path to scene> <path to ScanNet>
```
The scene directory must contain the following:
- `train.csv`: list of training views from the ScanNet scene
- `test.csv`: list of test views from the ScanNet scene
- `config.json`: parameters for the scene:
  - `name`: name of the scene
  - `max_depth`: maximal depth value in the scene, larger values are invalidated
  - `dist2m`: scaling factor that scales the sparse reconstruction to meters
  - `rgb_only`: write RGB only, f.ex. to get input for COLMAP
- `colmap`: directory containing 2 sparse reconstruction:
  - `sparse`: reconstruction run on train and test images together to determine the camera poses
  - `sparse_train`, reconstruction run on train images alone to determine the sparse depth maps.  

Please check the provided scenes as an example. 
The option `rgb_only` is used to preprocess the RGB images before running COLMAP. This cuts dark image borders from calibration, which harm the NeRF optimization. It is essential to crop them before running COLMAP to ensure that the determined intrinsics match the cropped RGB images. 

### Depth Prior Alignment

To obtain metric depth prior run the notebook `depth_alignment\align_depth_map_MG_scannetpp.ipynb` to transform the affine-invariant depth prior to metric scale.

### Optimize
```
python3 run_nerf.py train --scene_id <scene, e.g. scene0710_00> --data_dir <directory containing the scenes> --depth_prior_network_path <path to depth prior checkpoint> --ckpt_dir <path to write checkpoints>
```
Checkpoints are written into a subdirectory of the provided checkpoint directory. The subdirectory is named by the training start time in the format `jjjjmmdd_hhmmss`, which also serves as experiment name in the following. 

### Test
```
python3 run_nerf.py test --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints>
```
The test results are stored in the experiment directory. 
Running `python3 run_nerf.py test_opt ...` performs test time optimization of the latent codes before computing the test metrics. 

### Render Video
```
python3 run_nerf.py video  --expname <experiment name> --data_dir <directory containing the scenes> --ckpt_dir <path to write checkpoints>
```
The video is stored in the experiment directory. 

---

### Citation
If you find this repository useful, you can cite the original paper we build on: 
```
@inproceedings{roessle2022depthpriorsnerf,
    title={Dense Depth Priors for Neural Radiance Fields from Sparse Input Views}, 
    author={Barbara Roessle and Jonathan T. Barron and Ben Mildenhall and Pratul P. Srinivasan and Matthias Nie{\ss}ner},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022}
```

### Acknowledgements
We thank [Dense Depth Priors NeRF](https://github.com/barbararoessle/dense_depth_priors_nerf) which we use as a baseline, additionally we thank [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [CSPN](https://github.com/XinJCheng/CSPN), from which the original repository borrows code. 
