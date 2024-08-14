#!/bin/bash
# requirement train.csv, test.cst, rgb folder, rgb_train folder

export PROJECT_FOLDER=$(pwd)

assert_directory_exists() {
    if [ -d "$1" ]; then
        echo "Assertion passed: Directory '$1' exists."
    else
        echo "Assertion failed: Directory '$1' does not exist." >&2
        exit 1
    fi
}

if [ -z "$1" ]; then
    echo "Error: First argument is empty"
    exit 1
fi

export SCENE_ID=$1
echo $SCENE_ID
cd ../deep-image-matching/
export RGB_PATH=$PROJECT_FOLDER/data/$SCENE_ID/color/
export TR_RGB_PATH=$PROJECT_FOLDER/data/$SCENE_ID/color_train/

export EXP_TRAIN_FOLDER=exps/nerf_train/$SCENE_ID
export EXP_FOLDER=exps/nerf/$SCENE_ID
mkdir -p $EXP_TRAIN_FOLDER
mkdir -p $EXP_FOLDER

assert_directory_exists $RGB_PATH
assert_directory_exists $TR_RGB_PATH

echo "Scene Path" $SCENE_PATH
export CONDA_PATH=$(conda info --base)
$CONDA_PATH/envs/deep-image-matching/bin/python3 main.py -i $TR_RGB_PATH  -d $EXP_TRAIN_FOLDER  \
    -s bruteforce -p superpoint+superglue --force &

$CONDA_PATH/envs/deep-image-matching/bin/python3 main.py -i $RGB_PATH  -d $EXP_FOLDER  \
    -s bruteforce -p superpoint+superglue --force

cd ../nerf-depth-priors

export TRAIN_COLMAP=../deep-image-matching/$EXP_TRAIN_FOLDER/results_superpoint+superglue_bruteforce_quality_high/reconstruction
export COLMAP=../deep-image-matching/$EXP_FOLDER/results_superpoint+superglue_bruteforce_quality_high/reconstruction
assert_directory_exists $TRAIN_COLMAP
assert_directory_exists $COLMAP

# copy colmap results
mkdir -p data/$SCENE_ID/colmap/sparse/0
mkdir -p data/$SCENE_ID/colmap/sparse_train/0
cp $TRAIN_COLMAP/* data/$SCENE_ID/colmap/sparse_train/0/
cp $COLMAP/* data/$SCENE_ID/colmap/sparse/0/

# make fake_scannet
export SCANNET_FOLDER=scenes/fake_scannet/scans_test/$SCENE_ID
mkdir -p $SCANNET_FOLDER
cp -r data/$SCENE_ID/color $SCANNET_FOLDER
cp -r data/$SCENE_ID/depth $SCANNET_FOLDER

# generate config later
# and run extract scene
