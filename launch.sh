#!/bin/bash

# python3 run_nerfacc.py train --scene_id full_scene_downscaled  --data_dir ./data/ --depth_prior_network_path \
#     ./data/20211027_092436.tar --ckpt_dir ./ckpt/  \
#     --max_num_rays 1024 --occ_resolution 128 --render_step_size 1e-3 --occ_num_levels 32 \
#     --model_type ngp --netwidth 64 --i_print 1000 --i_img 10000 \
#     --log2_hashmap_size 24 --N_training_steps 20000  --lrate 5e-3 \
#     --ngp_n_levels 32 --ngp_max_resolution 524288 \
#     --depth_loss_weight 0 --is_full_scene

python3 run_nerfacc.py train --scene_id 4318f8bb3c_downscaled  --data_dir ./data/ --depth_prior_network_path \
    ./data/20211027_092436.tar --ckpt_dir ./ckpt/  \
    --max_num_rays 1024 --occ_resolution 128 --render_step_size 1e-3 --occ_num_levels 16 \
    --model_type ngp --netwidth 64 --i_print 1000 --i_img 3000 \
    --log2_hashmap_size 22 --N_training_steps 15000  --lrate 5e-3 --ngp_max_resolution 524288

python3 run_nerfacc.py train --scene_id 4318f8bb3c_downscaled  --data_dir ./data/ --depth_prior_network_path \
    ./data/20211027_092436.tar --ckpt_dir ./ckpt/  \
    --max_num_rays 1024 --occ_resolution 128 --render_step_size 1e-3 --occ_num_levels 16 \
    --model_type ngp --netwidth 64 --i_print 1000 --i_img 3000 \
    --log2_hashmap_size 22 --N_training_steps 15000  --lrate 5e-3 --ngp_max_resolution 524288 --depth_completion marigold


