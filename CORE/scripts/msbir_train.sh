set -ex
python3 /home/xinyi/FastFOD-Net/CORE/train_model.py \
--dataroot /home/xinyi/MSBIR/30dir_b1000_fod_bbox/ \
--maskroot /home/xinyi/MSBIR/brainmasks_vox1.25/ \
--gtroot /home/xinyi/MSBIR/fod_bbox/ \
--checkpoints_dir /home/xinyi/checkpoints3/checkpoints_hcp_sr_global \
--name msbir_sr_fastfodnet \
--normalization_mode z-score_v4 \
--model re \
--input_nc 45 \
--output_nc 45 \
--init_type kaiming \
--dataset_mode fod_re \
--num_threads 0 \
--batch_size 4 \
--beta1 0.99 \
--lr 0.001 \
--n_epochs 50 \
--print_freq 10000 \
--save_latest_freq 100000 \
--save_epoch_freq 50 \
--gpu_ids 0 \
--conv_type fastfodnet \
--test_fold 0 \
--phase train \
--index_pattern 'AU-\d{3}-\d{4}_\d{6}-bl' \
--sample_suffix "_WMfod_30dir_b1000_norm.mif.gz" \
--sample_gt_suffix "_WMfod_norm.mif.gz" \
--foldroot /home/xinyi/MSBIR/folds/ \
# --phase splitfolds \
