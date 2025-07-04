set -ex
python3 /home/xinyi/FastFOD-Net/CORE/train_model.py \
--dataroot /home/xinyi/MSBIR/30dir_b1000_fod_bbox/ \
--maskroot /home/xinyi/MSBIR/brainmasks_vox1.25/ \
--gtroot /home/xinyi/MSBIR/fod_bbox/ \
--checkpoints_dir /home/xinyi/checkpoints3/checkpoints_hcp_sr_global \
--name msbir_sr_fastfodnet \
--gpu_ids 0 \
--index_pattern 'AU-\d{3}-\d{4}_\d{6}-bl' \
--sample_suffix "_WMfod_30dir_b1000_norm.mif.gz" \
--sample_gt_suffix "_WMfod_norm.mif.gz" \
--foldroot /home/xinyi/MSBIR/folds_test1/ \
--phase splitfolds \
