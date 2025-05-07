#! /bin/sh -f


# Train Command w/o learnable uncertainty weight loss
CUDA_VISIBLE_DEVICES=4 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 3e-4 --cas_min_lr_dec 1e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.multiscale_dense.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --val_every 100 --freeze_enc_epoch 200 --output_dir ./pretrain_multiscale_dense_enc_rand_dec6 --batch_size 2

CUDA_VISIBLE_DEVICES=3 python ./main.py --lr_enc 2e-4 --cas_min_lr_enc 5e-5 --cas_final_lr_enc 3e-6 --lr_dec 5e-4 --cas_min_lr_dec 2e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.epoch10000.multiscale_dense.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --val_every 100 --freeze_enc_epoch 200 --output_dir ./pretrain_multiscale_dense_enc_rand_epoch10000_bz4_dec7 --batch_size 4

# -- Dead and Resume Command
CUDA_VISIBLE_DEVICES=4 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 3e-4 --cas_min_lr_dec 1e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 0 --load_pretrained 0 --load_resume 1 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.multiscale_dense.pth.tar --out_chkpt_file ./pretrain_multiscale_dense_enc_rand_dec6/checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --val_every 100 --freeze_enc_epoch 0 --output_dir ./pretrain_multis
cale_dense_enc_rand_dec6 --batch_size 2

# -- new pretrained epoch 10000 feature
CUDA_VISIBLE_DEVICES=3 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 3e-4 --cas_min_lr_dec 4e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.epoch10000.multiscale_dense.pth.tar --init_chkpt_file_dec ./weights/pretrained_hydranet_decoder.autoencoder.epoch10000.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_autoencoder.epoch10000.pth.tar --val_every 100 --freeze_enc_epoch 300 --output_dir ./pretrain_multiscale_dense_enc_autoencoder_dec_ep10000_bz4_0 --batch_size 4

#-------------------------------------------------------------------------------------------------------#
# Train Command w/ learnable uncertainty weight loss
#CUDA_VISIBLE_DEVICES=4 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 7e-4 --cas_min_lr_dec 9e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.multiscale_dense.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --val_every 100 --freeze_enc_epoch 200 --output_dir ./pretrain_multiscale_dense_enc_rand_dec7_uncweightloss/ --batch_size 8 --use_uncloss_weight --weight_decay_enc 6e-4 --weight_decay_dec 3e-4


CUDA_VISIBLE_DEVICES=4 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 2e-4 --cas_min_lr_dec 1e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.multiscale_dense.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --val_every 100 --freeze_enc_epoch 200 --output_dir ./pretrain_multiscale_dense_enc_rand_dec8_uncweightloss/ --batch_size 1 --use_uncloss_weight --weight_decay_enc 8e-4 --weight_decay_dec 3e-4

# -- new pretrained epoch 10000 feature
CUDA_VISIBLE_DEVICES=3 python ./main.py --lr_enc 1e-4 --cas_min_lr_enc 8e-5 --cas_final_lr_enc 3e-6 --lr_dec 3e-4 --cas_min_lr_dec 1e-5 --cas_final_lr_dec 7e-6 --cas_warmup_steps_enc 50 --cas_warmup_steps_dec 180 --max_iter 2301 --cas_T_0_enc 2301 --cas_T_0_dec 2301 --load_init 1 --load_pretrained 1 --load_resume 0 --init_chkpt_file_enc ./weights/pretrained_hydranet_encoder.320_256.epoch10000.multiscale_dense.pth.tar --init_chkpt_file_dec ./weights/pretrained_hydranet_decoder.autoencoder.epoch10000.pth.tar --out_chkpt_file checkpoint.pretrained.encoder_multiscale_dense.decoder_autoencoder.epoch10000.pth.tar --val_every 100 --freeze_enc_epoch 200 --output_dir ./pretrain_multiscale_dense_enc_autoencoder_dec_ep10000_bz1_uncweightloss_0/ --batch_size 1 --use_uncloss_weight --weight_decay_enc 8e-4 --weight_decay_dec 3e-4


# Evaluate Command
CUDA_VISIBLE_DEVICES=0 python ./eval.py --checkpoint_file ./pretrain_multiscale_dense_enc_rand_dec6/best_checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --output_dir ./pretrain_multiscale_dense_enc_rand_dec6

CUDA_VISIBLE_DEVICES=0 python ./eval.py --checkpoint_file ./pretrain_multiscale_dense_enc_rand_dec6_uncweightloss/best_checkpoint.pretrained.encoder_multiscale_dense.decoder_rand.pth.tar --output_dir ./pretrain_multiscale_dense_enc_rand_dec6_uncweightloss




