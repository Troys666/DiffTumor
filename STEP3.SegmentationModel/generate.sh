export CUDA_VISIBLE_DEVICES=0,1

healthy_datapath=/data/st/DiffTumor/HealthyCT/healthy_ct/
datapath=/data/st/DiffTumor/Task03_Liver/
cache_dir=/data/st/DiffTumor/cache
batch_size=12
val_every=50
workers=12
organ=liver
fold=0
cache_dir=/data/st/DiffTumor/cache
# U-Net
backbone=unet
logdir="runs/$organ.fold$fold.$backbone"
datafold_dir=cross_eval/"$organ"_aug_data_fold/
dist=$((RANDOM % 99999 + 10000))
python -W ignore main_generate.py --model_name $backbone --cache_dir $cache_dir --dist-url=tcp://127.0.0.1:$dist --workers $workers --max_epochs 2000 --val_every $val_every --batch_size=$batch_size --save_checkpoint --distributed --noamp --organ_type $organ --organ_model $organ --tumor_type tumor --fold $fold --ddim_ts 50 --logdir=$logdir --healthy_data_root $healthy_datapath --data_root $datapath --datafold_dir $datafold_dir --cache_dir $cache_dir
