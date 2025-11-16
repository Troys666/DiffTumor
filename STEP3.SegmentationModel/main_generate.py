import os
import numpy as np
import json, time
from functools import partial
import nibabel as nb
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast #native AMP
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.transforms.transform import MapTransform
import sys
from os import environ
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from monai.inferers import sliding_window_inference
# from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,Activations,Compose

from monai import transforms
from monai import data as monai_data
from monai_trainer import AMDistributedSampler, run_training
from optimizers.lr_scheduler import WarmupCosineSchedule,LinearWarmupCosineAnnealingLR
from networks.unetr import UNETR
from networks.swin3d_unetr import SwinUNETR
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='5fold cross val')

parser.add_argument('--syn',action='store_true')
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--logdir', default=None)
parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--max_epochs', default=3000, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optim_lr', default=4e-4, type=float)

parser.add_argument('--optim_name', default='adamw', type=str)
parser.add_argument('--reg_weight', default=1e-5, type=float)

parser.add_argument('--noamp', action='store_true') #experimental
parser.add_argument('--val_every', default=200, type=int)
parser.add_argument('--val_overlap', default=0.5, type=float)
parser.add_argument('--cache_dir', default='./cache', type=str, help='directory for persistent cache')

parser.add_argument('--distributed', action='store_true') #distributed multi gpu
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23457', type=str,  help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=4, type=int)


parser.add_argument('--model_name', default='unet', type=str)
parser.add_argument('--swin_type', default='base', type=str)
parser.add_argument('--tumor_type', default='tumor', type=str)
parser.add_argument('--organ_type', default='liver', type=str)
parser.add_argument('--organ_model', default='liver', type=str)
parser.add_argument('--diff_model', default=None, type=str)
parser.add_argument('--ddim_ts', default=50, type=int)
parser.add_argument('--fg_thresh', default=10, type=int)
parser.add_argument('--healthy_num', default=400, type=int)
parser.add_argument('--healthy_seed', default=0, type=int)
#segmentation flex params
parser.add_argument('--seg_block', default='basic_pre', type=str)
parser.add_argument('--seg_num_blocks', default = '1,2,2,4', type=str)
parser.add_argument('--seg_base_filters', default=16, type=int)
parser.add_argument('--seg_relu', default='relu', type=str)
parser.add_argument('--seg_lastnorm_init_zero', action='store_true')

parser.add_argument('--seg_mode', default=1, type=int)

parser.add_argument('--seg_use_se', action='store_true')
parser.add_argument('--seg_norm_name', default='instancenorm', type=str)
parser.add_argument('--seg_noskip', action='store_true')
parser.add_argument('--seg_aug_mode', default=0, type=int)
parser.add_argument('--seg_aug_noflip', action='store_true')

parser.add_argument('--seg_norm_mode', default=0, type=int)
parser.add_argument('--seg_crop_mode', default=0, type=int)

#unetr params
parser.add_argument('--pos_embedd', default='conv', type=str)
parser.add_argument('--norm_name', default='instance', type=str)
parser.add_argument('--num_steps', default=40000, type=int)
parser.add_argument('--eval_num', default=500, type=int)
parser.add_argument('--warmup_steps', default=500, type=int)
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--mlp_dim', default=3072, type=int)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--in_channels', default=1, type=int)
parser.add_argument('--out_channels', default=3, type=int)
parser.add_argument('--num_classes', default=3, type=int)
parser.add_argument('--res_block', action='store_true')
parser.add_argument('--conv_block', action='store_true')
parser.add_argument('--roi_x', default=96, type=int)
parser.add_argument('--roi_y', default=96, type=int)
parser.add_argument('--roi_z', default=96, type=int)
parser.add_argument('--dropout_rate', default=0.0, type=float)
parser.add_argument('--decay', default=1e-5, type=float)
parser.add_argument('--lrdecay', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--amp_scale', action='store_true')
parser.add_argument('--opt_level', default='O2', type=str)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lrschedule', default='warmup_cosine', type=str)
parser.add_argument('--warmup_epochs', default=100, type=int)
parser.add_argument('--resume_ckpt', action='store_true')
parser.add_argument('--pretrained_dir', default=None, type=str)
parser.add_argument('--data_root', default=None, type=str)
parser.add_argument('--healthy_data_root', default=None, type=str)
parser.add_argument('--datafold_dir', default=None, type=str)
parser.add_argument('--cache_num', default=200, type=int)

parser.add_argument('--use_pretrained', action='store_true')
parser.add_argument('--save_dir', default='./generated_samples', type=str, help='directory to save generated images')

class RandCropByPosNegLabeld_select(transforms.RandCropByPosNegLabeld):
    def __init__(self, keys, label_key, spatial_size, 
                 pos=1.0, neg=1.0, num_samples=1, 
                 image_key=None, image_threshold=0.0, allow_missing_keys=True,
                   fg_thresh=0):
        super().__init__(keys=keys, label_key=label_key, spatial_size=spatial_size, 
                 pos=pos, neg=neg, num_samples=num_samples, 
                 image_key=image_key, image_threshold=image_threshold, allow_missing_keys=allow_missing_keys)
        self.fg_thresh = fg_thresh

    def R2voxel(self,R):
        return (4/3*np.pi)*(R)**(3)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']
        d.pop('name')
        if 'kidney_label' in data_name or 'liver_label' in data_name or 'pancreas_label' in data_name:
            flag=0
            while 1:
                flag+=1
                d_crop = super().__call__(d)
                pixel_num = (d_crop[0]['label']>0).sum()
                # print(pixel_num)
                if pixel_num > self.R2voxel(self.fg_thresh):
                    break
                if flag>5 and pixel_num > self.R2voxel(max(self.fg_thresh-5, 5)):
                    break
                if flag>10 and pixel_num > self.R2voxel(max(self.fg_thresh-10, 5)):
                    break
                if flag>15 and pixel_num > self.R2voxel(max(self.fg_thresh-15, 5)):
                    break
                if flag>20 and pixel_num > self.R2voxel(max(self.fg_thresh-20, 5)):
                    break
                if flag>25 and pixel_num > self.R2voxel(max(self.fg_thresh-25, 5)):
                    break
                if flag>30:
                    break
        else:
            d_crop = super().__call__(d)
        d_crop[0]['name'] = data_name

        return d_crop

class LoadImage_train(MapTransform):
    def __init__(self,organ_type):
        self.reader1 = transforms.LoadImaged(keys=["image", "label"])
        self.organ_type = organ_type

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        if (not 'kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            d['label'][d['label']==3] = 1
        elif ('kidney_label' in data_name) and self.organ_type == 'kidney':
            d = self.reader1.__call__(d)
            d['label'][d['label']>0] = 1
            
        else :
            d = self.reader1.__call__(d)

        return d
    
class LoadImage_val(transforms.LoadImaged):
    def __init__(self, keys, *args,**kwargs, ):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        data_name = d['name']

        d = super().__call__(d)
        d['label'][d['label']==3] = 1

        return d
    
def save_synt_img(synt_data, synt_target, data_name, synt_type, save_dir, organ_type):
    """
    Save synthetic image and label as .nii.gz and visualization slices as .png.
    Saves axial, sagittal, and coronal views for comprehensive visualization.
    """
    # Sanitize data_name to be a valid filename
    safe_data_name = data_name.replace('/', '_')

    # Define paths for NIfTI files
    nii_img_path = os.path.join(save_dir, f"{safe_data_name}_{synt_type}_img.nii.gz")
    nii_lbl_path = os.path.join(save_dir, f"{safe_data_name}_{synt_type}_lbl.nii.gz")

    # Ensure the full directory path exists before saving
    os.makedirs(os.path.dirname(nii_img_path), exist_ok=True)

    # Define custom colormap: background transparent, organ green, tumor red
    colors = [(0, 0, 0, 0), (0, 1, 0, 0.5), (1, 0, 0, 0.5)]  # R, G, B, A
    cmap = ListedColormap(colors)

    # Convert to numpy arrays
    img_np = synt_data.squeeze().cpu().numpy()
    lbl_np = synt_target.squeeze().cpu().numpy()

    # Save as NIfTI
    nii_img = nb.Nifti1Image(img_np, affine=np.eye(4))
    nii_lbl = nb.Nifti1Image(lbl_np, affine=np.eye(4))
    nb.save(nii_img, nii_img_path)
    nb.save(nii_lbl, nii_lbl_path)

    # Save multi-view visualization (axial, sagittal, coronal)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Axial view (z-axis)
    axial_idx = img_np.shape[2] // 2
    axes[0, 0].imshow(img_np[:, :, axial_idx], cmap='gray')
    axes[0, 0].set_title(f'Axial - Image')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(img_np[:, :, axial_idx], cmap='gray')
    axes[1, 0].imshow(lbl_np[:, :, axial_idx], cmap=cmap, interpolation='none')
    axes[1, 0].set_title(f'Axial - Overlay')
    axes[1, 0].axis('off')
    
    # Sagittal view (x-axis)
    sagittal_idx = img_np.shape[0] // 2
    axes[0, 1].imshow(img_np[sagittal_idx, :, :].T, cmap='gray', origin='lower')
    axes[0, 1].set_title(f'Sagittal - Image')
    axes[0, 1].axis('off')
    axes[1, 1].imshow(img_np[sagittal_idx, :, :].T, cmap='gray', origin='lower')
    axes[1, 1].imshow(lbl_np[sagittal_idx, :, :].T, cmap=cmap, interpolation='none', origin='lower')
    axes[1, 1].set_title(f'Sagittal - Overlay')
    axes[1, 1].axis('off')
    
    # Coronal view (y-axis)
    coronal_idx = img_np.shape[1] // 2
    axes[0, 2].imshow(img_np[:, coronal_idx, :].T, cmap='gray', origin='lower')
    axes[0, 2].set_title(f'Coronal - Image')
    axes[0, 2].axis('off')
    axes[1, 2].imshow(img_np[:, coronal_idx, :].T, cmap='gray', origin='lower')
    axes[1, 2].imshow(lbl_np[:, coronal_idx, :].T, cmap=cmap, interpolation='none', origin='lower')
    axes[1, 2].set_title(f'Coronal - Overlay')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'{safe_data_name} - {synt_type} tumor ({organ_type})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{safe_data_name}_{synt_type}_multiview.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved synthetic {synt_type} tumor for {safe_data_name} to {save_dir}")

def _get_transform(args):

    train_transform = transforms.Compose(
    [
        LoadImage_train(args.organ_type),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        transforms.SpatialPadd(keys=["image", "label"], mode=["minimum", "constant"], spatial_size=[96, 96, 96]),
        RandCropByPosNegLabeld_select(
            keys=["image", "label", "name"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
            fg_thresh = args.fg_thresh,
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )

    val_transform = transforms.Compose(
        [
            LoadImage_val(keys=["image", "label", "organ_pseudo"]),
            transforms.AddChanneld(keys=["image", "label", "organ_pseudo"]),
            transforms.Orientationd(keys=["image", "label", "organ_pseudo"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label", "organ_pseudo"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest", "nearest")),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            transforms.SpatialPadd(keys=["image", "label", "organ_pseudo"], mode=["minimum", "constant", "constant"], spatial_size=[96, 96, 96]),
            transforms.ToTensord(keys=["image", "label", "organ_pseudo"]),
        ]
    )
    
    return train_transform, val_transform

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    
    print("MAIN Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)

        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))

    else:
        # Simply call main_worker function
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True) #in new Pytorch/python labda functions fail to pickle with spawn
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    print(args.rank, ' gpu', args.gpu)
    if args.rank==0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    roi_size = [args.roi_x, args.roi_y, args.roi_z]
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    
    data_root = args.data_root
    healthy_data_root = args.healthy_data_root
    datafold_dir = args.datafold_dir
    fold = args.fold
    tumor_type = args.tumor_type
    organ_type = args.organ_type
    if organ_type == 'liver':
        args.fg_thresh = 30
    elif organ_type == 'pancreas':
        args.fg_thresh = 15
    elif organ_type == 'kidney':
        args.fg_thresh = 25
    train_transform, val_transform = _get_transform(args)

    ## NETWORK
    if (args.model_name is None) or args.model_name == 'unet':
        from monai.networks.nets import UNet 
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
        
    elif args.model_name == 'swinunetr':
        
        if args.swin_type == 'tiny':
            feature_size=12
        elif args.swin_type == 'small':
            feature_size=24
        elif args.swin_type == 'base':
            feature_size=48

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=3,
                          img_size=(96, 96, 96),
                          feature_size=feature_size,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7])
        
        if args.use_pretrained:
            pretrained_add = 'model_swinvit.pt'
            model.load_from(weights=torch.load(pretrained_add))
            print('Use pretrained ViT weights from: {}'.format(pretrained_add))
    elif args.model_name == 'nnunet':
        from monai.networks.nets import DynUNet
        from dynunet_pipeline.create_network import get_kernels_strides
        from dynunet_pipeline.task_params import deep_supr_num
        task_id = 'custom'
        kernels, strides = get_kernels_strides(task_id)
        model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=False,
            deep_supr_num=deep_supr_num[task_id],
        )
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))
        
    if args.resume_ckpt:
        model_dict = torch.load(args.pretrained_dir)
        model.load_state_dict(model_dict['state_dict'])
        print('Use pretrained weights')


    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)
    val_channel_names=['val_liver_dice', 'val_tumor_dice']

    print('Crop size', roi_size)

    train_img_real=[]
    train_lbl_real=[]
    train_name_real=[]
    train_img_healthy=[]
    train_lbl_healthy=[]
    train_name_healthy=[]

    train_txt = os.path.join(datafold_dir, 'real_{}_train_{}.txt'.format(tumor_type, fold))

    for line in open(train_txt):
        name = line.strip().split()[1].split('.')[0]

        if 'kidney_label' in name or 'liver_label' in name or 'pancreas_label' in name:
            train_img_real.append(healthy_data_root + line.strip().split()[0])
            train_lbl_real.append(healthy_data_root + line.strip().split()[1])
            train_name_real.append(name)
        else:
            train_img_healthy.append(data_root + line.strip().split()[0])
            train_lbl_healthy.append(data_root + line.strip().split()[1])
            train_name_healthy.append(name)
    
    train_img = train_img_real + train_img_healthy
    train_lbl = train_lbl_real + train_lbl_healthy
    train_name = train_name_real + train_name_healthy
    data_dicts_train = [{'image': image, 'label': label, 'name': name}
            for image, label, name in zip(train_img, train_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    val_img=[]
    val_lbl=[]
    val_name=[]
    val_pseudo_lbl=[]
    for line in open(os.path.join(datafold_dir, 'real_{}_val_{}.txt'.format(tumor_type, fold))):
        name = line.strip().split()[1].split('.')[0]
        val_img.append(data_root + line.strip().split()[0])
        val_lbl.append(data_root + line.strip().split()[1])
        val_pseudo_lbl.append('organ_pseudo_swin_new/{}/'.format(organ_type) + os.path.basename(line.strip().split()[1]))
        val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'organ_pseudo': organ_pseudo, 'name': name}
                for image, label, organ_pseudo, name in zip(val_img, val_lbl, val_pseudo_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))

    
    val_shape_dict = {}
    for d in data_dicts_val:
        imagepath = d["image"]
        imagename = imagepath.split('/')[-1]
        imgnb = nb.load(imagepath)
        val_shape_dict[imagename] = [imgnb.shape[0], imgnb.shape[1], imgnb.shape[2]]

    train_ds = monai_data.PersistentDataset(data=data_dicts_train, transform=train_transform, cache_dir=args.cache_dir)
    train_sampler = AMDistributedSampler(train_ds) if args.distributed else None
    train_loader = monai_data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, sampler=train_sampler, pin_memory=True)


    val_ds = monai_data.Dataset(data=data_dicts_val, transform=val_transform)
    val_sampler = AMDistributedSampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = monai_data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers//2, sampler=val_sampler, pin_memory=True)

    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=args.val_overlap, mode='gaussian')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)


   
    best_acc = 0
    start_epoch = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))


    model.cuda(args.gpu)
    

    
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name=='batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu) #??

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name=='sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=0.99, nesterov=True, weight_decay=args.reg_weight) #momentum 0.99, nestorov=True, following nnUnet
    else:
        raise ValueError('Unsupported optim_name' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )


    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)

    else:
        scheduler = None


    from TumorGeneration.utils import synthesize_early_tumor, synthesize_medium_tumor, synthesize_large_tumor, synt_model_prepare
    import random
    if args.organ_type == 'liver':
        sample_thresh = 0.5
    elif args.organ_type == 'pancreas':
        sample_thresh = 0.5
    elif args.organ_type == 'kidney':
        sample_thresh = 0.5
    # model prepare
    vqgan, early_sampler, noearly_sampler= synt_model_prepare(device = torch.device("cuda", args.rank), fold=args.fold, organ=args.organ_model)

    for idx, batch_data in enumerate(train_loader):

        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target, data_names = batch_data['image'], batch_data['label'], batch_data['name']
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        for bs in range(data.shape[0]):
            data_name = data_names[bs]
            if 'kidney_label' in data_name or 'liver_label' in data_name or 'pancreas_label' in data_name:
                if random.random() > sample_thresh:
                    healthy_data = data[bs][None,...]
                    healthy_target = target[bs][None,...]
                    tumor_types = ['early', 'medium', 'large']
                    tumor_probs = np.array([0.8, 0.1, 0.1])
                    synthetic_tumor_type = np.random.choice(tumor_types, p=tumor_probs.ravel())
                    if synthetic_tumor_type == 'early':
                        synt_data, synt_target = synthesize_early_tumor(healthy_data, healthy_target, args.organ_type, vqgan, early_sampler)
                    elif synthetic_tumor_type == 'medium':
                        synt_data, synt_target = synthesize_medium_tumor(healthy_data, healthy_target, args.organ_type, vqgan, noearly_sampler, ddim_ts=args.ddim_ts)
                    elif synthetic_tumor_type == 'large':
                        synt_data, synt_target = synthesize_large_tumor(healthy_data, healthy_target, args.organ_type, vqgan, noearly_sampler, ddim_ts=args.ddim_ts)
                    
                    # Save the generated image for visualization
                    if args.rank == 0: # Ensure saving is done only by the main process
                        save_synt_img(synt_data.detach().cpu(), synt_target.detach().cpu(), data_name, synthetic_tumor_type, args.save_dir, args.organ_type)

                    data[bs,...] = synt_data[0]
                    target[bs,...] = synt_target[0]
        data=data.detach()
        target=target.detach()

    


if __name__ == '__main__':
    main()
