import argparse
from accelerate import DataLoaderConfiguration
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

from module.DiffDepth import ModifiedUNet, DiffDepth, SpatialDiffusionNet

def train_ddpm(args):
    rgb_path = os.path.join(args.path, 'raw_image/train')
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = (256, 1216),
        timesteps = 1000,
        sampling_timesteps = 250
    )

    trainer = Trainer(
        diffusion,
        rgb_path,
        train_batch_size = 6,
        train_lr = 8e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 4,
        ema_decay = 0.995,
        amp = True,
        split_batches=DataLoaderConfiguration(split_batches=True),
        calculate_fid = False,
        save_and_sample_every=10000
    )

    trainer.train()

def train_fsdc(args):
    few_shot_dataset = args.path

    model = ModifiedUNet(
        dim = 64,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = SpatialDiffusionNet(
        model, 
        image_size = (256, 1216),
        timesteps = 1000,
        sampling_timesteps = 250,
        loss_type = 'l2'
    )

    trainer = DiffDepth(
        diffusion,
        few_shot_dataset,
        ckpt_path = args.ddpm_ckpt,
        train_batch_size=18,
        augment_horizontal_flip=True,
        epoch = 100,
        train_lr=1e-4,
        out_size=(256, 1216),
        start_ep = 0
    )

    trainer.train()

def eval_fsdc(args):
    val_dataset = args.path

    model = ModifiedUNet(
        dim = 64,
        dim_mults = (1, 2, 4, 8, 16),
        flash_attn = True
    )

    diffusion = SpatialDiffusionNet(
        model, 
        image_size = (256, 1216),
        timesteps = 1000,
        sampling_timesteps = 250,
        loss_type = 'l2'
    )

    trainer = DiffDepth(
        diffusion,
        val_dataset,
        ckpt_path = args.ddpm_ckpt,
        train_batch_size=18,
        augment_horizontal_flip=False,
        epoch = 100,
        train_lr=1e-4,
        out_size=(256, 1216),
        start_ep = 0
    )

    trainer.val(args.fsdc_ckpt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='path/to/preprocessed/dataset')
    parser.add_argument('--mode', type=str, default='train_ddpm', help='train_ddpm, train_guidenet, eval')
    parser.add_argument('--ddpm_ckpt', type=str, default='path/to/pretrained/DDPM/checkpoint')
    parser.add_argument('--fsdc_ckpt', type=str, default='path/to/pretrained/FSDC/checkpoint')
    args = parser.parse_args()

    if args.mode == 'train_ddpm':
        train_ddpm(args)
    
    elif args.mode == 'train_fsdc':
        assert args.ddpm_ckpt is not None, 'To train FSDC model, you need to provide the path to the pretrained DDPM checkpoint!'
        train_fsdc(args)
    
    elif args.mode == 'eval':
        assert args.fsdc_ckpt is not None, 'To evaluate, you need to provide the path to the pretrained FSDC checkpoint!'
        eval_fsdc(args)