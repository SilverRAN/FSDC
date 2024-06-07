# [[CVPRW 2023] Few-Shot Depth Completion Using Denoising Diffusion Probabilistic Model](https://openaccess.thecvf.com/content/CVPR2023W/PCV/html/Ran_Few-Shot_Depth_Completion_Using_Denoising_Diffusion_Probabilistic_Model_CVPRW_2023_paper.html)
Weihang Ran, Wei Yuan, Ryosuke Shibasaki

## Get dataset
1.Download dataset from [KITTI homepage](https://www.cvlibs.net/datasets/kitti/index.php)

2.Organize the raw data and groundtruth as follows:

    KITTI
    ├── raw_image
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync
    │   │   ├── 2011_09_26_drive_0009_sync
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync
    │   │   ├── 2011_09_26_drive_0005_sync
    │   │   ├── ...
    ├── velodyne_raw
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync
    │   │   ├── 2011_09_26_drive_0009_sync
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync
    │   │   ├── 2011_09_26_drive_0005_sync
    │   │   ├── ...
    ├── groundtruth
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync
    │   │   ├── 2011_09_26_drive_0009_sync
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync
    │   │   ├── 2011_09_26_drive_0005_sync
    │   │   ├── ...

3.Run

    python prepare_dataset.py --current_dir /path/to/current/data --target_dir /path/for/saving/preprocessed/data

After running this code, the preprocessed dataset should be organized like following:

    KITTI
    ├── raw_image
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync_image_0000000000_image_02.png
    │   │   ├── 2011_09_26_drive_0001_sync_image_0000000000_image_03.png
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync_image_0000000000_image_02.png
    │   │   ├── 2011_09_26_drive_0002_sync_image_0000000000_image_03.png
    │   │   ├── ...
    ├── velodyne_raw
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync_velodyne_raw_0000000005_image_02.png
    │   │   ├── 2011_09_26_drive_0001_sync_velodyne_raw_0000000005_image_03.png
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync_velodyne_raw_0000000005_image_02.png
    │   │   ├── 2011_09_26_drive_0001_sync_velodyne_raw_0000000005_image_03.png
    │   │   ├── ...
    ├── groundtruth
    │   ├── train
    │   │   ├── 2011_09_26_drive_0001_sync_groundtruth_depth_0000000005_image_02.png
    │   │   ├── 2011_09_26_drive_0001_sync_groundtruth_depth_0000000005_image_03.png
    │   │   ├── ...
    │   ├── val
    │   │   ├── 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png
    │   │   ├── 2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_03.png
    │   │   ├── ...


## Install
### 1.install necessary dependencies
    conda env create -f environment.yaml
    conda activate fsdc
### 2.install the denoising_diffusion_pytorch package
    git clone https://github.com/lucidrains/denoising-diffusion-pytorch.git
    cd denoising-diffusion-pytorch
    pip install -v -e .
### 3.build GuideConv
    git clone https://github.com/kakaxi314/GuideNet.git
    cd GuideNet/exts
    python setup.py install


## Training
### 1.train DDPM on KITTI RGB images
    accelerate launch main.py --path /Path/to/your/preprocessed_KITTI --mode train_ddpm

After training the checkpoint will be saved in ./results/
### 2.train FSDC on Few-shot dataset
    accelerate launch main.py --path /Path/to/your/preprocessed_KITTI --mode train_fsdc --ddpm_ckpt /Path/to/trained/DDPM/checkpoint

After training the checkpoint will also be saved in ./results/

You can also directly download our pretrained DDPM checkpoint on KITTI dataset from [here](https://drive.google.com/file/d/1OEmQ0WZxJqj29KyrprzNDyrwbaqJohVf/view?usp=drive_link)

## Evaluation
### run
    accelerate launch main.py --path /Path/to/your/preprocessed_KITTI --mode eval --ddpm_ckpt /Path/to/trained/DDPM/checkpoint --fsdc_ckpt /Path/to/trained/FSDC/checkpoint

## Reference
[Denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch.git)

[GuideNet](https://github.com/kakaxi314/GuideNet.git)

[SemAttNet](https://github.com/danishnazir/SemAttNet.git)

## Citation
If this repo is helpful to you, please cite our work:

    @inproceedings{ran2023few,
    title={Few-Shot Depth Completion Using Denoising Diffusion Probabilistic Model},
    author={Ran, Weihang and Yuan, Wei and Shibasaki, Ryosuke},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={6558--6566},
    year={2023}
    }
