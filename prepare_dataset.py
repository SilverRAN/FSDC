"""
Rename train files and move them to a new folder.
"""
import argparse
import os
import shutil
from tqdm import tqdm
import random

def del_useless_folder(rgb_dir_path):
    """
    Since the velodyne data only has corresponding data for image_02 and image_03, the other folders can be deleted.
    """
    for folder in os.listdir(rgb_dir_path):
        assert os.path.isdir(os.path.join(rgb_dir_path, folder)), f"The file {folder} is not a folder."
        for subfolder in os.listdir(os.path.join(rgb_dir_path, folder)):
            if subfolder not in ['image_02', 'image_03']:
                shutil.rmtree(os.path.join(rgb_dir_path, folder, subfolder))


def rename_by_folder_name(dir_path: str, mode: str='rgb'):
    """
    A function used to rename the files in the directory according to the folder name.
    """
    if mode == 'rgb':
        subfolders = ['image_02/data', 'image_03/data']
    elif mode == 'velodyne_raw':
        subfolders = ['proj_depth/velodyne_raw/image_02', 'proj_depth/velodyne_raw/image_03']
    elif mode == 'groundtruth':
        subfolders = ['proj_depth/groundtruth/image_02', 'proj_depth/groundtruth/image_03']
    else:
        raise ValueError("The mode should be 'rgb', 'velodyne_raw' or 'groundtruth'.")
    for folder in tqdm(os.listdir(dir_path)):
        assert os.path.isdir(os.path.join(dir_path, folder)), f"The file {folder} is not a folder."
        for subfolder in subfolders:
            folder_path = os.path.join(dir_path, folder, subfolder)
            for filename in os.listdir(folder_path):
                if filename.endswith('.png') and filename.startswith('0'): # avoid some bugs
                    file_path = os.path.join(folder_path, filename)
                    if mode == 'rgb':
                        new_filename = folder + '_image_' + filename.split('.')[0] + '_' + subfolder.split('/')[0] + '.png'
                    elif mode == 'velodyne_raw':
                        new_filename = folder + '_velodyne_raw_' + filename.split('.')[0] + '_' + subfolder.split('/')[-1] + '.png'
                    elif mode == 'groundtruth':
                        new_filename = folder + '_groundtruth_depth_' + filename.split('.')[0] + '_' + subfolder.split('/')[-1] + '.png'
                    new_file_path = os.path.join(folder_path, new_filename)
                    os.rename(file_path, new_file_path)
        print(f"=> Renamed {folder} finished.\n")

def get_few_shot_list(cur_raw_dir_path, tar_raw_dir_path):
    """
    Get the list containing paths of few shot data.
    """
    for folder in os.listdir(cur_raw_dir_path): # train, val
        assert os.path.isdir(os.path.join(cur_raw_dir_path, folder)), f"The file {folder} is not a folder."
        sequence_dict = dict()
        for folder_name in os.listdir(os.path.join(cur_raw_dir_path, folder)):
            sequence = folder_name.split('_sync')[0]
            if sequence not in sequence_dict:
                sequence_dict[sequence] = []
        
        raw_file_generator = (file for file in os.listdir(os.path.join(tar_raw_dir_path, folder)) if 'image_02' in file)
        for raw_file in raw_file_generator:
            assert raw_file.endswith('.png'), f"The file {raw_file} is not a png file."
            sequence = raw_file.split('_sync')[0]
            assert sequence in sequence_dict, f"The date {sequence} is not in the date_dict."
            if len(sequence_dict[sequence]) < 100:
                sequence_dict[sequence].append(raw_file)
                sequence_dict[sequence].append(raw_file.replace('image_02', 'image_03'))
            else:
                continue

        with open(os.path.join(os.path.dirname(tar_raw_dir_path), f'{folder}.txt'), 'a') as f:
            for sequence in sequence_dict:
                for file_name in sequence_dict[sequence]:
                    f.write(f"{file_name}\n")
        print(f"=> The few shot data in {folder} has been saved into list.\n")

def main(args):
    """
    A function used to change the filename in current directory and move them to a new directory.
    """
    rgb_cur_dir = os.path.join(args.current_dir, 'raw_image')
    raw_cur_dir = os.path.join(args.current_dir, 'velodyne_raw')
    dense_cur_dir = os.path.join(args.current_dir, 'groundtruth')

    rgb_tar_dir = os.path.join(args.target_dir, 'raw_image')
    raw_tar_dir = os.path.join(args.target_dir, 'velodyne_raw')
    dense_tar_dir = os.path.join(args.target_dir, 'groundtruth')

    for tar_dir in [rgb_tar_dir, raw_tar_dir, dense_tar_dir]:
        for sub_tar_dir in ['train', 'val']:
            os.makedirs(os.path.join(tar_dir, sub_tar_dir), exist_ok=True)

    # do rename for rgb data, raw LiDAR data and dense depth data.
    for i in range(3):
        cur_dir = [rgb_cur_dir, raw_cur_dir, dense_cur_dir][i]
        print("Starting preprocessing the data in {}... \n".format(cur_dir))
        for sub_dir in ['train', 'val']:
            working_dir = os.path.join(cur_dir, sub_dir)
            rename_by_folder_name(working_dir, ['rgb', 'velodyne_raw', 'groundtruth'][i])

        # move the files to the target directory.
        tar_dir = [rgb_tar_dir, raw_tar_dir, dense_tar_dir][i]
        for root, dirs, files in os.walk(cur_dir):
            for filename in files:
                if filename.endswith('.png') and (not filename.startswith('0')):
                    file_path = os.path.join(root, filename)
                    if 'train' in root:
                        target_path = os.path.join(tar_dir, 'train', filename)
                    elif 'val' in root:
                        target_path = os.path.join(tar_dir, 'val', filename)
                    else:
                        raise ValueError("The file should be in train or val folder.")
                    shutil.move(file_path, target_path)
        print("=> Preprocessing the data in {} finised. \n".format(cur_dir))
    
    # get the few shot data list
    get_few_shot_list(raw_cur_dir, raw_tar_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_dir', type=str, default='/path/to/KITTI/dataset', help='The current directory of saved dataset.')
    parser.add_argument('--target_dir', type=str, default='/path/to/save/preprocessed/dataset', help='The target directory for saving dataset.')
    args = parser.parse_args()
    main(args)