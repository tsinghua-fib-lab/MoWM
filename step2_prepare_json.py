import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple
from tqdm import tqdm
import torch
import random
import imageio
from decord import VideoReader, cpu
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
# from finetune.constants import LOG_LEVEL, LOG_NAME
import numpy as np

def load_and_process_ann_file(data_root, ann_file, sequence_interval=1, start_interval=4, dataset_name='xhand_1024_v2', sequence_length=8):
    samples = []
    try:
        with open(f'{data_root}/{ann_file}', "r") as f:
            ann = json.load(f)
    except:
        print(f'skip {ann_file}')
        return samples
    try:
        n_frames = len(ann['action'])
    except:
        n_frames = ann['video_length']

    # create multiple samples for robot data      
    # sequence_interval = 1
    # start_interval = 4
    # record idx for each clip
    base_idx = np.arange(0,sequence_length)*sequence_interval
    max_idx = np.ones_like(base_idx)*(n_frames-1)
    for start_frame in range(0,n_frames,start_interval):
        idx = base_idx + start_frame
        idx = np.minimum(idx,max_idx)
        idx = idx.tolist()
        if len(idx) == sequence_length:
            sample = dict()
            sample['dataset_name'] = dataset_name
            sample['ann_file'] = ann_file
            sample['episode_id'] = ann['episode_id']
            sample['frame_ids'] = idx
            sample['states'] = np.array(ann['states'])[idx[0]:idx[0]+1]
            sample['actions'] = np.array(ann['actions'])[idx]
            samples.append(sample)

    return samples

def init_anns(dataset_root, data_dir):
    final_path = f'{dataset_root}/{data_dir}'
    ann_files = [os.path.join(data_dir, f) for f in os.listdir(final_path) if f.endswith('.json')]
    # data_dir = f'{dataset_root}/{data_dir}'
    # ann_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    return ann_files

def init_sequences(data_root, ann_files, sequence_interval, start_interval, dataset_name,sequence_length):
    samples = []
    with ThreadPoolExecutor(32) as executor:
        future_to_ann_file = {executor.submit(load_and_process_ann_file, data_root, ann_file, sequence_interval, start_interval, dataset_name, sequence_length): ann_file for ann_file in ann_files}
        for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            samples.extend(future.result())
    return samples
# start __main__

if __name__ == "__main__":
    dataset_names = 'xbot_0407+xbot_0409+xbot_0410'
    sequence_length = 8
    is_50hz = []
    trajs_each_demo = 1

    dataset_names = dataset_names.split('+')
    skip=1

    for data_type in ['val', 'train']:
        
        samples_all = []
        ann_files_all = []

        for dataset_name in dataset_names:
            data_save_path = '/localssd/gyj/data0924/opensource_robotdata'
            data_dir = f'annotation/{data_type}'
            data_root = f'{data_save_path}/{dataset_name}'
            if 'xhand_1125' in dataset_name:
                sequence_interval = int(skip*5)
                start_interval = 3
            else:
                sequence_interval = skip
                start_interval = 1

            ann_files = init_anns(data_root, data_dir)
            if dataset_name in is_50hz:
                ann_files = [f for f in ann_files if int(f.split('/')[-1].split('.')[0])%trajs_each_demo == 0]
            ann_files_all.extend(ann_files)
            # print(ann_files)
            samples = init_sequences(data_root, ann_files,sequence_interval, start_interval, dataset_name, sequence_length)
            print(f'{dataset_name} {len(samples)} samples')
            samples_all.extend(samples)
        
        # calculate the 1% and 99% perventile of the action and state for normalization
        print("########################### state ###########################")
        print(np.array(samples_all[0]['actions']).shape)
        print(np.array(samples_all[0]['states']).shape)
        state_all = [samples['states'] for samples in samples_all]
        state_all = np.array(state_all)
        print(state_all.shape)
        state_all = state_all.reshape(-1, state_all.shape[-1])
        # caculate the 1% and 99% of the action and state
        state_01 = np.percentile(state_all, 1, axis=0)
        state_99 = np.percentile(state_all, 99, axis=0)
        print('state_01:', state_01)
        print('state_99:', state_99)

        print("########################### action ###########################")
        action_all = [samples['actions']-samples['states'] for samples in samples_all]
        action_all = np.array(action_all)
        print(action_all.shape)
        action_all = action_all.reshape(-1, action_all.shape[-1])
        # caculate the 1% and 99% of the action and state
        action_01 = np.percentile(action_all, 1, axis=0)
        action_99 = np.percentile(action_all, 99, axis=0)
        print('action_01:', action_01)
        print('action_99:', action_99)

        # remove state and action from samples
        for samples in samples_all:
            del samples['states']
            del samples['actions']

        import random
        random.shuffle(samples_all)
        print('step_num',data_type,len(samples_all))
        print('traj_num',data_type, len(ann_files_all))

        date = '0413_xbot'
        # write to json file
        os.makedirs(f'{data_save_path}/annotation_all/{date}_interval{skip}/', exist_ok=True)
        with open(f'{data_save_path}/annotation_all/{date}_interval{skip}/{data_type}_all.json', 'w') as f:
            json.dump(samples_all, f, indent=4)
        
        stat = {
            'state_01': state_01.tolist(),
            'state_99': state_99.tolist(),
            'action_01': action_01.tolist(),
            'action_99': action_99.tolist()
        }
        with open(f'{data_save_path}/annotation_all/{date}_interval{skip}/{data_type}data.json', 'w') as f:
            json.dump(stat, f)