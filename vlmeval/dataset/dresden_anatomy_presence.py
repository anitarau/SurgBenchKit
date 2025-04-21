 # added by Anita Rau April 2025

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class DresdenAnatomyPresence(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        if split == 'train':
            folders = [1, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 19, 22, 23, 24, 25, 27, 28, 29, 30, 31]  # official split: https://www.medrxiv.org/content/10.1101/2022.11.11.22282215v5.full.pdf
        elif split == 'val':
            folders = [3, 21, 26]  # official split
        else:
            folders = [2, 7, 11, 13, 18, 20, 32]  # official split (14 should be part of test set; however, the annotations are missing from official dataset)

        self.image_dirs = [
            os.path.join(config['data_config']['data_dir'], anatomy.replace(' ','_'), str(folder).zfill(2))
            for anatomy in config['label_names']
            for folder in folders
            if os.path.exists(os.path.join(config['data_config']['data_dir'], anatomy.replace(' ','_'), str(folder).zfill(2)))
        ]       
        self.data_dir = config['data_config']['data_dir']
        self.anatomy_map = {v: idx for idx, v in enumerate(config['label_names'])}
        self.labels = self.load_labels()
        self.labels = self.labels[:10]  # TODO delete this, this is for debugging
    
    def load_labels(self):
        labels = []
        for folder in self.image_dirs:
            labels_path = os.path.join(self.data_dir, folder, 'weak_labels.csv')
            with open(labels_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip(',\n').split(',')
                    frame_name = os.path.join(self.data_dir, folder,line[0].replace('images0', 'image'))  # naming mismatch
                    anatomies = line[1:]
                    labels.append((frame_name, np.array(anatomies, dtype=np.int32)))
        return labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        frame_name, frame_label = self.labels[idx]
        frame = {'path': frame_name}
        return (
            frame,
            frame_label
        )
    