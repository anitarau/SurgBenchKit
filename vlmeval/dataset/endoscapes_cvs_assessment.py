 # added by Anita Rau April 2025

from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd

class EndoscapesCVSAssessment(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(config['data_config']['data_dir'], split)
        self.ann_path = os.path.join(self.data_dir , 'all_metadata.csv')
        self.split = split
        self.few_shot = True if config['shots'] != 'zero' else False
        self.labels = self.load_labels()
        self.labels = self.labels[:10] #TODO delete this for debugging
        print(' ')

    def load_labels(self):
        labels = []
        fps_rate = 1
        df = pd.read_csv(self.ann_path)
        df = df.dropna(subset=['cvs_annotator_1'])  # drop rows with no annotation. Dataset interpolates between annotations, which can be used for training but not testing

        # list videos in split
        split_videos = [int(float(vid)) for vid in open(os.path.join(self.data_dir, self.split + '_vids.txt')).read().splitlines()]

        
        label_columns = ['C1', 'C2', 'C3']
        for idx, row in df.iterrows():
            video_name = str(row['vid'])
            if int(video_name) not in split_videos:
                continue
            if self.few_shot:
                fps_rate = 2
            if row['frame'] % fps_rate != 0:
                continue    
            frame_name = str(row['frame'])
            frame_label = row[label_columns].astype(float).values.round(0) # Round CVS values at test time to get majority vote of three raters. output: array([0., 0., 0.])
            frame_path = os.path.join(self.image_dir, video_name + '_' + frame_name + '.jpg')
            labels.append((frame_path, frame_label))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_name, frame_label = self.labels[idx]
        frame = {'path': frame_name}
        return (
            frame,
            frame_label  # [C1, C2, C3]
        )
