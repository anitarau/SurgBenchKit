from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd


class AVOSActionRecognition(Dataset):
    def __init__(self, config, split, transform=None, use_api=False):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            split = 'test'
        self.data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(self.data_dir, 'images')

        self.transform = transform
        self.split = split
        self.map = {action: idx for idx, action in enumerate(config['label_names'])}
        self.labels = self.load_labels()
        self.labels = self.labels[::10]  # TODO remove this line - for debugging

    def load_labels(self):
        labels = []

        video_names_path = os.path.join(self.data_dir, f'{self.split}.csv')
        video_names = []
        with open(video_names_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                video_names.append(line.strip())

        annotations_path = os.path.join(self.data_dir, 'open_surgery_temporal_annotations_Jan16.csv')
        annotations_df = pd.read_csv(annotations_path)
        train_split = [line.strip() for line in open(os.path.join(self.data_dir, 'train.csv'), 'r').readlines()[1:]]
        test_split = [line.strip() for line in open(os.path.join(self.data_dir, 'test.csv'), 'r').readlines()[1:]]
        not_train_or_test = 0
        for image_name in os.listdir(self.image_dir):
            video_name = ''.join(image_name.split('-')[:-1])
            if video_name not in train_split and video_name not in test_split:
                not_train_or_test += 1
            if self.split == 'train' and video_name not in train_split: continue
            if self.split == 'test' and video_name not in test_split: continue
            frame_number = int(image_name.split('-')[-1].replace('.jpg', ''))
            ann = annotations_df[(annotations_df['video_id'] == video_name) & (annotations_df['start_frame'] <= frame_number) & (annotations_df['end_frame'] >= frame_number)]
            if len(ann) == 0: continue
            else:
                ann = ann.iloc[0]
                action_labels = ann[['label']]
                if len(set(action_labels)) == 3: continue

                action = action_labels.mode()[0]
                if action == 'none' or action == 'abstain': continue
                frame_path = os.path.join(self.image_dir, image_name)
                labels.append((frame_path, action))

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