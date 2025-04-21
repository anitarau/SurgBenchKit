from torch.utils.data import Dataset
import os
import numpy as np


class Cholect45Triplet(Dataset):
    def __init__(self, config, split, transform=None, use_api=False):
        assert split in ['train', 'val', 'test']
        if split == 'val':  # official split: https://arxiv.org/pdf/2204.05235
            self.folders = ['VID08', 'VID29', 'VID50', 'VID78']
        elif split == 'test':
            self.folders = ['VID06', 'VID51', 'VID10', 'VID73', 'VID14', 'VID74', 'VID32', 'VID80', 'VID42']
        else:
            self.folders = ['VID01', 'VID02', 'VID04', 'VID05', 'VID13', 'VID15', 'VID18', 'VID22', 'VID23', 'VID25', 'VID26', 'VID27', 'VID31', 'VID35', 'VID36', 'VID40', 'VID43', 'VID47', 'VID48', 'VID49', 'VID52', 'VID56', 'VID57', 'VID60', 'VID62', 'VID65', 'VID66', 'VID68', 'VID70', 'VID75', 'VID79']
        data_dir = config['data_config']['data_dir']
        self.image_dir = os.path.join(data_dir, 'rgb')
        self.label_dir = os.path.join(data_dir, 'triplet')
        self.data_dir = data_dir
        self.use_api = use_api
        self.split = split
        with open(os.path.join(data_dir, 'dict/triplet.txt'), 'r') as f:
            lines = f.readlines()
            self.triplet_map = {line.split(':')[0]: line.strip().split(':')[1:] for line in lines}
        with open(os.path.join(data_dir, 'dict/maps.txt'), 'r') as f:
            lines = f.readlines()
            self.maps = {int(line.split(',')[0]): [int(e) for e in line.strip().split(',')[1:4]] for line in lines[1:]} # first line is header
        with open(os.path.join(data_dir, 'dict/target.txt'), 'r') as f:
            lines = f.readlines()
            self.target_map = {int(line.split(':')[0]): line.strip().split(':')[1] for line in lines}
        with open(os.path.join(data_dir, 'dict/verb.txt'), 'r') as f:
            lines = f.readlines()
            self.verb_map = {int(line.split(':')[0]): line.strip().split(':')[1] for line in lines}
        with open(os.path.join(data_dir, 'dict/instrument.txt'), 'r') as f:
            lines = f.readlines()
            self.instrument_map = {int(line.split(':')[0]): line.strip().split(':')[1] for line in lines}

        self.transform = transform
        self.labels = self.load_labels()

    def load_labels(self):
        fps_rate = 5
        labels = []
        for video_name in self.folders:
            if video_name[:3] != 'VID':
                continue
            if video_name.split('.')[0] == 'VID12':
                continue
            video_labels_path = os.path.join(self.label_dir, video_name + '.txt')
            with open(video_labels_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(',')
                    frame_name = int(line[0])
                    if frame_name % fps_rate != 0:
                        continue
                    frame_label = [int(e) for e in line[1:]]
                    frame_label = np.array(frame_label, dtype=np.float32)
                    frame_path = os.path.join(self.image_dir, video_name.split('.')[0], f'{frame_name:06d}.png')
                    labels.append((frame_path, frame_label))
        labels = labels[:10]  # TODO remove this line - for debugging

        return labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        frame_path, frame_label = self.labels[idx]

        frame = {'path': frame_path}
        return (
            frame,
            frame_label
        )
