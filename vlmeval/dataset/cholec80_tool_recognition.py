from torch.utils.data import Dataset
import os
import numpy as np


class Cholec80ToolRecognition(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.image_dir = os.path.join(config['data_config']['data_dir'], 'frames_25fps', split)
        self.data_dir = config['data_config']['data_dir']
        self.map = {tool: idx for idx, tool in enumerate(config['label_names'])}

        self.split = split
        self.few_shot = True if config['shots'] != 'zero' else False
        self.labels = self.load_labels()
    
    def load_labels(self):
        labels = []
        fps_rate = 25
        if self.split == 'test':
            fps_rate = fps_rate * 5
            if self.few_shot:
                fps_rate = fps_rate * 15
        for video_name in os.listdir(self.image_dir):
            video_labels_path = os.path.join(self.data_dir, 'tool_annotations', f'{video_name}-tool.txt')
            with open(video_labels_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split()
                    if line[0] == 'Frame':
                        tools = line[1:]
                    else:
                        if int(line[0]) % fps_rate != 0:
                            continue  # only sample 1/5 frame per sec
                        frame_label = [None] * len(self.map)
                        for i, tool in enumerate(tools):
                            frame_label[self.map[tool]] = int(line[i+1])
                        frame_name = f'{line[0]}.jpg'
                        frame_label = np.array(frame_label, dtype=np.float32)
                        frame_path = os.path.join(self.image_dir, video_name, frame_name)
                        labels.append((frame_path, frame_label))
        labels = labels[:10]  # TODO remove this line - for debugging
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
    