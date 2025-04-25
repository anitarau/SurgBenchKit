from torch.utils.data import Dataset
import os

class AutoLaparoManeuverClassification(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']

        self.data_dir = config['data_config']['data_dir']
        self.split = split

        self.label_map = {'0': 'Static', '1': 'Up', '2': 'Down', '3': 'Left', '4': 'Right', '5': 'Zoom-in', '6': 'Zoom-out'}
        self.start_clip = 228
        self.labels = self.load_data()
    
    def load_data(self):
        labels = []

        file_path = f'{self.data_dir}/laparoscope_motion_label.txt'
        with open(file_path, 'r') as f:
            label = [line.strip().split('\t') for line in f.readlines()]
            label = label[1:]
            for clip_name, maneuver_num, _ in label:
                
                if int(clip_name) < self.start_clip:
                    continue

                video_path = os.path.join(self.data_dir, 'clips_0_to_T', clip_name + '.mp4')
                labels.append((video_path, self.label_map[maneuver_num]))

        return labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        video_path, video_label = self.labels[idx]

        video = {'path': video_path}
        return (
            video,
            video_label
        )