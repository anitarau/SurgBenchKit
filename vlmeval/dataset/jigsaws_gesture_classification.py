from torch.utils.data import Dataset
import os

class JIGSAWSGestureClassification(Dataset):
    def __init__(self, config, split, transform=None, use_api=False):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.category = config['data_config']['category'] # knot tying, needle passing, suturing
        assert self.category in ['Knot_Tying', 'Needle_Passing', 'Suturing']
        self.split = split
        self.labels = self.load_data()

    def load_data(self):
        labels = []

        label_dir = f'{self.data_dir}{self.category}/transcriptions/'
        video_dir = f'{self.data_dir}{self.category}/video/'
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)

            with open(file_path, 'r') as f:
                label = [line.strip().split(' ') for line in f.readlines()]

                # need processed videos according to start and end frame
                for start_frame, end_frame, class_label in label:
                    video_path = os.path.join(self.video_dir, f'{filename.replace(self.category + "_", "").replace(".txt", "")}_{start_frame}_{end_frame}_{class_label}.mp4')
                    labels.append((video_path, class_label))

        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path, video_label = self.labels[idx]
        return (
            video_path,
            video_label
        )