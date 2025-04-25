from torch.utils.data import Dataset
import os

class JIGSAWSSkillAssessment(Dataset):
    def __init__(self, config, split, transform=None, use_api=False):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.category = config['data_config']['category'] # knot tying, needle passing, suturing
        self.score = config['data_config']['score']
        assert self.category in ['Knot_Tying', 'Needle_Passing', 'Suturing']
        self.split = split
        self.labels = self.load_data()

    def load_data(self):
        labels = []

        label_path = f'{self.data_dir}{self.category}/meta_file_{self.category}.txt'
        video_dir = f'{self.data_dir}{self.category}/video/'
        with open (label_path, 'r') as f:
            label = [line.strip().split('\t') for line in f.readlines()]
            for example in label:
                if len(example) == 1: continue
                cols = ['video_name', 'experience', 'grs', 'respect_for_tissue', 'suture_needle_handling', 'time_and_motion', 'flow_of_operation', 'overall_performance', 'quality_of_final_product']
                example = [s for s in example if s != '']
                assert len(example) == len(cols)
                video_name = example[cols.index('video_name')]
                video_name = f'{video_name.replace(self.category + "_", "")}.mp4'
                _, video_name = os.path.split(video_name)
                video_path = os.path.join(self.video_dir, video_name)
                score = example[cols.index(self.score)]
                
                labels.append((video_path, score))
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path, video_label = self.labels[idx]
        return (
            video_path,
            video_label
        )