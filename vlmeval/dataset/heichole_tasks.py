# added by Anita Rau April 2025


from torch.utils.data import Dataset
import os

class HeiCholeDataloader(Dataset):
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        # no access to official test data, so defined our own split on train data:
        # random_list = np.random.permutation(24)
        # print(', '.join(map(str, random_list[:14] + 1)))
        # print(', '.join(map(str, random_list[14:19] + 1)))
        # print(', '.join(map(str, random_list[19:24] + 1)))
        if split == 'train':
            folders = [12, 11, 23, 15, 21, 2, 14, 24, 17, 9, 7, 18, 5, 3]
        elif split == 'val':
            folders = [6, 19, 10, 8, 20]
        else:
            folders = [4, 1, 22, 16, 13]
        self.image_dir = os.path.join(config['data_config']['data_dir'], 'extracted_frames')
        self.data_dir = config['data_config']['data_dir']

        self.image_dirs = [
            os.path.join(self.image_dir, 'Hei-Chole' + str(folder))
            for folder in folders
            if os.path.exists(os.path.join(self.image_dir, 'Hei-Chole' + str(folder)))
        ]    

        self.split = split

        self.map = {phase: idx for idx, phase in enumerate(config['label_names'])}
        if 'phase' in config['name']:
            self.task_name = 'Phase'
        elif 'tool' in config['name']:
            self.task_name = 'Instrument'
        elif 'action' in config['name']:
            self.task_name = 'Action'
        else:
            raise NotImplementedError
        
        self.few_shot = True if config['shots'] != 'zero' else False
        self.labels = self.load_labels()

    def load_labels(self):
        labels = []
        fps_rate = 25
        if self.few_shot:
            fps_rate = fps_rate * 15
        for video_name in self.image_dirs:
            video_name = video_name.split("/")[-1]
            video_labels_path = os.path.join(self.data_dir, 'Annotations', self.task_name, f'{video_name}_Annotation_' + self.task_name + '.csv')
            with open(video_labels_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = list(map(int, line.strip().split(',')))
                    if line[0] % fps_rate == 0:  # The dataset has videos with 25 or 50 fps. We select 1 or 2 fps. 
                        frame_label = line[1:]
                        frame_name = f'frame_{str(line[0]).zfill(5)}.png'
                        frame_path = os.path.join(self.image_dir, video_name, frame_name)
                        labels.append((frame_path, frame_label))

        labels = labels[:10] # TODO remove this line - for debugging
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