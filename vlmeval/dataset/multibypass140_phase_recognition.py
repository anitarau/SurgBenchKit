from torch.utils.data import Dataset
import os
import pickle
import logging

def read_pkl_data(pkl_path, img_path):
    # From original code
    logging.info('reading pickle file: '+ pkl_path)
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)
        fp.close()
    root_dir = img_path
    if not os.path.exists(root_dir):
        root_dir = root_dir.replace('train', '').replace('val', '').replace('test', '')
    imgs, phases, steps = [], [], []
    for vid_name in sorted(data.keys()):
        paths = [
                os.path.join(root_dir, vid_name, f"{item['Frame_id']}.jpg")
            for item in data[vid_name]
        ]
        imgs.append(paths)
        phases.append([item['Phase_gt'] for item in data[vid_name]])
        steps.append([item['Step_gt'] for item in data[vid_name]])
    
    return imgs, phases, steps

class MultiBypass140PhaseRecognition(Dataset):
    def __init__(self, config, split, transform=None, use_api=False):
        assert split in ['train', 'val', 'test']
        self.image_dir = os.path.join(config['data_config']['data_dir'], 'frames_25fps', split)
        self.data_dir = config['data_config']['data_dir']
        self.split = split
        self.transform = transform

        self.phase_map = {phase: idx for idx, phase in enumerate(config['label_names'])}
        self.labels = self.load_labels()
        self.labels = self.labels[:10]  # TODO remove this line - for debugging

    def load_labels(self):
         ## Read val pickle files
        id_split = 0 
        dataset = 'labels/bern/labels_by70_splits/'
        im_folder = 'datasets/MultiBypass140/BernBypass70'
        dataset_n = dataset.replace('bern', 'strasbourg')
        im_folder_n = im_folder.replace('BernBypass70', 'StrasBypass70')
        labels_file = os.path.join(self.data_dir, dataset, 'labels', self.split, f'1fps_{id_split}.pickle')
        images = os.path.join(self.data_dir, im_folder, 'frames')
        videos, phase_labels, step_labels = read_pkl_data(
            labels_file, images
        )
        if dataset_n != []:
            labels_file = os.path.join(self.data_dir, dataset_n, 'labels', self.split, f'1fps_{id_split}.pickle')
            images = os.path.join(self.data_dir, im_folder_n, 'frames')
            videos_n, phase_labels_n, step_labels_n = read_pkl_data(
                labels_file, images
            )
            videos += videos_n
            phase_labels += phase_labels_n
            step_labels += step_labels_n
        
        videos = [item for sublist in videos for item in sublist]
        phase_labels = [item for sublist in phase_labels for item in sublist]
        step_labels = [item for sublist in step_labels for item in sublist]
        labels = []
        # read pickle
        for i in range(len(videos)):
            if int(videos[i].split('/')[-1].strip('.jpg').split('_')[1]) % 20 == 0:  # data labeled at 1 fps, but only use every 20th frame
                labels.append((videos[i], phase_labels[i]))
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