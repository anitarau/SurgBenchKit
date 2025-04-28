# added by Josiah Aklilu April 2025

import cv2
import decord
import json
import numpy as np
import os
import pandas as pd
import pickle
import torch

from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class ErrorDetection(Dataset):
    """
    Error Detection dataset. The primary errors we focus on for 
    localization include Bleeding and Bile Spillage.
    """
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.dataset_name = config['data_config']['dataset_name']
        self.ann_dir = config['data_config']['ann_dir']
        self.split = split

        bleeding_file = os.path.join(self.ann_dir, 'bleeding', '2024-11-02', 'id2chosen_annotation.csv')
        bleeding_df = pd.read_csv(bleeding_file)
        bile_file = os.path.join(self.ann_dir, 'bile_spillage', '2024-11-04', 'id2chosen_annotation.csv')
        bile_df = pd.read_csv(bile_file)
        self.ann_df = pd.concat([bleeding_df, bile_df], ignore_index=True)

        errors_df = self.load_errors()

        np.random.seed(0)
        split_per = 0.50 if 'heichole' in self.dataset_name.lower() else 0.10

        test_indices = np.random.choice(len(errors_df), int(split_per*len(errors_df)), replace=False)  # test set
        val_indices = [i for i in range(len(errors_df)) if i not in test_indices]
        errors_df['split'] = ['test' if i in test_indices else 'val' for i in range(len(errors_df))]
        print(f' {len(errors_df)} error clips with different error types.')

        self.labels = [(row.clip_path, (row.label_start, row.label_end, row.label)) 
                       for row in errors_df[errors_df['split'] == split].itertuples()]
        print(f'The {split} set has {len(self.labels)} error clips.')
        print(list(errors_df[errors_df['split'] == split]['label'].value_counts()))

    def load_errors(self):
        df = self.ann_df[self.ann_df['dataset'] == self.dataset_name]   # filter by dataset name
        print(f'{self.dataset_name} has {len(df)} total error annotations for full videos, so we create', end='')

        if os.path.exists(os.path.join(self.data_dir, self.dataset_name, 'error_detection.csv')):
            error_df = pd.read_csv(os.path.join(self.data_dir, self.dataset_name, 'error_detection.csv'))
            return error_df

        np.random.seed(0)
        items = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            label_df = pd.read_csv(row['clip_csv_error'])

            for _, label_row in tqdm(label_df.iterrows(), total=len(label_df)):
                clip_path, (label_start, label_end) = self._save_or_load_error_clip(
                    clip_id=label_row['clip_id'],
                    full_video_path=label_row['video_path'],
                    duration=label_row['duration'],
                    fps=label_row['fps']
                )
                items.append((clip_path, label_row['label'], label_start, label_end))

        error_df = pd.DataFrame(items, columns=['clip_path', 'label', 'label_start', 'label_end'])
        error_df.to_csv(os.path.join(self.data_dir, self.dataset_name, 'error_detection.csv'), index=False)

        return error_df

    def _save_or_load_error_clip(self, clip_id, full_video_path, duration, fps=10):
        times = duration.split('-')
        start_time, end_time = int(times[0]), int(times[1])

        os.makedirs(os.path.join(self.data_dir, self.dataset_name), exist_ok=True)
        clip_paths = list(Path(self.data_dir).joinpath(self.dataset_name).glob(f'{clip_id}*.mp4'))

        video = decord.VideoReader(full_video_path)
        h, w = video[0].shape[:2]
        
        # Calculate random start time that includes at least part of the error duration
        clip_duration = 180 * fps  # 180 seconds at given fps
        error_duration = end_time - start_time
        
        # Ensure start_frame allows for full 180s clip
        latest_possible_start = max(0, len(video) - clip_duration)
        # Ensure we capture at least part of the error
        earliest_possible_start = max(0, start_time - clip_duration)
        latest_start_with_error = start_time
        
        # Random start frame that ensures we get some of the error
        if min(earliest_possible_start, latest_possible_start) == min(latest_start_with_error, latest_possible_start):
            clip_start = 0
        else:
            clip_start = np.random.randint(
                min(earliest_possible_start, latest_possible_start),
                min(latest_start_with_error, latest_possible_start)
            )
        clip_end = min(clip_start + clip_duration, len(video))

        # Adjust label start/end to be relative to new clip
        label_start = max(clip_start, start_time) - clip_start
        label_end = min(clip_end, end_time) - clip_start

        clip_path = os.path.join(self.data_dir, self.dataset_name, f'{clip_id}_cs{clip_start}_ce{clip_end}.mp4')
        if not os.path.exists(clip_path):
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            frames = list(range(clip_start, clip_end))
            clip = video.get_batch(frames).asnumpy()
            # Write frames individually
            for frame in clip:
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()

        return clip_path, (label_start, label_end)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path, (label_start, label_end, error_type) = self.labels[idx]
        return (
            {'video': None, 'path': video_path},
            {'error_type': error_type, 'start': label_start, 'end': label_end} 
        )