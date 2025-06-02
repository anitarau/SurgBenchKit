 # added by Josiah Aklilu April 2025

import cv2
import decord
import json
import numpy as np
import os
import pandas as pd
import pickle
import time
import torch

from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset


class ErrorRecognition(Dataset):
    """
    The types of errors include: [Bleeding, Bile Spillage, Thermal Injury, Perforation]
    """
    def __init__(self, config, split):
        assert split in ['train', 'val', 'test']
        self.data_dir = config['data_config']['data_dir']
        self.dataset_name = config['data_config']['dataset_name']
        self.ann_dir = config['data_config']['ann_dir']
        self.split = split

        bleeding_df = pd.read_csv(os.path.join(
            self.ann_dir, 
            'id2chosen_csv_errors_v2', 
            'bleeding', 
            '2024-11-02', 
            'id2chosen_annotation.csv'
        ))
        bile_df = pd.read_csv(
            os.path.join(self.ann_dir, 
            'id2chosen_csv_errors_v2', 
            'bile_spillage', 
            '2024-11-04', 
            'id2chosen_annotation.csv'
        ))
        thermal_df = pd.read_csv(
            os.path.join(self.ann_dir, 
            'id2chosen_csv', 
            '2024-07-22_2024-04-23', 
            'id2chosen_annotation.csv'
        ))
        perforation_df = pd.read_csv(
            os.path.join(self.ann_dir, 
            'merged_cleaned_perforation_id2chosen_csv', 
            '2024-07-22_2024-04-23', 
            'id2chosen_annotation.csv'
        ))
        self.ann_df = pd.concat([bleeding_df, bile_df, thermal_df, perforation_df], ignore_index=True)

        errors_df = self.load_errors()

        np.random.seed(0)
        split_per = 0.50 if 'heichole' in self.dataset_name.lower() else 0.20

        test_indices = np.random.choice(len(errors_df), int(split_per*len(errors_df)), replace=False)
        val_indices = [i for i in range(len(errors_df)) if i not in test_indices]
        errors_df['split'] = ['test' if i in test_indices else 'val' for i in range(len(errors_df))]
        print(f' {len(errors_df)} error clips with different error types.')

        self.labels = list(errors_df[
            errors_df['split'] == split
        ].drop('split', axis=1).itertuples(index=False, name=None))
        print(f'The {split} set has {len(self.labels)} error clips.')
        print(list(errors_df[errors_df['split'] == split]['label'].value_counts()))

    def load_errors(self):
        df = self.ann_df[self.ann_df['dataset'] == self.dataset_name]   # filter by dataset name
        print('-'*60)
        print(f'{self.dataset_name} has {len(df)} total error annotation files, resulting in', end='')

        label_path = os.path.join(self.data_dir, self.dataset_name, 'error_classification.csv')
        if os.path.exists(label_path):
            error_df = pd.read_csv(label_path)
            return error_df

        np.random.seed(0)
        items = []
        grouped_df = df.groupby('video_id').first().reset_index()
        for i, row in tqdm(grouped_df.iterrows(), total=len(grouped_df)):
            try:
                label_df = pd.read_csv(row['clip_csv_error'])
            except:
                continue
        
            for _, label_row in tqdm(label_df.iterrows(), total=len(label_df)):
                raw_label = label_row['label']
                if raw_label == 'Bleeding' or raw_label == 'Bile Spillage':
                    label = raw_label
                elif 'InjuryToNontargetStructureOrTissue_Burn' in raw_label:
                    label = 'Thermal Injury'
                elif 'Perforation' in raw_label:
                    label = 'Perforation'
                else:
                    continue
                try:
                    clip_path = self._save_or_load_error_clip(
                        clip_id=label_row['clip_id'],
                        full_video_path=label_row['video_path'],
                        duration=label_row['duration'],
                        fps=label_row['fps']
                    )
                except:
                    continue
                items.append((clip_path, label))

        error_df = pd.DataFrame(items, columns=['clip_path', 'label'])
        error_df.to_csv(label_path, index=False)

        return error_df

    def _save_or_load_error_clip(self, clip_id, full_video_path, duration, fps=10):
        times = duration.split('-')
        start_time, end_time = int(times[0]), int(times[1])

        os.makedirs(os.path.join(self.data_dir, self.dataset_name), exist_ok=True)
        clip_paths = list(Path(self.data_dir).joinpath(self.dataset_name).glob(f'{clip_id}*.mp4'))

        video = decord.VideoReader(full_video_path)
        original_fps = int(video.get_avg_fps())
        stride = original_fps // fps
        h, w = video[0].shape[:2]
        
        # Calculate random start time that includes at least part of the error duration
        clip_duration = 30 * original_fps  # 30 seconds at given fps
        error_duration = end_time - start_time
        
        # If error duration is >= 30 sec, sample a 30 sec window within it
        if end_time - start_time >= clip_duration:
            # Calculate latest possible start within error duration
            latest_start = end_time - clip_duration
            # Sample start frame from within error duration
            clip_start = np.random.randint(start_time, latest_start + 1)
            clip_end = clip_start + clip_duration

        # If error duration < 30 sec, center the error and pad to 30 sec
        else:
            # Calculate how much padding needed on each side
            total_padding = clip_duration - (end_time - start_time)
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
            
            # Apply padding but respect video boundaries
            clip_start = max(0, start_time - left_padding)
            clip_end = min(len(video), end_time + right_padding)
            
            # If we hit a boundary, shift the window to get full duration
            if clip_start == 0:
                clip_end = min(len(video), clip_duration)
            elif clip_end == len(video):
                clip_start = max(0, len(video) - clip_duration)

        clip_path = os.path.join(self.data_dir, self.dataset_name, f'{clip_id}_cs{clip_start}_ce{clip_end}.mp4')
        if not os.path.exists(clip_path):
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            frames = list(range(clip_start, clip_end, stride))
            assert len(frames) == 30 * fps
            clip = video.get_batch(frames).asnumpy()
            # Write frames individually
            for frame in clip:
                # Convert from RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()

        return clip_path
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        video_path, video_label = self.labels[idx]
        video = {'path': video_path}
        return (
            video,
            video_label
        )