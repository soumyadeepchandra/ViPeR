
import os
import random
import time

import cv2
import h5py
import numpy as np
import pandas as pd
import statistics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange

# Using RESNET18 convert [f,c,h,w] -> [f,2048]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class new_Resnet18(nn.Module):
    def __init__(self, output_layer = None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(512, 11)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

class npyLoader(Dataset):
    def __init__(self, video_path, anno_path, feature_path):
        
        self.videos_dir = video_path
        self.ann_dir = anno_path
        assert os.path.isdir(self.ann_dir)
        self.ann_paths = [
            os.path.join(self.ann_dir, item) for item in os.listdir(self.ann_dir)
        ]

        videos_names = os.listdir(self.videos_dir)
        self.video_paths = [
            os.path.join(self.videos_dir, item) for item in videos_names
        ]

        self.res_model = new_Resnet18(output_layer = 'layer4')
        self.res_model.eval().to(device)
        res_model_dir = './models_feature/epoch-60-0.5001785714285715.model'
        print("Resnet Loaded")
        self.res_model.load_state_dict(torch.load(res_model_dir))

    def __getitem__(self, idx):
        
        final_frames = list()
        seq_lens = list()
        steps = list()
        annotations = list()

        for video in self.video_paths:
            # print(video)
            video_frames_count = self.get_num_frames(video)
            video_fps = self.get_video_fps(video)
            npy_file_name = self._extract_frames(video)
            
        return 0

    
    def get_num_frames(self, video):
        """
        This method is used to calculate the number of frames in a video.
        """
        cap = cv2.VideoCapture(video)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return num_frames

    def get_video_fps(self, video):
        """
        This method is used to calculate the fps of a video.
        """
        cap = cv2.VideoCapture(video)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def _extract_frames(self, video_path, anno_path, feature_path, gt_path):
        """
        This method is to extract npy frames from the video
        """

        print(video_path)
        video_name = video_path.split('/')[-1].split('.')[0]
        print(video_name)
        csv_file = os.path.join(anno_path, f'{video_name}.csv')
        annotation_data = pd.read_csv(open(csv_file, 'r'), header=None)

        gt_file = os.path.join(gt_path, f'{video_name}.txt')
        
        all_ranges = []
        current_frame = 0
        count = 0
        with open(gt_file, 'w') as txt_file:
            
            for step in annotation_data.values:
                start = step[0]
                end = step[1]
                diff = step[1] - step[0]
                row = step[2].split(' ')
                task = ' '.join(row[1:])
                all_ranges.append((start,end))
                for i in range(diff):
                    txt_file.write(''.join(task) + '\n')
                    count += 1

        
        print(all_ranges)
        print(count)
    
        npy_file_path = os.path.join(feature_path, f'{video_name}.npy')

        videocap = cv2.VideoCapture(video_path)
        frames = list()
        frames_temp = list()
        frames_converted = torch.zeros(count, 2048, dtype=torch.float)
        desired_shorter_side = 384
        frame_count = 0
        idx = 0
        while True:
            success, frame = videocap.read()
            if not success:
                break
        
            for eachrange in all_ranges:
                if current_frame >= eachrange[0] and current_frame < eachrange[1]:
                    original_height, original_width, _ = frame.shape
                    if original_height < original_width:
                        # Height is the shorter side
                        new_height = desired_shorter_side
                        new_width = np.round(
                            original_width*(desired_shorter_side/original_height)
                        ).astype(np.int64)
                    elif original_height > original_width:
                        # Width is the shorter side
                        new_width = desired_shorter_side
                        new_height = np.round(
                            original_height*(desired_shorter_side/original_width)
                        ).astype(np.int64)
                    else:
                        # Both are the same
                        new_height = desired_shorter_side
                        new_width = desired_shorter_side
                    assert np.isclose(
                        new_width/new_height,
                        original_width/original_height,
                        0.01
                    ), f'{new_width/new_height}; {original_width/original_height}'
                    frame = cv2.resize(
                        frame,
                        (256, 256),
                        interpolation=cv2.INTER_AREA
                    )
                    frames.append(frame)
                    frame = rearrange(frame, 'h w c -> c h w')
                    frames_temp.append(frame)
                    frame_count += 1
                    if frame_count % 200 == 0 or frame_count == count:
                        frames_temp_npy = np.array(frames_temp)
                        # print(f"Frames size: {frames_temp_npy.shape}")
                        
                        frames_temp_tensor = torch.tensor(frames_temp_npy)
                        # print(f"Frames_conv size: {frames_temp_tensor.shape}")
                        
                        frames_temp_tensor = frames_temp_tensor.to(device).float()
                        frames_temp_tensor, _ = self.res_model(frames_temp_tensor)
                        # print(f"Frames_conv size: {frames_temp_tensor.shape}")
                        frames_temp_tensor = frames_temp_tensor.detach()
                        frames_converted[idx:frame_count, :] = frames_temp_tensor
                        idx = frame_count
                        frames_temp = list()

            current_frame += 1
            
        
        
        videocap.release()
        # print("Check2")
        
        frames_npy = np.array(frames)
        frames_npy_conv = frames_converted.numpy()

        print(f"Frames size: {frames_npy.shape}")
        print(f"Frames_conv size: {frames_npy_conv.shape}")

         
        np.save(npy_file_path, frames_npy_conv)
        print("NPY Dataset Created")

        return 0
    

if __name__ == '__main__':
    
    video_path = "./Dataset/videos"
    anno_path = "./Dataset/annotations"
    feature_path = "./Dataset/features"
    gt_path = "./Dataset/groundTruth"

    obj = npyLoader(video_path, anno_path, feature_path)
    for vid in obj.video_paths:
        #print(vid)
        video_name = vid.split('/')[-1].split('.')[0]
        #print(video_name)

        obj._extract_frames(vid, anno_path, feature_path, gt_path)