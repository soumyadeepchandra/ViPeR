'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
import logger as logging
import pandas as pd
logger = logging.get_logger(__name__)

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, anno_path, sample_rate, features_dim):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.features_dim = features_dim
        self.anno_path = anno_path

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.gts = [self.gt_path + vid.split('.')[0] + '.txt' for vid in self.list_of_examples]
        self.features = [self.features_path + vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.anno = [self.anno_path + vid.split('.')[0] + '.csv' for vid in self.list_of_examples]
        self.my_shuffle()

        logger.critical("List of examples (Video file names): {}".format(len(self.list_of_examples)))
        logger.critical("Ground truth paths: {}".format(len(self.gts)))
        logger.critical("Feature paths of .npy files: {}".format(len(self.features)))

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)
        random.seed(randnum)
        random.shuffle(self.anno)


    def next_batch(self, batch_size, clip_size): 
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            # print("Index: ", idx)
            # Need to convert: [f h w c] -> [(h w c) f]

            features = np.load(batch_features[idx])
            features = rearrange(features, 'f h w c -> f c h w')
                      
            file_ptr = open(batch_gts[idx], 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[0], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            # Downsample the videos 
            samplerate = 15 + random.randint(0, 1)
            if np.shape(features)[0] > 50000:
                samplerate = 30 + random.randint(0, 1)
            if np.shape(features)[0] > 100000: 
                samplerate = 45 + random.randint(0, 1)
            if np.shape(features)[0] > 150000: 
                samplerate = 60 + random.randint(0, 1)
            if np.shape(features)[0] > 200000: 
                samplerate = 75 + random.randint(0, 1)
            if np.shape(features)[0] > 250000: 
                samplerate = 90 + random.randint(0, 1)


            if samplerate == 1:
                feature = features[::samplerate, :]
                target = classes[::samplerate]
            else:  # allows all frames to be learned
                temp = random.randint(0,samplerate-1)
                feature = features[temp::samplerate, :]
                target = classes[temp::samplerate]
            
            batch_input.append(feature)
            batch_target.append(target)
        
        frames = 700
        batch_size_input = np.shape(batch_input[0])[0] // frames

        
        # batch_size_input = 1
        # frames = np.shape(batch_input[0])[0] // batch_size_input

        length_of_sequences = list(map(len, batch_target))
        

        batch_input_tensor = torch.zeros(batch_size_input, frames, np.shape(batch_input[0])[1], np.shape(batch_input[0])[2], np.shape(batch_input[0])[3], dtype=torch.float)  # bs, f, c, h ,w
        batch_target_tensor = torch.ones(batch_size_input, frames, dtype=torch.long) * (-100)

        for i in range(batch_size_input):
            start = i * frames
            stop = (i+1) * frames
            batch_input_tensor[i, :frames, :] = torch.from_numpy(batch_input[0][start:stop])
            batch_target_tensor[i, :frames] = torch.from_numpy(batch_target[0][start:stop])
           

        return batch_input_tensor, batch_target_tensor, batch

   


if __name__ == '__main__':
    pass