import torch
 
from batch_gen_batch import BatchGenerator

import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import logger as logging

import copy
import math
from torchvision import models

seed = 19980928 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--split', default='1')

parser.add_argument('--model_dir', default='models_feature')
parser.add_argument('--result_dir', default='results_feature')
parser.add_argument('--log_dir', default='log_feature')

args = parser.parse_args()
 
num_epochs = 120

learning_rate = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
clip = 90 # no_of_consecutive frames

# batch_input_dim = 196608   # (h w c) = (256*256*3) = 196608

channel_mask_rate = 0.3

sample_rate = 75

vid_list_file = "./Dataset/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./Dataset/splits/test.split"+args.split+".bundle"

features_path = "./Dataset/features/"
gt_path = "./Dataset/groundTruth/"
anno_path = "./Dataset/annotations/"

mapping_file = "./Dataset/mapping.txt"
model_dir = "./{}/".format(args.model_dir)
results_dir = "./{}/".format(args.result_dir)
log_dir = "./{}/".format(args.log_dir)
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

'''
Logging Setup
'''
logger = logging.get_logger(__name__)
logging.setup_logging(output_dir=log_dir)

logger.critical("Video_train_list: {}".format(vid_list_file))
logger.critical("Video_test_list: {}".format(vid_list_file_tst))
logger.critical("Feature path: {}".format(features_path))
logger.critical("GT path: {}".format(gt_path))
logger.critical("Mapping_file path: {}".format(mapping_file))
logger.critical("Model save dir: {}".format(model_dir))
logger.critical("Result save dir: {}".format(results_dir))
logger.critical("Log save dir: {}".format(log_dir))

'''
Read the mapping file and create and action dictionary
'''
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    #print(a)
    row = a.split(' ')
    task = ' '.join(row[1:])
    actions_dict[task] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)
logger.critical("Actions_dict: {}".format(actions_dict))
logger.critical("num_classes: {}".format(num_classes))


batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, anno_path, sample_rate, features_dim)
batch_gen.read_data(vid_list_file)

batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, anno_path, sample_rate, features_dim)
batch_gen_tst.read_data(vid_list_file_tst)

logger.critical("Dataset Created")

class new_model(nn.Module):
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

model = new_model(output_layer = 'layer4')
model = model.to(device)
model = model.train()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
logger.critical('LR:{}'.format(learning_rate))

# If using pre-trained weights
# ckpt_name= "./models_feature/epoch-60-0.5001785714285715.model" 
# ckpt_opt = "./models_feature/epoch-60-0.5001785714285715.opt"
# print('##### Loaded model from epoch 60 with Accuracy 50')

# model.load_state_dict(torch.load(ckpt_name))
# optimizer.load_state_dict(torch.load(ckpt_opt))

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
ce = nn.CrossEntropyLoss(ignore_index=-100)
parameters = sum(p.numel() for p in model.parameters())/1000000
logger.critical('Model Size: {} M' .format(parameters))

mse = nn.MSELoss(reduction='none')

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    total = 0
    best_acc = 0
    best_epoch = 0

    while batch_gen.has_next():
        batch_input, batch_target, vids = batch_gen.next_batch(bz, clip)

        batch_input, batch_target = batch_input.to(device), batch_target.to(device)
        optimizer.zero_grad()

        out = torch.zeros(np.shape(batch_input)[0], np.shape(batch_input)[1], 11, dtype=torch.float).to(device)  # bs, f, num_classes

        for idx, batch in enumerate(batch_input):
            _,output = model(batch)
            out[idx] = output
            del output
        
        loss = 0
        loss += ce(out.contiguous().view(-1, num_classes), batch_target.view(-1))
        loss += 0.15 * torch.mean(torch.clamp(
            mse(F.log_softmax(out[:, 1:], dim=1), F.log_softmax(out.detach()[:, :-1], dim=1)), min=0,
            max=16))

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        out = out.unsqueeze(dim = 0)
        
        _, predicted = torch.max(out.data[-1], 2)
        batch_target_shape = np.shape(batch_target)[0]* np.shape(batch_target)[1]
        correct += ((predicted == batch_target).float()).sum().item()
        total += batch_target_shape

        temp1 = ((predicted == batch_target).float()).sum().item()
        temp2 = batch_target_shape
        print("Train: {}: Correct: {}/{}".format(vids, temp1, temp2))

        batch_input.detach()
        batch_target.detach()
        del batch_input
        del batch_target
    
    scheduler.step(epoch_loss)
    batch_gen.reset()
    
    logger.critical("[epoch {}]: epoch loss = {},   acc = {}" .format(epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                        float(correct) / total))                                                               

    if (epoch + 1) % 10 == 0 and batch_gen_tst is not None:
        model = model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target, vids = batch_gen_tst.next_batch(bz, clip)
                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                out = torch.zeros(np.shape(batch_input)[0], np.shape(batch_input)[1], 11, dtype=torch.float).to(device)  # bs, f, num_classes

                for idx, batch in enumerate(batch_input):
                    _,output = model(batch)
                    out[idx] = output
                    del output

                out = out.unsqueeze(dim = 0)
                _, predicted = torch.max(out.data[-1], 2)

                batch_target_shape = np.shape(batch_target)[0]* np.shape(batch_target)[1]
                correct += ((predicted == batch_target).float()).sum().item()
                total += batch_target_shape

                temp1 = ((predicted == batch_target).float()).sum().item()
                temp2 = batch_target_shape
                print("Test: {}: Correct: {}/{}".format(vids, temp1, temp2))


        acc = float(correct) / total
        logger.critical("---[epoch %d]---: tst acc = %f" % (epoch + 1, acc))

        model = model.train()
        batch_gen_tst.reset()
        torch.save(model.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + "-" + str(acc) + ".model")
        torch.save(optimizer.state_dict(), model_dir + "/epoch-" + str(epoch + 1) + "-" + str(acc) + ".opt")
        logger.critical("Current Test Accuracy: [epoch: {}] is {}".format(str(epoch + 1), str(acc)))

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        logger.critical("Best Test Accuracy: [epoch: {}] is {}".format(str(best_epoch + 1), str(best_acc)))
